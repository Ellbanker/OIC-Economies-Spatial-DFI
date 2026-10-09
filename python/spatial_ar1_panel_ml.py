#ahmed alakrawi
#!/usr/bin/env python3
"""Gaussian spatial Mundlak panel ML with country random intercept + AR(1) disturbance.

Time-major panel: y_t = rho W y_t + X_t b + a_i + epsilon_it.
Across i, a_i ~ iid N(0,sigma_a2); epsilon_i has stationary AR(1)
Cov(epsilon_it, epsilon_is) = sigma_e2 * phi**abs(t-s).

Conditional on fixed/exogenous W and exogenous X incl Mundlak means.
Year fixed effects in X. Does NOT make DFI effects causal; alternatives
may yield other estimates and error-model uncertainty remains substantial.
"""
from pathlib import Path
import numpy as np,json, math
from scipy import optimize,stats
from statsmodels.tools.numdiff import approx_hess
from run_oic_analysis import (load_panel,enrich_with_supplement,make_mundlak,
                              read_W_csv,spatial_design,impacts,write_csv,write_json)
ROOT=Path(__file__).resolve().parent

def fit_ar1_spatial(y,X,names,W,T=7, include_hessian=True):
    N=W.shape[0];n=N*T;K=X.shape[1];Eye=np.eye(N)
    Y=y.reshape((T,N));WY=Y@W.T
    Xt=X.reshape((T,N,K))
    tt=np.arange(T)
    lag=np.abs(tt[:,None]-tt[None,:])

    def common(rho,phi,s):
        if not(-.999<rho<.999 and -.985<phi<.985 and 0<=s<23):return None
        a=Eye-rho*W
        signA,ldA=np.linalg.slogdet(a)
        if signA<=0:return None
        ratio=np.expm1(s)/T
        Kmat=np.power(phi,lag).astype(float)+ratio*np.ones((T,T))
        signK,ldK=np.linalg.slogdet(Kmat)
        if signK<=0:return None
        Qi=np.linalg.inv(Kmat)
        return float(ldA),float(ldK),Qi

    def profile(rho,phi,s):
        val=common(rho,phi,s)
        if val is None:return -1e80,None
        ldA,ldK,Qi=val
        Ystar=Y-rho*WY
        XX=np.einsum('tik,ts,sil->kl',Xt,Qi,Xt,optimize=True)
        XY=np.einsum('tik,ts,si->k',Xt,Qi,Ystar,optimize=True)
        try:beta=np.linalg.solve(XX,XY)
        except np.linalg.LinAlgError:return -1e80,None
        res=Ystar-np.einsum('tik,k->ti',Xt,beta)
        SSR=float(np.einsum('ti,ts,si->',res,Qi,res,optimize=True))
        sig2=SSR/n
        if sig2<=0:return -1e80,None
        ll=T*ldA-(N/2)*ldK-(n/2)*(math.log(2*math.pi)+math.log(sig2)+1)
        return float(ll),{'rho':rho,'phi':phi,'s':s,'beta':beta,'sigma_e2':sig2,
                          'sigma_a2':sig2*np.expm1(s)/T,'log_likelihood':float(ll),'resid':res}

    starts=[(-.2,.7,5),(-.2,.8,5),(.0,.6,5),(.1,.8,5),(-.6,.6,4),(-.2,.3,4),(-.4,.9,6),(-.2,0.,5),(0.,0.,0.)]
    candidates=[]
    for st in starts:
        m=optimize.minimize(lambda a:-profile(*map(float,a))[0],st,bounds=[(-.98,.98),(-.97,.97),(0,20)],method='L-BFGS-B',
                            options={'maxiter':500,'ftol':3e-12,'gtol':1e-8,'maxls':35})
        if np.isfinite(m.fun): candidates.append(m)
    if not candidates:raise RuntimeError('No converged candidate')
    winner=min(candidates,key=lambda z:z.fun)
    ll,part=profile(*map(float,winner.x))
    if not np.isfinite(ll):raise RuntimeError('Bad optimum')
    rho=float(part['rho']);phi=float(part['phi']);s=float(part['s']);
    beta=part['beta'];loge=float(math.log(part['sigma_e2']))

    # Hessian with joint beta and variance parameters at profile optimum.
    zhat=np.r_[rho,phi,beta,loge,s]
    def nll(z):
        r=float(z[0]);ph=float(z[1]);b=z[2:-2];loges=float(z[-2]);ss=float(z[-1])
        val=common(r,ph,ss)
        if val is None:return 1e90
        lda,ldk,q=val
        res=Y-r*WY-np.einsum('tik,k->ti',Xt,b)
        S=float(np.einsum('ti,ts,si->',res,q,res,optimize=True))
        return -(T*lda-N*ldk/2-n*(math.log(2*math.pi)+loges)/2-math.exp(-loges)*S/2)
    covariance=None
    se=np.full(len(zhat),np.nan)
    se_note='No standard errors calculated'
    if include_hessian:
        hess=approx_hess(zhat,nll)
        eig=np.linalg.eigvalsh(hess)
        if eig.min()>0 and np.linalg.cond(hess)<1e13:
            covariance=np.linalg.inv(hess)
            se=np.sqrt(np.diag(covariance))
            se_note='Observed Hessian model-based Gaussian ML (NOT cluster-robust)'
        else:se_note=f'Cannot calculate model-based SE; Hessian min eig={eig.min()} cond={np.linalg.cond(hess)}'
    parnames=['rho_W_ln_GDPpc','phi_AR1']+names+['log_sigma_e2','log_one_plus_T_sigma_a2_over_sigma_e2']
    rows=[]
    for term,zval,sd in zip(parnames,zhat,se):
        sd=float(sd) if np.isfinite(sd) and sd>0 else None
        rows.append({'term':term,'estimate':float(zval),'std_error':sd,
                     'p_value':float(2*stats.norm.sf(abs(zval/sd))) if sd else None,
                     'ci_low':float(zval-1.96*sd) if sd else None,
                     'ci_high':float(zval+1.96*sd) if sd else None})
    _,_,qi=common(rho,phi,s)
    l=np.linalg.cholesky(np.linalg.inv(qi))
    whitened=np.linalg.solve(l,part['resid'])
    diagnostics={'whitened_residual_adjacent_year_corr':float(np.corrcoef(whitened[:-1].ravel(),whitened[1:].ravel())[0,1]),
                 'loglik_profile_joints_differ':abs(float(ll)-float(-nll(zhat))),
                 'optimizer_success':winner.success,
                 'objective_min':float(min(c.fun for c in candidates)),'objective_max':float(max(c.fun for c in candidates))}
    result={'n':n,'n_countries':N,'n_years':T,'rho':rho,'phi':phi,'sigma_a2':part['sigma_a2'],
            'sigma_e2':part['sigma_e2'],'log_likelihood':ll,
            'likelihood':'spatial-lag gaussian ML with a_i random intercept + stationary AR1 idiosyncratic disturbance',
            'model_standard_errors':se_note,
            'W_fixed_exogenous_assumed':True,'parameters':rows,'diagnostics':diagnostics}
    if covariance is not None:
        d={p['term']:p['estimate'] for p in rows}
        result['impacts']=impacts(rho,d['DFI_within'],d['W_DFI_within'],d['Gov_within'],W,covariance,parnames)
    return result

if __name__=='__main__':
    variants={'primary_2015_2023_available':'W_primary_2015_2023_available.csv',
              'sensitivity_2017_2023_available':'W_sensitivity_2017_2023_available.csv',
              'sensitivity_2017_2021_common':'W_sensitivity_2017_2021_common.csv',
              'sensitivity_zscore_2015_2023':'W_sensitivity_zscore_2015_2023.csv'}
    records,countries,_=load_panel(ROOT/'source_inputs/OIC_Input_37x7.csv')
    enrich_with_supplement(records,ROOT/'income_supplement_37x7.csv')
    make_mundlak(records,countries,True)
    for tag,fn in variants.items():
        W=read_W_csv(ROOT/fn,countries)
        y,X,names=spatial_design(records,countries,W,include_year=True)
        result=fit_ar1_spatial(y,X,names,W,include_hessian=True)
        path=ROOT/f'{tag}_AR1_results';path.mkdir(exist_ok=True)
        write_json(path/'spatial_panel_AR1_results.json',result)
        write_csv(path/'spatial_panel_AR1_coefficients.csv',result['parameters'])
        if 'impacts' in result:write_csv(path/'spatial_panel_AR1_impacts.csv',result['impacts'])
        print(tag,'ll',round(result['log_likelihood'],3),'rho',round(result['rho'],4),'phi',round(result['phi'],4),'\n  sigma_a2',round(result['sigma_a2'],5),'sigma_e2',round(result['sigma_e2'],5),'whiteCorr',round(result['diagnostics']['whitened_residual_adjacent_year_corr'],4),'optim',result['diagnostics']['optimizer_success'])
        for t in ['rho_W_ln_GDPpc','DFI_within','W_DFI_within','Gov_within','Gov_between','Income_Level']:
            z=next(c for c in result['parameters'] if c['term']==t)
            print(' ',t,round(z['estimate'],7),'p',z['p_value'])
