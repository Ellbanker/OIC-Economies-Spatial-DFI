#ahmed alakrawi
#!/usr/bin/env python3
"""Reviewer 1 final: pre-2024 PCA refit and country-clustered score diagnostics.

All calculations use original author-supplied inputs and explicitly state assumptions.
"""
from pathlib import Path
import sys,csv, json, math, copy
import numpy as np
from scipy import stats
from statsmodels.tools.numdiff import approx_hess
ROOT=Path(__file__).resolve().parent
SP=ROOT
sys.path.insert(0,str(SP))
from run_oic_analysis import load_panel,enrich_with_supplement,make_mundlak,read_W_csv,spatial_design
from spatial_random_effects_ml import fit_cre_spatial
PCA=ROOT/'data/OIC_Data_Cleaned_Imputed.csv'
V=['Mobile_Agents_per_100k','ATMs_per_100k','Internet_Penetration','Mobile_Accounts_per_1k','Mobile_Transactions_per_1k','Value_of_Transactions_GDP']
OUT=ROOT/'results';OUT.mkdir(parents=True,exist_ok=True)

def alternate_pca():
  with PCA.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
  X=np.array([[float(r[v]) for v in V] for r in rows]);yr=np.array([int(r['Year']) for r in rows]); ref=X[yr<=2023]
  mu=ref.mean(axis=0);sd=ref.std(axis=0,ddof=0)
  zz=(ref-mu)/sd;zzall=(X-mu)/sd
  eig,Q=np.linalg.eigh(zz.T@zz/(len(zz)-1));ind=np.argsort(eig)[::-1];eig=eig[ind];Q=Q[:,ind]
  keep=eig>1;load=(Q[:,:sum(keep)]*np.sqrt(eig[:sum(keep)]))**2
  w=(load/load.sum(axis=0))@(eig[keep]/eig[keep].sum())
  raw=zzall@w;minraw=(zz@w).min();maxraw=(zz@w).max()
  alt=100*(raw-minraw)/(maxraw-minraw)
  return dict(zip(((r['Country'],int(r['Year'])) for r in rows),alt.tolist())),w


def calculate_scores(y,X,names,W,params):
    N=W.shape[0];T=len(y)//N;n=N*T;K=X.shape[1]
    rho=params['rho'];sig_e2=params['sigma_e2'];s=params['s_log_total_to_idio'];beta=np.array([next(p['estimate'] for p in params['parameters'] if p['term']==name) for name in names])
    Y=y.reshape(T,N);XX=X.reshape(T,N,K);WY=Y@W.T
    R=Y-rho*WY-np.einsum('tnk,k->tn',XX,beta)
    rw=R-R.mean(axis=0)[None,:]
    rm=R.mean(axis=0)*np.sqrt(T)
    xw=XX-XX.mean(axis=0)[None,:,:]
    xm=XX.mean(axis=0)*np.sqrt(T)
    yww=WY-WY.mean(axis=0)[None,:]
    ywm=WY.mean(axis=0)*np.sqrt(T)
    z=np.exp(-s)
    A=np.eye(N)-rho*W
    trace=np.trace(np.linalg.solve(A,W))
    scr=np.empty((N,K+3))
    scr[:,0]=-(T/N)*trace+(np.sum(yww*rw,axis=0)+z*ywm*rm)/sig_e2
    scr[:,1:1+K]=(np.einsum('tnk,tn->nk',xw,rw)+z*xm*rm[:,None])/sig_e2
    quad=np.sum(rw**2,axis=0)+z*rm**2
    scr[:,1+K]=-T/2+quad/(2*sig_e2)
    scr[:,2+K]=-1/2+z*rm**2/(2*sig_e2)
    return scr,R

def robust_se(y,X,names,W,res):
    N=W.shape[0];T=len(y)//N;n=N*T;K=X.shape[1]
    p=res['parameters'];paramnames=['rho_W_ln_GDPpc']+names+['log_sigma_e2','log_sigma_total_to_idiosyncratic_ratio']
    par=np.array([next(x['estimate'] for x in p if x['term']==q) for q in paramnames]); rho=par[0];beta=par[1:-2];loge=par[-2];s=par[-1]
    Y=y.reshape(T,N); WY=Y@W.T;XX=X.reshape(T,N,K)
    def nll(q):
        r=q[0];b=q[1:-2];le=q[-2];ss=q[-1]
        if abs(r)>=.999 or ss<0:return 1e100
        A=np.eye(N)-r*W;sign,logdet=np.linalg.slogdet(A)
        if sign<=0:return 1e100
        E=Y-r*WY-np.einsum('tnk,k->tn',XX,b)
        Ew=E-E.mean(axis=0,keepdims=True)
        Eb=np.sqrt(T)*E.mean(axis=0)
        quad=(Ew**2).sum()+np.exp(-ss)*(Eb**2).sum()
        return -(T*logdet-.5*(n*(math.log(2*math.pi)+le)+N*ss+np.exp(-le)*quad))
    H=approx_hess(par,nll)
    sc,R=calculate_scores(y,X,names,W,res)
    ev=np.linalg.eigvalsh(H)
    assert ev.min()>0,(ev.min(),ev.max())
    Ainv=np.linalg.inv(H)
    cov=Ainv@sc.T@sc@Ainv
    cov*=N/(N-1)
    est=par;se=np.sqrt(np.diag(cov))
    pvals=2*stats.t.sf(abs(est/se),df=N-1)
    ans={key:{'estimate':float(est[i]),'se_country_score':float(se[i]),'p_t36_exploratory':float(pvals[i]),'ci95_t36':[float(est[i]-stats.t.ppf(.975,N-1)*se[i]),float(est[i]+stats.t.ppf(.975,N-1)*se[i])]} for i,key in enumerate(paramnames)}
    check={'score_sum_max_abs':float(abs(sc.sum(axis=0)).max()),'hessian_condition_number':float(np.linalg.cond(H)),'bread_eigen_min':float(ev.min()),'n_countries':N,'limitations':'Quasi-ML country-score sandwich; handles serial dependence within transformed country disturbances conditionally; spatially coupled scores and errors across countries mean not a fully spatial-HAC-robust method.'}
    return ans,check,R

if __name__=='__main__':
    records,countries,_=load_panel(SP/'source_inputs/OIC_Input_37x7.csv')
    enrich_with_supplement(records,SP/'income_supplement_37x7.csv')
    W=read_W_csv(SP/'W_primary_2015_2023_available.csv',countries)
    make_mundlak(records,countries,include_income=True)
    y,X,names=spatial_design(records,countries,W,include_year=True)
    original=json.loads((SP/'primary_2015_2023_available_RE_results/spatial_panel_RE_ML.json').read_text())
    scores,diag,R=robust_se(y,X,names,W,original)
    alt,w=alternate_pca()
    alt_records=copy.deepcopy(records)
    d=[]
    for r in alt_records:
      alt_value=float(alt[(r['Country'],r['Year'])]);d.append(alt_value-r['DFI_Index']);r['DFI_Index']=alt_value
    make_mundlak(alt_records,countries,include_income=True)
    ya,Xa,anames=spatial_design(alt_records,countries,W,include_year=True)
    assert names==anames and np.array_equal(y,ya)
    fit_alt=fit_cre_spatial(ya,Xa,anames,W)
    alt_rob,alt_diag,alt_R=robust_se(ya,Xa,anames,W,fit_alt)
    result={'alternative_pca_weights':dict(zip(V,w.tolist())),'alternative_original_index_differences':{'mean_abs':float(np.mean(abs(np.array(d)))),'max_abs':float(np.max(abs(np.array(d))))},'original_primary_ml':{'rho':original['rho'],'params':original['parameters'],'country_score_sandwich':scores,'diagnostics':diag},'pre2024_pca_ml':{'rho':fit_alt['rho'],'params':fit_alt['parameters'],'country_score_sandwich':alt_rob,'diagnostics':alt_diag,'residual_adjacent_year_corr':fit_alt['quasi_demeaned_adjacent_year_residual_correlation']},'notes':'PCA sensitivity refits 2015-2023 weights and normalization from 360 source rows, retains 259 country-year regression observations and same W. Country-score SE diagnostic assumes independent structural error paths across countries conditional on W and regressors; this assumption is NOT verified.'}
    (OUT/'pca_reference_window_regression_and_score_sandwich.json').write_text(json.dumps(result,indent=2))
    print('ORIGINAL score-sandwich diagnostics',diag)
    print('DFI REFIT weights',dict(zip(V,w)))
    print('diff meanmax',result['alternative_original_index_differences'])
    for name in ['rho_W_ln_GDPpc','DFI_within','W_DFI_within','DFI_between','Gov_within','Gov_between','Income_Level']:
      o=scores[name];a=alt_rob[name]
      print(name,'old',f'{o["estimate"]:.6f}',f'{o["se_country_score"]:.6f}',f'{o["p_t36_exploratory"]:.5g}','alt',f'{a["estimate"]:.6f}',f'{a["se_country_score"]:.6f}',f'{a["p_t36_exploratory"]:.5g}')
    print('ALT fit LL',fit_alt['log_likelihood'],'residual_adjacent_corr',fit_alt['quasi_demeaned_adjacent_year_residual_correlation'])
