#ahmed alakrawi
#!/usr/bin/env python3
"""Transparent, descriptive model diagnostics, not a substitute for robust panel inference."""
import json,csv,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_oic_analysis import load_panel,enrich_with_supplement,read_W_csv,make_mundlak,spatial_design
base=Path(__file__).resolve().parent
records,countries,_=load_panel(base/'source_inputs/OIC_Input_37x7.csv')
enrich_with_supplement(records,base/'income_supplement_37x7.csv')
for name in ['primary_2015_2023_available','sensitivity_2017_2023_available','sensitivity_2017_2021_common','sensitivity_zscore_2015_2023']:
  wpath=base/f'W_{name}.csv';W=read_W_csv(wpath,countries)
  model=json.loads((base/('primary_results' if name.startswith('primary') else f'{name}_results')/'spatial_ML_results.json').read_text())
  make_mundlak(records,countries,True)
  y,X,xnames=spatial_design(records,countries,W)
  p={r['term']:r['estimate'] for r in model['spatial_parameters']}
  b=np.array([p[n] for n in xnames]); rho=p['rho_W_ln_GDPpc']
  resid=(y-rho*(y.reshape(7,37)@W.T).reshape(-1)-X@b).reshape(7,37)
  within_country_lag1=float(np.corrcoef(resid[:-1].ravel(),resid[1:].ravel())[0,1])
  country_avg=np.mean(resid,axis=0)
  sse=float(np.sum(resid**2))
  mean_amplitude=float(np.sum(country_avg**2)*7/sse)
  # standardized row-dependent Moran-like weighted cross-moment; descriptive only
  moran=[]
  for ti in range(7):
   e=resid[ti]-resid[ti].mean()
   moran.append(float(e@W@e/(e@e)))
  print(name, 'Residual year-adjacent correlation:',round(within_country_lag1,4),
        'between share squared residual:',round(mean_amplitude,4),
        'Moran annual:', [round(x,3) for x in moran])
  if name.startswith('primary'):
    out={'residual_year_adjacent_correlation':within_country_lag1,
         'residual_country_mean_share_of_total_sq':mean_amplitude,
         'moran_like_residual_correlation_by_year':dict(zip(range(2017,2024),moran)),
         'note':'Descriptive checks only; not robust tests. Positive residual serial dependence would violate iid standard-error assumptions.'}
    (base/'primary_results'/'residual_diagnostics.json').write_text(json.dumps(out,indent=2)+'\n')
