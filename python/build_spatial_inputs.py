#ahmed alakrawi
#!/usr/bin/env python3
"""Recover manuscript's income classification and trade-openness W with source provenance.

Inputs supplied by author on 2026-10-09:
  - income level/New Microsoft Word Document.docx inside income level.zip
  - OIC_Trade_Openness_2015_2023.csv inside Trade (% of GDP).rar

Income group: author's reported modal historical World Bank class, 2015-2023.
Trade W primary: 2015-2023 available-case country means, no imputation.
Trade W sensitivity: 2017-2023 available-case and 2017-2021 complete-year means.
An author assertion about the income group is not independently validated
against the underlying year-by-year historical classification workbook.
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree
import numpy as np
import sys

ROOT=Path(__file__).resolve().parent
INPUT_TRADE=ROOT/'source_inputs'/'OIC_Trade_Openness_2015_2023.csv'
INPUT_INCOME=ROOT/'source_inputs'/'New Microsoft Word Document.docx'
PANEL=ROOT/'source_inputs'/'OIC_Input_37x7.csv'
WORLD_BANK_TRADE='https://data.worldbank.org/indicator/NE.TRD.GNFS.ZS'
WORLD_BANK_INCOME='https://datahelpdesk.worldbank.org/knowledgebase/articles/906519-world-bank-country-and-lending-groups'
EPSILON=1e-5

def read_docx_text(path):
    with ZipFile(path) as z:
        root=ElementTree.fromstring(z.read('word/document.xml'))
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    return ' '.join(''.join(t.text or '' for t in p.findall('.//w:t',ns))
                    for p in root.findall('.//w:p',ns))

def parse_income_groups(text):
    pat1=r'include 11 countries:\s*(.*?)\.\s*Conversely'
    pat0=r'comprising 26 countries:\s*(.*?)\.\s*The historical'
    bycode={}
    for code,pattern,count in [(1,pat1,11),(0,pat0,26)]:
        match=re.search(pattern,text,flags=re.I|re.S)
        if not match:
            raise ValueError('Cannot locate exact income list in author document')
        line=match.group(1).replace(', and the ', ', ').replace(', and ', ', ')
        names=[x.strip().removeprefix('the ') for x in line.split(',') if x.strip()]
        if len(names)!=count: raise ValueError(f'Income class {code} expected {count}, got {len(names)}: {names}')
        for c in names:
            if c in bycode: raise ValueError('Duplicate income country '+c)
            bycode[c]=code
    return bycode

def save_csv(path, rows, cols):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def read_panel():
    with PANEL.open(newline='',encoding='utf-8') as f:
        vals=list(csv.DictReader(f))
    countries=sorted(set(r['Country'] for r in vals))
    pairs={(r['Country'],int(r['Year'])) for r in vals}
    assert len(vals)==259 and len(countries)==37 and len(pairs)==259
    return vals,countries

def compute_W(mean_map,incomes,countries,standardize=False):
    m=np.array([mean_map[c] for c in countries],float)
    inc=np.array([incomes[c] for c in countries],float)
    coords=np.column_stack((m,inc))
    if standardize:
        sd=coords.std(axis=0,ddof=1)
        coords=(coords-coords.mean(axis=0))/sd
    pair=coords[:,None,:]-coords[None,:,:]
    d=np.sqrt((pair**2).sum(axis=2))
    a=1/(d+EPSILON)
    np.fill_diagonal(a,0.0)
    W=a/a.sum(axis=1,keepdims=True)
    assert np.isfinite(W).all() and np.allclose(W.sum(axis=1),1) and np.allclose(np.diag(W),0)
    return W

def run():
    ROOT.mkdir(exist_ok=True,parents=True)
    panel,countries=read_panel()
    incomes=parse_income_groups(read_docx_text(INPUT_INCOME))
    assert set(incomes)==set(countries),{'in_income_not_in_panel':set(incomes)-set(countries),'in_panel_not_in_income':set(countries)-set(incomes)}
    assert sum(incomes.values())==11
    with INPUT_TRADE.open(newline='',encoding='utf-8-sig') as f: trade=list(csv.DictReader(f))
    by=defaultdict(dict)
    for row in trade:
        c,year=row['Country'],int(row['Year'])
        if year in by[c]: raise ValueError(f'Duplicate trade row {c}/{year}')
        s=row['Trade_Openness_pct'].strip()
        by[c][year]=float(s) if s else None
    assert set(countries).issubset(by)
    assert all(set(by[c])==set(range(2015,2024)) for c in countries)
    periods={'primary_2015_2023_available':list(range(2015,2024)),
             'sensitivity_2017_2023_available':list(range(2017,2024)),
             'sensitivity_2017_2021_common':list(range(2017,2022))}
    means={}
    counts={}
    for key,period in periods.items():
        means[key]={c:float(np.mean([by[c][y] for y in period if by[c][y] is not None])) for c in countries}
        counts[key]={c:sum(by[c][y] is not None for y in period) for c in countries}
        assert all(counts[key][c]>=1 for c in countries)
    assert all(counts['sensitivity_2017_2021_common'][c]==5 for c in countries)
    wstats={}
    for key,means_country in means.items():
        W=compute_W(means_country,incomes,countries)
        path=ROOT/f'W_{key}.csv'
        save_csv(path,[{'Country':countries[i],**{countries[j]:float(W[i,j]) for j in range(37)}} for i in range(37)],['Country']+countries)
        wstats[key]={'matrix_file':path.name,'max_weight':float(W.max()),'median_max_row_weight':float(np.median(W.max(axis=1))),'max_abs_row_sum_error':float(np.max(abs(W.sum(axis=1)-1))),'year_observations_min':min(counts[key].values()),'year_observations_max':max(counts[key].values())}
    zs=compute_W(means['primary_2015_2023_available'],incomes,countries,standardize=True)
    save_csv(ROOT/'W_sensitivity_zscore_2015_2023.csv',[{'Country':countries[i],**{countries[j]:float(zs[i,j]) for j in range(37)}} for i in range(37)], ['Country']+countries)
    wstats['sensitivity_zscore_2015_2023']={'matrix_file':'W_sensitivity_zscore_2015_2023.csv','max_weight':float(zs.max()),'median_max_row_weight':float(np.median(zs.max(axis=1))),'max_abs_row_sum_error':float(np.max(abs(zs.sum(axis=1)-1)))}
    supp=[{'Country':r['Country'],'Year':r['Year'],'Income_Level':incomes[r['Country']]} for r in panel]
    save_csv(ROOT/'income_supplement_37x7.csv',supp,['Country','Year','Income_Level'])
    annotated=[]
    for r in panel:
        c,year=r['Country'],int(r['Year'])
        annotated.append({**r,'Income_Level':incomes[c],
                          'Trade_Op_pct':by[c][year] if by[c][year] is not None else '',
                          'Trade_Missing':int(by[c][year] is None)})
    save_csv(ROOT/'combined_panel_with_trade_income.csv',annotated,list(annotated[0]))
    summary=[]
    for c in countries:
        missing=[y for y in range(2015,2024) if by[c][y] is None]
        summary.append({'Country':c,'Income_Level':incomes[c],
                        'Mean_Trade_2015_2023_Available':means['primary_2015_2023_available'][c],
                        'N_2015_2023':counts['primary_2015_2023_available'][c],
                        'Mean_Trade_2017_2023_Available':means['sensitivity_2017_2023_available'][c],
                        'N_2017_2023':counts['sensitivity_2017_2023_available'][c],
                        'Mean_Trade_2017_2021_Common':means['sensitivity_2017_2021_common'][c],
                        'N_2017_2021':counts['sensitivity_2017_2021_common'][c],
                        'Missing_Trade_Years':';'.join(map(str,missing)),
                        'Trade_Source':WORLD_BANK_TRADE,'Income_Class_Source':WORLD_BANK_INCOME})
    save_csv(ROOT/'spatial_weight_country_inputs.csv',summary,list(summary[0]))
    missing_count=sum(by[c][year] is None for c in countries for year in range(2017,2024))
    meta={'n_countries':len(countries),'n_panel':len(panel),'income_1_countries':sorted(c for c in countries if incomes[c]==1),
          'income_0_countries':sorted(c for c in countries if incomes[c]==0),
          'trade_missing_2017_2023':missing_count,'trade_missing_2015_2023':sum(by[c][y] is None for c in countries for y in range(2015,2024)),
          'trade_average_method':'arithmetic mean of NONMISSING observations in the stated window; no imputation',
          'income_method':'author-reported modal World Bank category over 2015-2023; class coded 1 for high/upper-middle, 0 otherwise',
          'income_yearly_classification_underlying_file_supplied':False,
          'income_assignments_independently_verified':False,
          'source_trade_url':WORLD_BANK_TRADE,'source_income_url':WORLD_BANK_INCOME,
          'weights':'W_ij = (1 / (sqrt((mean_trade_i-mean_trade_j)^2+(income_i-income_j)^2) + 1e-5)) normalized by row for i!=j; W_ii=0',
          'W_variations':wstats}
    (ROOT/'reconstruction_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Dataset:',len(countries),'countries',len(panel),'observations')
    print('Income:',sum(incomes.values()),'high/upper-middle and',sum(v==0 for v in incomes.values()),'low/lower-middle')
    print('Missing trade values, 2015-2023:',meta['trade_missing_2015_2023'])
    print('Missing trade values, 2017-2023:',missing_count)
    for name,st in wstats.items():print('Matrix',name,'maximum weight',round(st['max_weight'],5),'median max-per-row weight',round(st['median_max_row_weight'],5))
    return meta
if __name__=='__main__':run()
