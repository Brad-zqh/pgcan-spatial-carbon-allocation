from pathlib import Path
import numpy as np,pandas as pd
P=Path(__file__).resolve().parents[1]/'Supplementary_Data'
d=pd.read_csv(P/'Source_Data_S12_grid_shares.csv');columns=[c for c in d if c.endswith('_share')]
assert len(d)==1980 and d.grid_id.is_unique
for c in columns:
 p=d[c].to_numpy();assert (p>=0).all();assert abs(p.sum()-1)<1e-10
 a=np.lexsort((d.grid_id.to_numpy(str),-p))[:198]
 print(c,'hits',int(d.iloc[a].primary_count.sum()),'top10_mass',p[a].sum())
for key in ['Clean_prior','Direct_mixture']:
 p=d.Full_share.to_numpy();q=d[key+'_share'].to_numpy();print('TV Full vs',key,abs(p-q).sum()/2)
print('Source-data checks passed')
