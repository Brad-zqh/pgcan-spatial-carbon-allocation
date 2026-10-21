from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
import allocation_core as m
p=argparse.ArgumentParser();p.add_argument('--inputs',required=True,type=Path);p.add_argument('--counts',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True);m.OUT=a.output
d,labels,prior,activity,target,xy=m.prepare(pd.read_csv(a.inputs))
c=pd.read_csv(a.counts).set_index('grid_id').reindex(d.grid_id)
assert not c[['primary_count','secondary_count']].isna().any().any()
counts={k:c[k+'_count'].to_numpy() for k in ['primary','secondary']}
rows=[];maps={}
for key,v in [('Nightlights',m.normalize(d.ntl_sum)),('Enterprise_POI',m.normalize(d.enterprise_poi_density)),('Activity',activity),('Clean_prior',prior),('Direct_mixture',target)]:
 maps[key]=v;rows.append(dict(method=key,seed=0,**m.evaluate(d,v,counts)))
variants={'Full':{},'No_context':{'fun':False},'No_graph_features':{'gra':False},'No_graph_penalty':{'penalty':False},'No_graph':{'gra':False,'penalty':False},'Matched_postnorm':{'output':'postnorm'},'Matched_softmax':{'output':'softmax'}}
for key,opts in variants.items():
 ps=[]
 for seed in m.SEEDS:
  v,history=m.train(d,labels,prior,target,xy,seed,save=f'{key}_{seed}',**opts);ps.append(v);rows.append(dict(method=key,seed=seed,**m.evaluate(d,v,counts)))
  (a.output/f'{key}_{seed}.json').write_text(json.dumps(history))
 maps[key]=np.mean(ps,axis=0);np.save(a.output/f'probabilities_{key}.npy',ps)
pd.DataFrame(rows).to_csv(a.output/'metrics_runs.csv',index=False)
pd.DataFrame({'grid_id':d.grid_id,**{k+'_share':v for k,v in maps.items()}}).to_csv(a.output/'allocation_shares.csv',index=False)
