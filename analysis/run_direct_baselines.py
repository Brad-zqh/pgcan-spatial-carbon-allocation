"""Additional direct-share baselines, avoiding dependence on a log-ratio head.

Same spatial splits, covariates and train-only neighbour aggregation as the main
experiment. Both prediction heads are reported; the better head is not selected
using test outcomes. This direct-share family is the prespecified main comparator.
"""
import os
import run_model_benchmark as b
from pathlib import Path
import numpy as np
import pandas as pd
import json,time
from scipy.spatial import cKDTree
from scipy.special import softmax

def fit(name,train,seed):
 t,*_=b.prepare(train,seed)
 z=np.c_[t[0].numpy(),t[2].numpy(),t[3].numpy(),t[1].numpy(),b.p0*b.n]
 y=b.target[train]*b.n
 models={
 'Ridge':lambda:b.Ridge(alpha=1.),
 'Random_forest':lambda:b.RandomForestRegressor(n_estimators=220,min_samples_leaf=3,max_features=.8,n_jobs=1,random_state=seed),
 'Extra_trees':lambda:b.ExtraTreesRegressor(n_estimators=220,min_samples_leaf=3,max_features=.8,n_jobs=1,random_state=seed),
 'Hist_gradient':lambda:b.HistGradientBoostingRegressor(max_iter=220,max_leaf_nodes=15,l2_regularization=1.,early_stopping=False,random_state=seed),
 'XGBoost':lambda:b.XGBRegressor(n_estimators=250,max_depth=3,learning_rate=.04,subsample=.85,colsample_bytree=.85,reg_lambda=3,n_jobs=1,random_state=seed),
 'MLP':lambda:b.MLPRegressor(hidden_layer_sizes=(48,24),alpha=.01,max_iter=700,early_stopping=False,random_state=seed,tol=1e-5)}
 if name=='Local_ridge':
  dd,ii=cKDTree(b.xy[train]).query(b.xy,k=min(128,len(train)))
  f=np.c_[np.ones(b.n),z[:,:9],b.p0*b.n];out=np.empty(b.n)
  for j in range(b.n):
   ix=train[ii[j]];w=np.exp(-.5*(dd[j]/max(np.median(dd[j]),1))**2)
   a=f[ix];pen=np.eye(f.shape[1])*.1;pen[0,0]=1e-6
   beta=np.linalg.solve((a.T*w)@a+pen,(a.T*w)@y[ii[j]])
   out[j]=f[j]@beta
 else:
  model=models[name]();model.fit(z[train],y);out=model.predict(z)
 return np.maximum(out/b.n,1e-12)

def main():
 splits=json.loads((b.OUT/'split_manifest.json').read_text());rows=[];maprows=[];start=time.perf_counter()
 for family in b.CLASSICS:
  name=family+'_direct';seeds=b.cfg.seeds[:1] if family in ['Ridge','Local_ridge'] else b.cfg.seeds
  oofs=[];full=[]
  for seed in seeds:
   oof=np.zeros(b.n)
   for s in splits:
    train=np.array(s['train_indices']);test=np.array(s['test_indices']);tick=time.perf_counter()
    score=fit(family,train,seed)
    pred=score*(b.target[train].sum()/score[train].sum())
    oof[test]=pred[test]
    rows.append({'method':name,'seed':seed,'fold':s['fold'],'n_train':len(train),'n_test':len(test),
     'epochs':0,'elapsed_seconds':time.perf_counter()-tick,**b.metrics(b.target[test],pred[test])})
    print('DIRECT_CV',name,seed,s['fold'],'RMSE_pp',round(rows[-1]['proxy_rmse_pp'],6),flush=True)
   oofs.append(oof)
   maprows.append({'method':name,'seed':seed,'scope':'spatial_oof',**b.metrics(b.target,oof),**b.screen(oof)})
   tick=time.perf_counter();pred=b.m.normalize(fit(family,np.arange(b.n),seed));full.append(pred)
   maprows.append({'method':name,'seed':seed,'scope':'full_domain_refit','elapsed_seconds':time.perf_counter()-tick,
    **b.metrics(b.target,pred),**b.screen(pred)})
  np.save(b.OUT/f'oof_{name}.npy',oofs);np.save(b.OUT/f'full_{name}.npy',full)
  pd.DataFrame(rows).to_csv(b.SD/'Benchmark_direct_fold_metrics.csv',index=False)
  pd.DataFrame(maprows).to_csv(b.SD/'Benchmark_direct_map_metrics.csv',index=False)
 (b.OUT/'direct_baseline_protocol.json').write_text(json.dumps({'completed':True,'prediction_head':'Nonnegative direct shares calibrated on training target mass',
  'additional_input':'Declared prior share; available to both baseline and PGCAN families',
  'families':b.CLASSICS,'elapsed_seconds':time.perf_counter()-start},indent=2),encoding='utf-8')
 print('DIRECT_BASELINES_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
