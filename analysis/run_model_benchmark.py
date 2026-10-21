"""Spatially buffered proxy-reconstruction benchmarks and controlled graph ablations.

Targets are deterministic predictor-derived allocation shares, NOT measured carbon.
Benchmark site membership NEVER enters preparation, fitting or epoch selection.
Training messages come only from training nodes; queries cannot message training.
"""
from pathlib import Path
import os,sys,json,time,hashlib,random,platform,importlib.util
from dataclasses import dataclass,asdict
P=Path(__file__).resolve().parent
os.environ['TEMP']=os.environ['TMP']=str(P/'cache')
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree,distance
from scipy.special import logsumexp,softmax
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,average_precision_score,roc_auc_score
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import sklearn,scipy,xgboost
torch.set_num_threads(1)
ROOT=Path(os.environ.get('PGCAN_PROJECT_ROOT',P.parent/'data/local/project'))
DATA=ROOT/'submission_revision_20260922'/'corrected'
OUT=P/'experiments';OUT.mkdir(exist_ok=True)
SD=P/'source_data'
spec=importlib.util.spec_from_file_location('original_model',DATA/'experiment_core.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
d,category,p0,activity,target,xy=m.prepare(pd.read_csv(DATA/'predictors_only.csv'))
grid=pd.read_csv(DATA/'allocation_500m.csv')
assert np.array_equal(d.grid_id,grid.grid_id)
raw=d[m.FEATURES].to_numpy(float);n=len(d)
base=np.log(np.maximum(p0,1e-12));response=np.log(np.maximum(target,1e-12))-base
onehot=np.eye(5,dtype=np.float32)[category]
reg=grid.primary_count.to_numpy(int)
energy=pd.read_csv(DATA/'energy_benchmark_sites.csv')
ec=np.bincount(pd.Index(d.grid_id).get_indexer(energy.grid_id),minlength=n)

@dataclass
class Config:
 seeds:tuple=(20260918,20260919,20260920)
 n_folds:int=5
 block_m:int=10000
 buffer_m:int=1000
 graph_k:int=8
 relation_radius_m:int=10000
 max_epochs:int=450
 epoch_interval:int=15
 early_patience:int=5
 lr:float=.008
 prior_penalty:float=.025
 graph_penalty:float=.015
 coefficient_penalty:float=.0005
cfg=Config()

VARIANTS={
 'PGCAN':{},
 'No_coordinate':{'coordinate':False},
 'No_context_branch':{'context':False},
 'No_function_input':{'function':False},
 'No_graph_input':{'graph':False},
 'No_graph_penalty':{'penalty':False},
 'No_graph':{'graph':False,'penalty':False},
 'No_prior_anchor':{'anchor':False},
 'No_prior_penalty':{'prior_penalty':False},
 'Multi_relation':{'multi':True},
 'MR_no_spatial':{'multi':True,'omit':0,'graph':False,'penalty':False},
 'MR_no_function':{'multi':True,'omit':1},
 'MR_no_built':{'multi':True,'omit':2},
 'MR_fixed_gate':{'multi':True,'fixed_gate':True},
 'MR_rewired':{'multi':True,'rewired':True},
}
CLASSICS=['Ridge','Random_forest','Extra_trees','Hist_gradient','XGBoost','MLP','Local_ridge']

def graph_edges(train,feat,kind,seed,rewire=False):
 """Query-to-training edges; no held-out node is ever a message source."""
 train=np.asarray(train);trxy=xy[train]
 if kind==0:
  dst,local=cKDTree(trxy).query(xy,k=min(cfg.graph_k+1,len(train)))
 else:
  indices=[1,2,5,6,7,8] if kind==1 else [0,3,4]
  z=feat[:,indices]
  if kind==1:z=np.c_[z,onehot]
  dst,local=cKDTree(z[train]).query(z,k=min(65,len(train)))
 rows=[];cols=[];weights=[];rng=np.random.default_rng(seed+kind*13)
 for i in range(n):
  js=train[np.atleast_1d(local[i])];ds=np.atleast_1d(dst[i])
  keep=js!=i
  if kind:keep&=(np.linalg.norm(xy[js]-xy[i],axis=1)<=cfg.relation_radius_m)
  js,ds=js[keep][:cfg.graph_k],ds[keep][:cfg.graph_k]
  if len(js)==0:
   dd,jj=cKDTree(trxy).query(xy[i],k=min(cfg.graph_k+1,len(train)))
   js=train[np.atleast_1d(jj)];ds=np.atleast_1d(dd);keep=js!=i
   js,ds=js[keep][:cfg.graph_k],ds[keep][:cfg.graph_k]
  scale=max(float(np.median(ds)),1e-5)
  w=np.exp(-.5*(ds/scale)**2);w=w/w.sum()
  if rewire:js=rng.choice(train[train!=i],size=len(js),replace=False)
  rows.extend([i]*len(js));cols.extend(js.tolist());weights.extend(w.tolist())
 rows=np.asarray(rows);cols=np.asarray(cols);weights=np.asarray(weights,np.float32)
 assert np.isin(cols,train).all()
 return rows,cols,weights

def prepare(train,seed,rewire=False):
 # Does train indirectly contain test labels? No. Only fixed raw covariates,
 # the declared prior recipe and graph geometry are used; scalers fit TRAIN.
 train=np.asarray(train);scaler=StandardScaler().fit(raw[train])
 z=scaler.transform(raw).astype(np.float32)
 cs=StandardScaler().fit(xy[train]);coords=cs.transform(xy).astype(np.float32)
 graphs=[graph_edges(train,z,r,seed,rewire) for r in range(3)]
 aggregates=[]
 for rows,cols,w in graphs:
  out=np.zeros_like(z);np.add.at(out,rows,w[:,None]*z[cols]);aggregates.append(out)
 r,c,w=graphs[0];mask=np.isin(r,train)
 tensors=[torch.from_numpy(a) for a in [z,coords,onehot,*aggregates]]
 return tensors,graphs,(torch.tensor(r[mask]),torch.tensor(c[mask]),torch.tensor(w[mask])),scaler,cs

class Net(torch.nn.Module):
 def __init__(self,opts):
  super().__init__();self.opts=opts
  self.coord=torch.nn.Sequential(torch.nn.Linear(2,48),torch.nn.Tanh(),torch.nn.Linear(48,48),torch.nn.Tanh(),torch.nn.Linear(48,10))
  dim=9+5*opts.get('function',True)+9*opts.get('graph',True)
  if opts.get('multi'):
   self.encoders=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(9,16),torch.nn.ReLU()) for _ in range(3)])
   self.gate=torch.nn.Linear(16,3);dim+=16
  self.context=torch.nn.Sequential(torch.nn.Linear(dim,48),torch.nn.ReLU(),torch.nn.Linear(48,24),torch.nn.ReLU(),torch.nn.Linear(24,1))
 def forward(self,t):
  x,coords,f,g,*other=t;opts=self.opts
  b=self.coord(coords);coord=(b[:,:9]*x).sum(1)+b[:,9]
  if not opts.get('coordinate',True):coord=coord*0
  parts=[x]
  if opts.get('function',True):parts.append(f)
  if opts.get('graph',True):parts.append(g)
  gates=None
  if opts.get('multi'):
   logits=self.gate(torch.cat([x,coords,f],1))
   if opts.get('fixed_gate'):logits=logits*0
   if 'omit' in opts:logits=logits.clone();logits[:,opts['omit']]=-1e9
   gates=torch.softmax(logits,1)
   embedded=torch.stack([net(a) for net,a in zip(self.encoders,[g,*other])],1)
   parts.append((embedded*gates[:,:,None]).sum(1))
  context=self.context(torch.cat(parts,1)).squeeze(1)
  if not opts.get('context',True):context=context*0
  return coord+context,b[:,:9],gates

def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def fit_net(name,train,validation,seed,epochs=None,save_path=None):
 opts=VARIANTS[name];seed_all(seed)
 t,graphs,edges,scaler,cs=prepare(train,seed,opts.get('rewired',False))
 net=Net(opts);optimizer=torch.optim.Adam(net.parameters(),lr=cfg.lr,weight_decay=.0002)
 tr=torch.tensor(train);ti=torch.tensor(target[train]/target[train].sum(),dtype=torch.float32)
 priorlog=torch.tensor(base if opts.get('anchor',True) else np.zeros(n),dtype=torch.float32)
 prior_tr=torch.tensor(base[train]-logsumexp(base[train]),dtype=torch.float32)
 ei,ej,ew=edges;trace=[];best=float('inf');bestepoch=cfg.epoch_interval;stale=0
 maxep=epochs if epochs else cfg.max_epochs
 for epoch in range(1,maxep+1):
  optimizer.zero_grad();r,b,_=net(t);lp=torch.log_softmax(priorlog[tr]+r[tr],0)
  fit=-(ti*lp).sum();pl=((lp-prior_tr)**2).mean()
  gl=(ew*(r[ei]-r[ej])**2).sum()/ew.sum() if opts.get('penalty',True) else r.sum()*0
  loss=fit+cfg.prior_penalty*pl*opts.get('prior_penalty',True)+cfg.graph_penalty*gl+cfg.coefficient_penalty*(b[tr]**2).mean()
  loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step()
  if epoch%cfg.epoch_interval==0 or epoch==maxep:
   with torch.no_grad():
    rr,_,_=net(t)
    if validation is not None:
     tv=target[validation];vlog=priorlog[validation]+rr[validation]
     val=float(-(torch.tensor(tv/tv.sum(),dtype=torch.float32)*torch.log_softmax(vlog,0)).sum())
    else:val=float(fit)
   trace.append({'epoch':epoch,'train_ce':float(fit.detach()),'validation_ce':val,'total_loss':float(loss.detach())})
   if validation is not None:
    if val<best-1e-6:best=val;bestepoch=epoch;stale=0
    else:stale+=1
    if stale>=cfg.early_patience and epoch>=120:break
 net.eval()
 with torch.no_grad():r,b,gates=net(t);r=r.numpy();b=b.numpy();gates=None if gates is None else gates.numpy()
 if save_path:
  torch.save({'state_dict':net.state_dict(),'options':opts,'feature_mean':scaler.mean_,'feature_scale':scaler.scale_,
   'coordinate_mean':cs.mean_,'coordinate_scale':cs.scale_,'train_indices':np.asarray(train),'epochs':maxep,'seed':seed},save_path)
 return r,trace,b,gates,sum(p.numel() for p in net.parameters()),bestepoch

def calibrated(residual,train,anchor=True):
 logscore=(base if anchor else np.zeros(n))+np.clip(residual,-20,20)
 offset=logsumexp(logscore[train])-np.log(target[train].sum())
 return np.exp(logscore-offset)

def classic_fit(name,train,seed):
 t,*_=prepare(train,seed);z=np.c_[t[0].numpy(),t[2].numpy(),t[3].numpy(),t[1].numpy()]
 y=response[train]
 models={
  'Ridge':lambda:Ridge(alpha=1.),
  'Random_forest':lambda:RandomForestRegressor(n_estimators=220,min_samples_leaf=3,max_features=.8,n_jobs=1,random_state=seed),
  'Extra_trees':lambda:ExtraTreesRegressor(n_estimators=220,min_samples_leaf=3,max_features=.8,n_jobs=1,random_state=seed),
  'Hist_gradient':lambda:HistGradientBoostingRegressor(max_iter=220,max_leaf_nodes=15,l2_regularization=1.,early_stopping=False,random_state=seed),
  'XGBoost':lambda:XGBRegressor(n_estimators=250,max_depth=3,learning_rate=.04,subsample=.85,colsample_bytree=.85,reg_lambda=3,n_jobs=1,random_state=seed),
  'MLP':lambda:MLPRegressor(hidden_layer_sizes=(48,24),alpha=.01,max_iter=700,early_stopping=False,random_state=seed,tol=1e-5),
 }
 if name=='Local_ridge':
  # Regularized geographically weighted linear regression; fixed 128-neighbour
  # Gaussian kernel, no bandwidth search and no temporal component.
  train=np.asarray(train);dd,ii=cKDTree(xy[train]).query(xy,k=min(128,len(train)))
  feature=np.c_[np.ones(n),z[:,:9]];r=np.zeros(n)
  for j in range(n):
   ids=train[ii[j]];w=np.exp(-.5*(dd[j]/max(np.median(dd[j]),1))**2)
   f=feature[ids];pen=np.eye(10)*.1;pen[0,0]=1e-6
   beta=np.linalg.solve((f.T*w)@f+pen,(f.T*w)@y[ii[j]])
   r[j]=feature[j]@beta
  return r,10000
 model=models[name]();model.fit(z[train],y);pred=model.predict(z)
 if name=='MLP':size=sum(a.size for a in model.coefs_)+sum(a.size for a in model.intercepts_)
 elif name=='Ridge':size=len(model.coef_)+1
 elif hasattr(model,'estimators_'):size=sum(a.tree_.node_count for a in model.estimators_)
 else:size=-1
 return pred,size

def metrics(observed,pred):
 return {'proxy_rmse_pp':float(np.sqrt(mean_squared_error(observed,pred))*100),
  'proxy_mae_pp':float(mean_absolute_error(observed,pred)*100),'proxy_r2':float(r2_score(observed,pred)),
  'proxy_bias_pp':float(np.mean(pred-observed)*100)}

def screen(p):
 p=m.normalize(p);order=np.lexsort((d.grid_id.to_numpy(str),-p));top=order[:198]
 return {'regulatory_hits':int(reg[top].sum()),'regulatory_coverage_pct':float(reg[top].sum()/146*100),
  'energy_hits':int(ec[top].sum()),'energy_coverage_pct':float(ec[top].sum()/69*100),
  'regulatory_ap':float(average_precision_score(reg>0,p)),'regulatory_auc':float(roc_auc_score(reg>0,p)),
  'tv_proxy':float(abs(p-target).sum()/2),'tv_prior':float(abs(p-p0).sum()/2),'top10_mass_pct':float(p[top].sum()*100)}

def main():
 start=time.perf_counter();allidx=np.arange(n)
 # Covariate-only distance correlogram; target never chooses the block design.
 rng=np.random.default_rng(20261020);a=rng.integers(n,size=100000);b=rng.integers(n,size=100000)
 dist=np.linalg.norm(xy[a]-xy[b],axis=1);v=np.log1p(raw[:,0]);v=(v-v.mean())/v.std()
 corr=[]
 for lo in range(0,20000,1000):
  mask=(dist>=lo)&(dist<lo+1000)&(a!=b)
  corr.append({'distance_lower_m':lo,'distance_upper_m':lo+1000,'pairs':int(mask.sum()),'ntl_standardized_product':float(np.mean(v[a[mask]]*v[b[mask]]))})
 pd.DataFrame(corr).to_csv(SD/'Benchmark_covariate_correlogram.csv',index=False)
 block=(np.floor((xy[:,0]-xy[:,0].min())/cfg.block_m).astype(int).astype(str))
 block=np.char.add(np.char.add(block,'_'),np.floor((xy[:,1]-xy[:,1].min())/cfg.block_m).astype(int).astype(str))
 splits=[];manifest=[];fold_id=np.zeros(n,int)
 for fold,(tr,te) in enumerate(GroupKFold(n_splits=cfg.n_folds).split(raw,groups=block)):
  near=cKDTree(xy[te]).query(xy[tr])[0];tr=tr[near>=cfg.buffer_m]
  assert not set(block[tr])&set(block[te]);minimum=float(cKDTree(xy[tr]).query(xy[te])[0].min())
  assert minimum>=cfg.buffer_m
  # Fixed spatial inner validation, never selected by outer-test performance.
  inner_tr,va=next(GroupKFold(n_splits=4).split(raw[tr],groups=block[tr]))
  fit,va=tr[inner_tr],tr[va]
  fit=fit[cKDTree(xy[va]).query(xy[fit])[0]>=cfg.buffer_m]
  assert len(fit)>100
  splits.append((tr,te,fit,va));fold_id[te]=fold+1
  manifest.append({'fold':fold+1,'n_train':len(tr),'n_test':len(te),'inner_train':len(fit),'inner_validation':len(va),
   'min_train_test_centroid_m':minimum,'train_indices':tr.tolist(),'test_indices':te.tolist(),'inner_train_indices':fit.tolist(),'inner_validation_indices':va.tolist()})
 (OUT/'split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 eda=pd.DataFrame({'predictor':m.FEATURES,'missing':np.isnan(raw).sum(0),'zero_fraction':(raw==0).mean(0),
  'min':raw.min(0),'median':np.median(raw,axis=0),'max':raw.max(0)})
 eda.to_csv(SD/'Benchmark_input_audit.csv',index=False)
 pd.DataFrame({'grid_id':d.grid_id,'fold':fold_id,'block':block,'target_share':target,'prior_share':p0,
  'category':[m.CLASSES[i] for i in category],'center_lon':d.center_lon,'center_lat':d.center_lat}).to_csv(SD/'Benchmark_cell_manifest.csv',index=False)
 protocol={'config':asdict(cfg),'data_sha256':hashlib.sha256((DATA/'predictors_only.csv').read_bytes()).hexdigest(),
  'target':'Predictor-derived t=0.70 prior + 0.30 activity; not observed emissions.',
  'spatial_scope':'Single fixed-domain snapshot; no temporal forecasting or intercity transfer claim.',
  'leakage':'Train-only scalers; query messages only from training nodes. Site membership excluded from fitting/selection. Prior/target are globally normalized deterministic recipes on the known domain; this is proxy reconstruction, not independent outcome validation.',
  'uncertainty':'Five fixed buffered geographic folds x three training seeds; fold and seed spread are descriptive, not independent replicates.',
  'normalization':'Absolute proxy shares calibrated on training mass only; RMSE/MAE/R2 use unrenormalized held-out predictions. OOF maps normalized only for allocation/site screening.',
  'comparison':'Fixed declared hyperparameters; neural epochs chosen on spatial inner validation then refit on all outer training nodes. Analytical direct mixture is an exact target reference, not a learned competitor.',
  'relations':['distance 8NN','functional similarity within 10 km','built/activity similarity within 10 km'],
  'versions':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'xgboost':xgboost.__version__},
  'variants':VARIANTS,'classic_models':CLASSICS}
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
 rows=[];traces={};oofs={};maps={};fullrows=[];graphrows=[];architectures=[]
 names=['Clean_prior','Direct_mixture']+CLASSICS+list(VARIANTS)
 for name in names:
  deterministic=name in ['Clean_prior','Direct_mixture','Ridge','Local_ridge']
  seeds=cfg.seeds[:1] if deterministic else cfg.seeds
  oofs[name]=[]
  for seed in seeds:
   oof=np.zeros(n)
   for fold,(tr,te,fit,va) in enumerate(splits,1):
    tick=time.perf_counter();epochs=0;params=0
    if name in ['Clean_prior','Direct_mixture']:pred=p0 if name=='Clean_prior' else target
    elif name in CLASSICS:
     residual,params=classic_fit(name,tr,seed);pred=calibrated(residual,tr)
    else:
     _,history,_,_,_,epochs=fit_net(name,fit,va,seed)
     residual,finalhist,_,_,params,_=fit_net(name,tr,None,seed,epochs=epochs)
     traces[f'{name}_{seed}_fold{fold}']={'selection':history,'refit':finalhist}
     pred=calibrated(residual,tr,VARIANTS[name].get('anchor',True))
    assert np.isfinite(pred).all() and (pred>=0).all()
    oof[te]=pred[te];row={'method':name,'seed':seed,'fold':fold,'n_train':len(tr),'n_test':len(te),
     'epochs':epochs,'capacity_count':params,'elapsed_seconds':time.perf_counter()-tick,**metrics(target[te],pred[te])}
    rows.append(row)
    pd.DataFrame(rows).to_csv(SD/'Benchmark_fold_metrics.csv',index=False)
    print('CV',name,seed,'fold',fold,'RMSE_pp',round(row['proxy_rmse_pp'],6),'epochs',epochs,flush=True)
   oofs[name].append(oof)
   fullrows.append({'method':name,'seed':seed,'scope':'spatial_oof',**metrics(target,oof),**screen(oof)})
  np.save(OUT/f'oof_{name}.npy',oofs[name])
  # Full-domain refits are reported separately from held-out reconstruction.
  if name in ['Clean_prior','Direct_mixture']:
   maps[name]=np.array([p0 if name=='Clean_prior' else target])
  else:
   maps[name]=[]
   tr,te,fit,va=splits[0]
   for seed in seeds:
    tick=time.perf_counter()
    if name in CLASSICS:
     residual,params=classic_fit(name,allidx,seed);pred=softmax(base+np.clip(residual,-20,20));gates=None
    else:
     # Full-map epoch selection uses the same predetermined spatial inner holdout.
     _,history,_,_,_,epoch=fit_net(name,fit,va,seed)
     residual,history,beta,gates,params,_=fit_net(name,allidx,None,seed,epochs=epoch,save_path=OUT/f'model_{name}_{seed}.pt')
     pred=softmax((base if VARIANTS[name].get('anchor',True) else np.zeros(n))+np.clip(residual,-20,20))
     traces[f'{name}_{seed}_full']=history
     if gates is not None:np.save(OUT/f'gates_{name}_{seed}.npy',gates)
     if seed==cfg.seeds[0]:architectures.append({'method':name,'trainable_parameters':params,'selected_epochs_full':epoch,**VARIANTS[name]})
    maps[name].append(pred)
    fullrows.append({'method':name,'seed':seed,'scope':'full_domain_refit','elapsed_seconds':time.perf_counter()-tick,**metrics(target,pred),**screen(pred)})
   maps[name]=np.array(maps[name])
  np.save(OUT/f'full_{name}.npy',maps[name])
  pd.DataFrame(fullrows).to_csv(SD/'Benchmark_map_metrics.csv',index=False)
  (OUT/'training_histories.json').write_text(json.dumps(traces),encoding='utf-8')
  pd.DataFrame(architectures).to_csv(SD/'Benchmark_architecture_counts.csv',index=False)
 # Relation audit in the final full-domain graph; directed duplicates identified.
 tensors,graphs,*_=prepare(allidx,cfg.seeds[0])
 for rid,(r,c,w) in enumerate(graphs):
  edge=np.c_[r,c];physical=np.linalg.norm(xy[r]-xy[c],axis=1)
  np.savez_compressed(OUT/f'relation_{rid}.npz',row=r,col=c,weight=w)
  graphrows.append({'relation':protocol['relations'][rid],'directed_edges':len(r),'median_distance_m':float(np.median(physical)),
   'p90_distance_m':float(np.quantile(physical,.9)),'same_function_fraction':float(np.mean(category[r]==category[c])),
   'zero_degree_nodes':int((np.bincount(r,minlength=n)==0).sum()),'max_row_sum_error':float(abs(np.bincount(r,weights=w,minlength=n)-1).max())})
 sets=[set(zip(r.tolist(),c.tolist())) for r,c,w in graphs]
 overlap=np.array([[len(a&b)/len(a|b) for b in sets] for a in sets])
 pd.DataFrame(overlap,index=protocol['relations'],columns=protocol['relations']).to_csv(SD/'Benchmark_relation_jaccard.csv')
 pd.DataFrame(graphrows).to_csv(SD/'Benchmark_relation_audit.csv',index=False)
 # Node-support strata use training-neighbour distances, never outcome values.
 strata=[]
 for fold,(tr,te,_,_) in enumerate(splits,1):
  dist_te=cKDTree(xy[tr]).query(xy[te],k=8)[0][:,-1]
  # Relative held-out spatial support: covariate/geometry-only test-query
  # distance tertiles. No target or site label defines these strata.
  limits=np.quantile(dist_te,[1/3,2/3])
  group=np.searchsorted(limits,dist_te)
  for name,pp in oofs.items():
   for seed,pr in zip(cfg.seeds,pp):
    for k,label in enumerate(['dense','middle','sparse']):
     ix=te[group==k]
     if len(ix)>1:strata.append({'method':name,'seed':seed,'fold':fold,'support':label,'n':len(ix),**metrics(target[ix],pr[ix])})
 pd.DataFrame(strata).to_csv(SD/'Benchmark_support_strata.csv',index=False)
 summary=pd.DataFrame(rows).groupby('method').agg(
  proxy_rmse_pp_mean=('proxy_rmse_pp','mean'),proxy_rmse_pp_sd=('proxy_rmse_pp','std'),
  proxy_mae_pp_mean=('proxy_mae_pp','mean'),proxy_mae_pp_sd=('proxy_mae_pp','std'),
  proxy_r2_mean=('proxy_r2','mean'),proxy_r2_sd=('proxy_r2','std'),
  seconds_mean=('elapsed_seconds','mean'),epochs_mean=('epochs','mean')).reset_index()
 summary.to_csv(SD/'Benchmark_summary.csv',index=False)
 protocol['elapsed_seconds']=time.perf_counter()-start;protocol['completed']=True
 protocol['model_count']=len(names);protocol['fold_fit_count']=len(rows)
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
 print('BENCHMARK_COMPLETE',protocol['model_count'],'models',len(rows),'fold fits',protocol['elapsed_seconds'],'seconds',flush=True)

if __name__=='__main__':main()
