"""Independent executable checks of saved full refits and selected OOF predictions."""
import os
import run_model_benchmark as b
import numpy as np
import json
import torch
from scipy.special import softmax

def main():
 result={'full_models':[],'outer_predictions':[],'message_sources':[]}
 for name in ['PGCAN','Multi_relation','MR_rewired','No_graph','No_coordinate']:
  seed=b.cfg.seeds[0];path=b.OUT/f'model_{name}_{seed}.pt'
  checkpoint=torch.load(path,map_location='cpu',weights_only=False)
  train=checkpoint['train_indices'];opts=checkpoint['options']
  t,graphs,edges,scaler,coords=b.prepare(train,seed,opts.get('rewired',False))
  np.testing.assert_array_equal(checkpoint['feature_mean'],scaler.mean_)
  np.testing.assert_array_equal(checkpoint['feature_scale'],scaler.scale_)
  model=b.Net(opts);model.load_state_dict(checkpoint['state_dict']);model.eval()
  with torch.no_grad():r,_,gate=model(t);r=r.numpy()
  pred=softmax((b.base if opts.get('anchor',True) else np.zeros(b.n))+np.clip(r,-20,20))
  saved=np.load(b.OUT/f'full_{name}.npy')[0]
  np.testing.assert_allclose(pred,saved,rtol=0,atol=1e-14)
  if gate is not None:
   np.testing.assert_array_equal(gate.numpy(),np.load(b.OUT/f'gates_{name}_{seed}.npy'))
  result['full_models'].append({'method':name,'max_abs_share_difference':float(abs(pred-saved).max())})
 split=json.loads((b.OUT/'split_manifest.json').read_text())[0]
 train=np.array(split['train_indices']);test=np.array(split['test_indices'])
 for name in ['PGCAN','Multi_relation']:
  seed=b.cfg.seeds[0]
  import pandas as pd
  metrics=pd.read_csv(b.SD/'Benchmark_fold_metrics.csv')
  row=metrics[(metrics.method==name)&(metrics.seed==seed)&(metrics.fold==1)].iloc[0]
  r,*_=b.fit_net(name,train,None,seed,epochs=int(row.epochs))
  pred=b.calibrated(r,train,b.VARIANTS[name].get('anchor',True))
  expected=np.load(b.OUT/f'oof_{name}.npy')[0]
  np.testing.assert_allclose(pred[test],expected[test],rtol=0,atol=1e-14)
  result['outer_predictions'].append({'method':name,'fold':1,'seed':seed,'max_abs_share_difference':float(abs(pred[test]-expected[test]).max())})
  _,graphs,_,scaler,_=b.prepare(train,seed)
  np.testing.assert_array_equal(scaler.mean_,b.raw[train].mean(0))
  for relation,(r,c,w) in enumerate(graphs):
   assert np.isin(c,train).all() and not np.isin(c,test).any()
   result['message_sources'].append({'relation':relation,'train_only':True,'fold':1})
 result['status']='PASS'
 (b.P/'qa/experiment_checkpoint_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print('CHECKPOINT_AND_OUTER_REFIT_REPRODUCTION_PASS',flush=True)
if __name__=='__main__':main()
