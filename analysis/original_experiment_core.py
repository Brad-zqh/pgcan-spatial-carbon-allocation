"""Leakage-free, reproducible rerun. Original inputs and outputs are read-only.
Run with a scientific Python environment containing torch, scipy, pandas,
scikit-learn, geopandas and pyproj. No validation field enters prepare/train.
"""
from pathlib import Path
import json, hashlib, platform, time, random
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from pyproj import Transformer
import geopandas as gpd

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
DATA=ROOT/'实验相关/杭州经济技术开发区实验数据0903'
READY=DATA/'论文可用整理版'
SOURCE=READY/'增强实验_500m土地利用_严格重点单位/01_500m网格_增强模型变量矩阵_含土地利用与严格重点单位.csv'
KEY=READY/'增强实验_500m土地利用_严格重点单位/04_园区边界内重点单位_严格高置信样本.csv'
FEATURES=['ntl_sum','enterprise_poi_density','industrial_poi_density','floor_area_proxy','road_density_km_per_km2','lowcarbon_patent_count','landuse_class1_ratio','landuse_class2_ratio','landuse_class3_ratio']
RAW=FEATURES+['grid_id','center_lon','center_lat','industrial_poi_count','building_density','innovation_enterprise_count','enterprise_poi_count','ntl_mean','cell_area_km2']
CLASSES=['industrial','transport_built','research','mixed','low_activity']
SEEDS=list(range(20260918,20260923))
TOTAL=15_000_000.
EPOCHS=900
torch.set_num_threads(1)

def normalize(a):
    a=np.maximum(np.asarray(a,dtype=np.float64),0)
    return a/a.sum() if a.sum()>0 else np.ones(len(a))/len(a)
def minmax(a):
    a=np.asarray(a,float);return (a-a.min())/(a.max()-a.min()) if a.max()>a.min() else a*0
def standard(a):
    a=np.asarray(a,float);s=a.std(axis=0);return ((a-a.mean(axis=0))/np.where(s>0,s,1)).astype('float32')
def carbon(p):
    c=normalize(p)*TOTAL;c[np.argmax(c)]+=TOTAL-c.sum();return c

def prepare(frame,strength=1.,alpha=.7):
    # Whitelist raw predictors. Never load old score/label/count columns here.
    d=frame[RAW].copy()
    for c in RAW:
        if c!='grid_id': d[c]=pd.to_numeric(d[c],errors='coerce').fillna(0)
    label=np.select([d.industrial_poi_count>=5,(d.road_density_km_per_km2>=8)&(d.building_density>=.25),(d.lowcarbon_patent_count>=5)|(d.innovation_enterprise_count>=2),(d.landuse_class3_ratio>=.55)&(d.enterprise_poi_count==0),(d.enterprise_poi_count>0)|(d.ntl_mean>0)],[0,1,2,4,3],default=4)
    z={c:minmax(d[c]) for c in FEATURES}
    a=.30*z['ntl_sum']+.25*z['enterprise_poi_density']+.15*z['industrial_poi_density']+.15*z['floor_area_proxy']+.10*z['landuse_class1_ratio']+.05*z['lowcarbon_patent_count']
    h=1+strength*(np.array([1.25,1.05,.90,.70,.15])[label]-1)
    prior=normalize(a*h)
    activity=normalize(.30*z['ntl_sum']+.25*z['enterprise_poi_density']+.20*z['floor_area_proxy']+.15*z['road_density_km_per_km2']+.10*z['lowcarbon_patent_count'])
    target=normalize(alpha*prior+(1-alpha)*activity)
    xy=np.column_stack(Transformer.from_crs(4326,32651,always_xy=True).transform(d.center_lon.to_numpy(),d.center_lat.to_numpy()))
    return d,label,prior,activity,target,xy

def graph_for(xy,k=8):
    distances,idx=cKDTree(xy).query(xy,k=k+1)
    sigma=float(np.median(distances[:,1:]))
    edges={}
    for i in range(len(xy)):
        for dist,j in zip(distances[i,1:],idx[i,1:]):
            pair=(min(i,int(j)),max(i,int(j)))
            edges[pair]=float(np.exp(-dist*dist/(2*sigma*sigma)))
    ij=np.array(list(edges));w=np.array(list(edges.values()),np.float32)
    ii=np.r_[ij[:,0],ij[:,1],np.arange(len(xy))];jj=np.r_[ij[:,1],ij[:,0],np.arange(len(xy))]
    ww=np.r_[w,w,np.ones(len(xy),np.float32)]
    degree=np.bincount(ii,weights=ww,minlength=len(xy))
    nw=ww/np.sqrt(degree[ii]*degree[jj])
    adj=torch.sparse_coo_tensor(torch.tensor(np.array([ii,jj])),torch.tensor(nw,dtype=torch.float32),(len(xy),len(xy))).coalesce()
    return adj,torch.tensor(ij[:,0]),torch.tensor(ij[:,1]),torch.tensor(w),sigma

class Model(torch.nn.Module):
    def __init__(self,function=True,graph=True):
        super().__init__();self.fun=function;self.gra=graph
        self.coord=torch.nn.Sequential(torch.nn.Linear(2,48),torch.nn.Tanh(),torch.nn.Linear(48,48),torch.nn.Tanh(),torch.nn.Linear(48,10))
        dim=9+5*function+9*graph
        self.context=torch.nn.Sequential(torch.nn.Linear(dim,48),torch.nn.ReLU(),torch.nn.Linear(48,24),torch.nn.ReLU(),torch.nn.Linear(24,1))
    def forward(self,x,xy,f,g):
        b=self.coord(xy);parts=[x]
        if self.fun:parts.append(f)
        if self.gra:parts.append(g)
        return (b[:,:9]*x).sum(1)+b[:,9]+self.context(torch.cat(parts,1)).squeeze(1),b[:,:9]

def train(d,label,prior,target,xy,seed,fun=True,gra=True,penalty=True,output='prior_softmax',k=8,save=None,prior_weight=.025):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    x=torch.tensor(standard(d[FEATURES]));coords=torch.tensor(standard(xy));f=torch.tensor(np.eye(5,dtype='float32')[label])
    adj,ei,ej,ew,sigma=graph_for(xy,k)
    gx=torch.sparse.mm(adj,x)
    prior_log=torch.tensor(np.log(np.maximum(prior,1e-12)),dtype=torch.float32)
    target_t=torch.tensor(target,dtype=torch.float32)
    model=Model(fun,gra);optimizer=torch.optim.Adam(model.parameters(),lr=.008,weight_decay=.0002)
    trace=[]
    for epoch in range(EPOCHS):
        optimizer.zero_grad();r,b=model(x,coords,f,gx)
        if output=='postnorm':
            score=torch.nn.functional.softplus(r);p=score/score.sum()
        elif output=='softmax':p=torch.softmax(r,0)
        else:p=torch.softmax(prior_log+r,0)
        logp=torch.log(p.clamp_min(1e-12))
        fit=-(target_t*logp).sum();pl=((logp-prior_log)**2).mean()
        gl=(ew*(r[ei]-r[ej])**2).sum()/ew.sum() if penalty else r.sum()*0
        loss=fit+prior_weight*pl+.015*gl+.0005*(b*b).mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if epoch%150==0 or epoch==EPOCHS-1:trace.append([epoch,float(loss.detach()),float(fit.detach()),float(pl.detach()),float(gl.detach()),float((torch.abs(p-target_t).sum()/2).detach())])
    with torch.no_grad():
        r,b=model(x,coords,f,gx)
        if output=='postnorm':p=torch.nn.functional.softplus(r);p=p/p.sum()
        elif output=='softmax':p=torch.softmax(r,0)
        else:p=torch.softmax(prior_log+r,0)
    if save:
        torch.save(model.state_dict(),OUT/(save+'.pt'))
    return normalize(p.numpy()),{'seed':seed,'trace':trace,'sigma_m':sigma,'graph_edges':len(ew),'function_context':fun,'graph_features':gra,'graph_penalty':penalty,'output':output}

def validation(d):
    if not KEY.exists():
        cached=pd.read_csv(OUT/'validation_counts.csv').set_index('grid_id').reindex(d.grid_id)
        assert not cached.isna().any().any()
        info=json.loads((OUT/'audit.json').read_text(encoding='utf-8'))['validation']
        return {name:cached[name+'_count'].to_numpy(int) for name in ['primary','secondary','records']},info
    key=pd.read_csv(KEY,dtype={'matched_poi_id':str})
    names=key.groupby('matched_poi_id').normalized_name.nunique()
    excluded=key[key.matched_poi_id.isin(names[names>1].index)]
    excluded.to_csv(OUT/'excluded_ambiguous_matches.csv',index=False,encoding='utf-8-sig')
    clean=key[key.matched_poi_id.isin(names[names==1].index)]
    primary=clean[clean.match_method=='exact_full'].drop_duplicates('matched_poi_id').copy()
    secondary=clean.drop_duplicates('matched_poi_id').copy()
    primary.to_csv(OUT/'validation_primary_exact_sites.csv',index=False,encoding='utf-8-sig')
    secondary.to_csv(OUT/'validation_secondary_sites.csv',index=False,encoding='utf-8-sig')
    vectors={}
    for name,r in [('primary',primary),('secondary',secondary),('records',key)]:
        vectors[name]=d.grid_id.map(r.groupby('grid_id').size()).fillna(0).to_numpy(int)
    return vectors,{'records':len(key),'distinct_names':int(key.enterprise_name.nunique()),'all_poi_ids':int(key.matched_poi_id.nunique()),'ambiguous_poi_ids':len(names[names>1]),'excluded_ambiguous_records':len(excluded),'primary_exact_sites':len(primary),'secondary_unique_sites':len(secondary)}

def evaluate(d,p,counts):
    n=max(1,int(.1*len(d)));order=np.lexsort((d.grid_id.to_numpy(str),-p));top=order[:n];c=carbon(p)
    result={'n_grids':len(d),'top_n':n,'mass_error_t':float(c.sum()-TOTAL),'min_t':float(c.min()),'max_t':float(c.max()),'top10_mass_share':float(p[top].sum())}
    for key,count in counts.items():
        result[key+'_n']=int(count.sum());result[key+'_hits']=int(count[top].sum())
        result[key+'_coverage']=float(count[top].sum()/count.sum())
        result[key+'_rho']=float(spearmanr(p,count).statistic)
        if key!='records':
            binary=count>0;result[key+'_ap']=float(average_precision_score(binary,p));result[key+'_auc']=float(roc_auc_score(binary,p))
    return result

def main():
    OUT.mkdir(exist_ok=True);start=time.time()
    source_path=SOURCE if SOURCE.exists() else OUT/'predictors_only.csv'
    original=pd.read_csv(source_path);d,label,prior,activity,target,xy=prepare(original)
    d.to_csv(OUT/'predictors_only.csv',index=False,encoding='utf-8-sig')
    # Executable leakage check: arbitrary validation/old-score values cannot affect preparation.
    altered=original.copy()
    for c in altered.columns:
        if c not in RAW:altered[c]=0
    check=prepare(altered)
    assert all(np.allclose(a,b) for a,b in zip((label,prior,activity,target,xy),check[1:]))
    counts,vaudit=validation(d)
    pd.DataFrame({'grid_id':d.grid_id,**{name+'_count':values for name,values in counts.items()}}).to_csv(OUT/'validation_counts.csv',index=False)
    results=[];histories={};probs={};grid=d[['grid_id','center_lon','center_lat']].copy()
    grid['function_class']=[CLASSES[i] for i in label]
    for name,y in counts.items():grid[name+'_count']=y
    for name,p in [('Nightlights',normalize(d.ntl_sum)),('Enterprise_POI',normalize(d.enterprise_poi_density)),('Activity',activity),('Clean_prior',prior),('Direct_mixture',target)]:
        probs[name]=p;results.append({'method':name,'seed':0,**evaluate(d,p,counts)})
    variants={'Full':{},'No_context':{'fun':False},'No_graph_features':{'gra':False},'No_graph_penalty':{'penalty':False},'No_graph':{'gra':False,'penalty':False},'Matched_postnorm':{'output':'postnorm'},'Matched_softmax':{'output':'softmax'}}
    for name,opts in variants.items():
        pp=[]
        for seed in SEEDS:
            p,info=train(d,label,prior,target,xy,seed,**opts,save=f'model_{name}_{seed}')
            pp.append(p);results.append({'method':name,'seed':seed,**evaluate(d,p,counts)})
            histories[f'{name}_{seed}']=info
            print(name,seed,'coverage',results[-1]['primary_coverage'],flush=True)
            pd.DataFrame(results).to_csv(OUT/'metrics_runs.csv',index=False)
        probs[name]=np.mean(pp,axis=0)
        np.save(OUT/f'probabilities_{name}.npy',np.array(pp))
    for name,p in probs.items():
        grid[name+'_share']=p;grid[name+'_tonnes']=carbon(p)
    grid.to_csv(OUT/'allocation_500m.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'method':name,**evaluate(d,p,counts)} for name,p in probs.items()]).to_csv(OUT/'metrics_ensemble.csv',index=False)
    curves=[]
    for name,p in probs.items():
        order=np.lexsort((d.grid_id.to_numpy(str),-p))
        for q in np.arange(.01,.501,.01):
            top=order[:max(1,int(q*len(d)))];curves.append({'method':name,'fraction':q,'primary_coverage':counts['primary'][top].sum()/counts['primary'].sum(),'secondary_coverage':counts['secondary'][top].sum()/counts['secondary'].sum(),'mass_share':p[top].sum()})
    pd.DataFrame(curves).to_csv(OUT/'coverage_curves.csv',index=False)
    # Sensitivity runs are fixed a priori and never used to select the reported model.
    sensitivity=[]
    for category,values in [('strength',[0.,.5,1.5]),('alpha',[.5,.9]),('k',[4,16])]:
        for value in values:
            dd,ll,pr,ac,ta,xx=prepare(original,**({category:value} if category!='k' else {}))
            p,info=train(dd,ll,pr,ta,xx,SEEDS[0],k=int(value) if category=='k' else 8)
            sensitivity.append({'parameter':category,'value':value,**evaluate(dd,p,counts)})
            print('sensitivity',category,value,flush=True)
    pd.DataFrame(sensitivity).to_csv(OUT/'sensitivity.csv',index=False)
    # Valid boundary exercise: contraction only, clearly not outward robustness.
    boundary_source=DATA/'园区边界/矢量边界/杭州经济技术开发区矢量边界0423.shp'
    boundary=gpd.read_file(boundary_source if boundary_source.exists() else OUT/'boundary.geojson').to_crs(32651)
    geom=boundary.geometry.union_all();pts=gpd.points_from_xy(xy[:,0],xy[:,1]);bd=[]
    full0=np.load(OUT/'probabilities_Full.npy')[0];nominal=set(d.iloc[np.lexsort((d.grid_id.to_numpy(str),-full0))[:198]].grid_id)
    for m in [100,200]:
        mask=np.array([geom.buffer(-m).contains(p) for p in pts]);subset=original.loc[mask].reset_index(drop=True)
        dd,ll,pr,ac,ta,xx=prepare(subset);co={k:v[mask] for k,v in counts.items()}
        p,info=train(dd,ll,pr,ta,xx,SEEDS[0]);top=set(dd.iloc[np.lexsort((dd.grid_id.to_numpy(str),-p))[:int(.1*len(dd))]].grid_id)
        bd.append({'contraction_m':m,'jaccard_nominal':len(top&nominal)/len(top|nominal),**evaluate(dd,p,co)})
        print('contraction',m,flush=True)
    pd.DataFrame(bd).to_csv(OUT/'boundary_contraction.csv',index=False)
    # Store geometries for artifact rendering without requiring GIS libraries there.
    geographic=boundary.to_crs(4326)
    (OUT/'boundary.geojson').write_text(geographic.to_json(),encoding='utf-8')
    audit={'validation':vaudit,'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),'key_sha256':hashlib.sha256(KEY.read_bytes()).hexdigest() if KEY.exists() else 'derived validation_counts.csv fallback','predictor_whitelist':RAW,'leakage_invariance_check':True,'seeds':SEEDS,'epochs':EPOCHS,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),'boundary_area_km2':geom.area/1e6,'nominal_grid_area_km2':d.cell_area_km2.sum(),'function_counts':dict(zip(*np.unique(np.array(CLASSES)[label],return_counts=True))),'seconds':time.time()-start}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,default=lambda x:int(x) if isinstance(x,np.integer) else str(x)),encoding='utf-8')
    (OUT/'training_history.json').write_text(json.dumps(histories,indent=2),encoding='utf-8')
    print('COMPLETE',json.dumps(audit,default=str),flush=True)

if __name__=='__main__':main()
