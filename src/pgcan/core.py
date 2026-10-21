"""Original PGCAN equations, graph construction and fitting, unchanged."""
from pathlib import Path
import json,hashlib,platform,time,random
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from pyproj import Transformer
OUT=Path("outputs")
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
