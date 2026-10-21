"""Prepare provider administrative context with documented coordinate conversion.

Display context only: these boundaries never replace the analytical study support.
DataV documents GCJ-02 map coordinates. An iterative inverse is applied before
projection; its numerical round-trip error does not imply survey accuracy.
"""
import os
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
import numpy as np
import geopandas as gpd
from shapely.ops import transform
import requests
P=Path(__file__).resolve().parent;S=P/'source_data';C=P/'cache'

def forward(lon,lat):
 lon=np.asarray(lon,dtype=float);lat=np.asarray(lat,dtype=float)
 x=lon-105;y=lat-35;pi=np.pi
 dlat=-100+2*x+3*y+.2*y*y+.1*x*y+.2*np.sqrt(abs(x))
 dlat+=(20*np.sin(6*x*pi)+20*np.sin(2*x*pi))*2/3
 dlat+=(20*np.sin(y*pi)+40*np.sin(y*pi/3))*2/3
 dlat+=(160*np.sin(y*pi/12)+320*np.sin(y*pi/30))*2/3
 dlon=300+x+2*y+.1*x*x+.1*x*y+.1*np.sqrt(abs(x))
 dlon+=(20*np.sin(6*x*pi)+20*np.sin(2*x*pi))*2/3
 dlon+=(20*np.sin(x*pi)+40*np.sin(x*pi/3))*2/3
 dlon+=(150*np.sin(x*pi/12)+300*np.sin(x*pi/30))*2/3
 rad=lat/180*pi;magic=1-.00669342162296594323*np.sin(rad)**2;root=np.sqrt(magic)
 dlat=dlat*180/((6378245*(1-.00669342162296594323)/(magic*root))*pi)
 dlon=dlon*180/(6378245/root*np.cos(rad)*pi)
 return lon+dlon,lat+dlat
def inverse(lon,lat,z=None):
 lon=np.asarray(lon,dtype=float);lat=np.asarray(lat,dtype=float);x=lon.copy();y=lat.copy()
 for _ in range(8):
  xx,yy=forward(x,y);x-=xx-lon;y-=yy-lat
 return (x,y) if z is None else (x,y,z)

def main():
 output=S/'administrative_context.gpkg';pending=S/'administrative_context_pending.gpkg'
 if pending.exists():pending.unlink()
 audit={'retrieved_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Display context; excluded from analytical support and model inputs',
  'provider':'Alibaba DataV GeoAtlas','coordinate_documentation':'https://help.aliyun.com/zh/datav/datav-7-0/user-guide/map-data-format-1',
  'source_coordinates':'GCJ-02, interpreted according to provider map-data documentation',
  'conversion':'Eight-iteration approximate inverse GCJ-02 to WGS84, then EPSG:32651. Numerical round-trip test only; no survey-accuracy claim.',
  'provider_snapshot_date':'Not specified by the service; retrieval date recorded','layers':[]}
 for i,(code,layer,expected) in enumerate([('330000','zhejiang_prefectures',11),('330100','hangzhou_districts',13)]):
  file=C/f'admin_{code}.json';url=f'https://geo.datav.aliyun.com/areas_v3/bound/{code}_full.json'
  if not file.exists():
   response=requests.get(url,timeout=40);response.raise_for_status();file.write_bytes(response.content)
  data=json.loads(file.read_text(encoding='utf-8'));raw=gpd.GeoDataFrame.from_features(data['features'])
  assert len(raw)==expected and raw.adcode.is_unique
  before_bad=int((~raw.is_valid).sum())
  geometries=[transform(inverse,geometry) for geometry in raw.geometry]
  frame=gpd.GeoDataFrame(raw[['adcode','name','level']].copy(),geometry=geometries,crs=4326)
  frame.geometry=frame.geometry.make_valid();frame=frame.to_crs(32651)
  assert frame.is_valid.all() and not frame.is_empty.any()
  frame.to_file(pending,layer=layer,driver='GPKG',mode='w' if i==0 else 'a')
  coord=np.array(data['features'][0]['geometry']['coordinates'][0][0])
  while coord.ndim>2:coord=coord[0]
  if coord.shape[-1]!=2:coord=coord.reshape(-1,2)
  x,y=inverse(coord[:,0],coord[:,1]);xx,yy=forward(x,y)
  error=float(max(abs(xx-coord[:,0]).max(),abs(yy-coord[:,1]).max()));assert error<1e-8
  audit['layers'].append({'layer':layer,'source_url':url,'raw_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
   'rows':len(frame),'invalid_source_geometries':before_bad,'valid_output_geometries':bool(frame.is_valid.all()),
   'numeric_roundtrip_max_degrees':error,'output_crs':'EPSG:32651','names':frame.name.tolist()})
 pending.replace(output)
 (S/'administrative_context_provenance.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
 print('ADMIN_CONTEXT_READY',11,'prefectures;',13,'Hangzhou district/county units',flush=True)
if __name__=='__main__':main()
