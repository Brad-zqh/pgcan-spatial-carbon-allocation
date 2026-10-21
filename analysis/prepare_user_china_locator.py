"""Use the author's actual national, Taiwan, island and nine-dash geometries."""
import os
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
import geopandas as gpd
P=Path(__file__).resolve().parent
SOURCE=Path('D:/OneDrive/Paper/data1/tourism_comeptitiveness_spatial_temporal/Data/data_model/china_map')
FILES={
 'provinces':SOURCE/'China_shapfile/区划（审图号）/省（等积投影）.shp',
 'national_polygons':SOURCE/'轮廓线/国界.shp',
 'national_outline':SOURCE/'轮廓线/中国轮廓线.shp',
 'nine_dash':SOURCE/'九段线-WGS84/九段线.shp'}
if __name__=='__main__':
 out=P/'source_data/china_locator_user.gpkg';temp=out.with_name('china_locator_user_pending.gpkg')
 if temp.exists():temp.unlink()
 audit={'prepared_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Author-supplied national locator context; not analytical support',
  'source_dataset_identifier':'Source directory references GS(2019)1822; not an approval claim for the redrawn figure.',
  'layers':[]}
 for i,(name,file) in enumerate(FILES.items()):
  raw=gpd.read_file(file);assert raw.crs is not None
  frame=raw.to_crs(4326).copy();invalid=int((~frame.is_valid).sum())
  frame.geometry=frame.geometry.make_valid();assert frame.is_valid.all() and not frame.is_empty.any()
  if name=='provinces':
   assert frame['省'].astype(str).str.contains('台湾').any()
   frame=frame.rename(columns={'省代码':'adcode','省':'name','类型':'unit_type'})
  elif name=='national_polygons':
   assert raw['省'].astype(str).str.contains('台湾').any()
   frame=frame.rename(columns={'省代码':'adcode','省':'name','类型':'unit_type'})
  elif name=='national_outline':frame=frame[['geometry']]
  elif name=='nine_dash':frame=frame[['geometry']]
  frame.to_file(temp,layer=name,driver='GPKG',mode='w' if i==0 else 'a')
  audit['layers'].append({'layer':name,'source':str(file),'source_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
   'source_crs':str(raw.crs),'output_crs':'EPSG:4326','rows':len(frame),'invalid_before':invalid,
   'wgs84_bounds':frame.total_bounds.tolist(),'geometry_types':frame.geom_type.value_counts().to_dict()})
 temp.replace(out)
 (P/'source_data/china_locator_user_provenance.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
 print('AUTHOR_NATIONAL_BOUNDARY_READY',json.dumps({x['layer']:x['rows'] for x in audit['layers']}),flush=True)
