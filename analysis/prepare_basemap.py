"""Prepare the existing local vector basemap and reproducible locator context."""
from pathlib import Path
import os,sys,json,hashlib,zipfile,importlib.util
from datetime import datetime,timezone
P=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ['TMP']=os.environ['TEMP']=str(P/'cache')
os.environ['MPLCONFIGDIR']=str(P/'cache')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import geopandas as gpd
import pandas as pd
import requests
from cartography_style import context_bounds
from shapely.geometry import box
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(os.environ.get('PGCAN_PROJECT_ROOT',P.parent/'data/local/project'))
RAW=ROOT/'实验相关'/'杭州经济技术开发区实验数据0903'
OUT=P/'source_data';CACHE=P/'cache'/'locator_sources';CACHE.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('vector_hygiene',r'E:\CodexData\.codex\skills\geo-data-engineering\scripts\clean_vector.py')
clean=importlib.util.module_from_spec(spec);spec.loader.exec_module(clean)

def download(name):
    file=CACHE/(name+'.zip')
    url='https://naturalearth.s3.amazonaws.com/'+('10m_cultural/' if name.startswith('ne_10m') else '50m_cultural/')+name+'.zip'
    if not file.exists():
        r=requests.get(url,timeout=(10,55));r.raise_for_status()
        assert r.content[:2]==b'PK'
        file.write_bytes(r.content)
    dest=CACHE/name;dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(file) as z:
        for member in z.namelist():
            target=(dest/member).resolve()
            assert target.is_relative_to(dest.resolve())
        z.extractall(dest)
    shp=next(dest.glob('*.shp'))
    return name,url,file,shp

def main():
    prepared_geometry=OUT/'plotting_geometry.gpkg'
    if prepared_geometry.exists():geo=gpd.read_file(prepared_geometry,layer='boundary')
    else:
        source=ROOT/'论文撰写0903'/'投稿修订版_20260923'/'图3_ArcGIS独立地图_20260923'/'Figure_3a_Clean_Prior.gpkg'
        geo=gpd.read_file(source,layer='study_boundary')
    extent=context_bounds(geo.total_bounds)
    clip=gpd.GeoDataFrame(geometry=[box(*extent)],crs=32651)
    manifest={'prepared_utc':datetime.now(timezone.utc).isoformat(),'target_crs':'EPSG:32651',
              'clip_bounds':list(extent),'context_only':True,'local_sources':[],
              'attribution':'Local OSM-derived vectors: © OpenStreetMap contributors; locator context: Natural Earth',
              'source_snapshot_date':'Existing project extract; capture date not independently verified'}
    layers=[]
    paths={k:RAW/'道路网络'/'工业园区道路系统'/(k+'.shp') for k in ['roads','waters','waterway','railway']}
    paths['buildings']=RAW/'建筑轮廓'/'建筑轮廓定'/'buildings.shp'
    for name,file in paths.items():
        original=gpd.read_file(file)
        report={'layer':name,'source':str(file),'original_rows':len(original),'source_crs':str(original.crs),
                'invalid_before':int((~original.is_valid).sum()),'source_sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
        prepared=clean.clean_vector(original,32651)
        clipped=gpd.clip(prepared,clip,keep_geom_type=True)
        clipped=clipped[~clipped.geometry.is_empty].copy()
        keep=[x for x in ['osm_id','fclass','name','geometry'] if x in clipped.columns]
        clipped=clipped[keep].sort_values('osm_id' if 'osm_id' in keep else 'fclass').reset_index(drop=True)
        report.update(prepared_rows=len(prepared),clipped_rows=len(clipped),invalid_after=int((~clipped.is_valid).sum()),
                      class_counts=clipped.fclass.value_counts().to_dict() if 'fclass' in clipped else {})
        manifest['local_sources'].append(report);layers.append((name,clipped))
    gpkg=OUT/'cartographic_basemap.gpkg'
    # Replace the task-owned cache atomically; no historical copies are kept.
    pending=OUT/'cartographic_basemap_pending.gpkg'
    if pending.exists():pending.unlink()
    for i,(name,frame) in enumerate(layers):
        frame.to_file(pending,layer=name,driver='GPKG',mode='w' if i==0 else 'a')
    pending.replace(gpkg)
    names=['ne_50m_admin_0_countries','ne_10m_admin_1_states_provinces']
    with ThreadPoolExecutor(max_workers=2) as pool:downloads=list(pool.map(download,names))
    locator=OUT/'locator_context.gpkg';temp=OUT/'locator_context_pending.gpkg'
    if temp.exists():temp.unlink()
    manifest['locator_sources']=[]
    for i,(name,url,file,shp) in enumerate(downloads):
        frame=gpd.read_file(shp,bbox=(70,0,140,56))
        if i==0:
            keep=['ADMIN','SOVEREIGNT','geometry'];layer='countries'
        else:
            frame=frame[frame.admin.isin(['China','Taiwan','Hong Kong S.A.R.','Macao S.A.R.'])]
            keep=['name','name_en','admin','iso_3166_2','geometry'];layer='provinces'
        frame=frame[[k for k in keep if k in frame]].copy()
        assert frame.crs is not None and frame.is_valid.all()
        frame.to_file(temp,layer=layer,driver='GPKG',mode='w' if i==0 else 'a')
        manifest['locator_sources'].append({'layer':layer,'source_url':url,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'rows':len(frame),'crs':str(frame.crs)})
    temp.replace(locator)
    (OUT/'basemap_provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Prepared local basemap and locator context.',flush=True)

if __name__=='__main__':main()
