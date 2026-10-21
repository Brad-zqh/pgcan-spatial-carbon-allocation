"""Cache and project a label-free imagery basemap for the complete map frame.

Imagery provides geographic context only. Grayscale/contrast are display
transforms and do not enter the analytical predictors or allocations.
"""
from pathlib import Path
import os,sys,json,hashlib,io,math
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
P=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
os.environ['TMP']=os.environ['TEMP']=str(P/'cache')
os.environ['MPLCONFIGDIR']=str(P/'cache')
import requests,numpy as np,geopandas as gpd,mercantile,rasterio
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_bounds
from rasterio.warp import reproject,Resampling
from cartography_style import context_bounds

BASE='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer'
CACHE=P/'cache/imagery_tiles_z13';CACHE.mkdir(parents=True,exist_ok=True)
SD=P/'source_data'
def tile(job):
    x,y,z=job
    f=CACHE/f'{z}_{x}_{y}.jpg'
    if not f.exists():
        url=f'{BASE}/tile/{z}/{y}/{x}'
        error=None
        for _ in range(2):
            try:
                r=requests.get(url,timeout=(8,25));r.raise_for_status()
                im=Image.open(io.BytesIO(r.content));assert im.size==(256,256)
                f.write_bytes(r.content);break
            except Exception as exc:error=exc
        else:raise RuntimeError(f'Failed tile {z}/{x}/{y}: {error}')
    return x,y,np.asarray(Image.open(f).convert('RGB'))

if __name__=='__main__':
    g=gpd.read_file(SD/'plotting_geometry.gpkg',layer='boundary')
    xmin,ymin,xmax,ymax=context_bounds(g.total_bounds)
    trans=Transformer.from_crs(32651,4326,always_xy=True)
    ll=np.array([trans.transform(x,y) for x in [xmin,xmax] for y in [ymin,ymax]])
    jobs=list(mercantile.tiles(ll[:,0].min(),ll[:,1].min(),ll[:,0].max(),ll[:,1].max(),13))
    xs=[j.x for j in jobs];ys=[j.y for j in jobs]
    mosaic=np.zeros(((max(ys)-min(ys)+1)*256,(max(xs)-min(xs)+1)*256,3),np.uint8)
    print('IMAGERY_TILES',len(jobs),flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        for x,y,rgb in ex.map(tile,[(j.x,j.y,j.z) for j in jobs]):
            xx=(x-min(xs))*256;yy=(y-min(ys))*256;mosaic[yy:yy+256,xx:xx+256]=rgb
    nw=mercantile.xy_bounds(min(xs),min(ys),13);se=mercantile.xy_bounds(max(xs),max(ys),13)
    source_transform=from_bounds(nw.left,se.bottom,se.right,nw.top,mosaic.shape[1],mosaic.shape[0])
    # Match native ground sampling rather than presenting interpolation as detail.
    resolution=17.0;width=math.ceil((xmax-xmin)/resolution);height=math.ceil((ymax-ymin)/resolution)
    destination=np.zeros((3,height,width),np.uint8)
    transform=from_bounds(xmin,ymin,xmax,ymax,width,height)
    for i in range(3):reproject(mosaic[:,:,i],destination[i],src_transform=source_transform,src_crs=3857,
        dst_transform=transform,dst_crs=32651,resampling=Resampling.bilinear)
    file=SD/'imagery_basemap_utm51n.tif'
    with rasterio.open(file,'w',driver='COG',height=height,width=width,count=3,dtype='uint8',
        crs=32651,transform=transform,compress='DEFLATE',blocksize=256) as dst:dst.write(destination)
    response=requests.get(BASE,params={'f':'json'},timeout=(8,25));response.raise_for_status();meta=response.json()
    attribution=meta.get('copyrightText','Sources: Esri, Vantor, Earthstar Geographics, and the GIS User Community')
    provenance={'provider':'Esri World Imagery','service_url':BASE,'copyrightText':attribution,
        'retrieved_utc':datetime.now(timezone.utc).isoformat(),'acquisition_date':'Mixed-date basemap; not inferred from retrieval date',
        'tile_zoom':13,'tile_count':len(jobs),'source_crs':'EPSG:3857','display_crs':'EPSG:32651',
        'bounds_utm':[xmin,ymin,xmax,ymax],'pixel_size_m':list(transform)[:6],
        'raster_shape':[height,width],'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
        'display_transform':'Grayscale tonal stretch for contextual display only; analytical values unchanged',
        'font_policy':'No imagery labels; all added coordinates and labels are Times New Roman'}
    (SD/'imagery_basemap_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    print('IMAGERY_READY',width,height,'attribution:',attribution,flush=True)
