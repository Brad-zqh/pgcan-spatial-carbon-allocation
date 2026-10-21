"""Real vector context, a split north arrow, and projected scale furniture.

The basemap is an existing OSM-derived project extract, prepared separately.
All shapes and letters remain vector artists in PDF/SVG.
"""
import os
from pathlib import Path
import numpy as np
import geopandas as gpd
from pyproj import Transformer
from matplotlib.patches import Polygon,Rectangle
from matplotlib.offsetbox import DrawingArea,AnchoredOffsetbox
from matplotlib.text import Text
from matplotlib.transforms import Affine2D
from shapely.geometry import box

ATTRIBUTION='Basemap: Esri, Vantor, Earthstar Geographics, GIS User Community; © OpenStreetMap contributors'
THEME_ALPHA=.74

def context_bounds(bounds):
    """Projected frame with room for surrounding landscape and map furniture."""
    xmin,ymin,xmax,ymax=bounds
    cx=(xmin+xmax)/2;cy=(ymin+ymax)/2
    sx=xmax-xmin;sy=ymax-ymin
    return (cx-.63*sx,cy-.73*sy,cx+.63*sx,cy+.76*sy)
def load_basemap(path):
    assert Path(path).exists(),'Run prepare_basemap.py first.'
    layers={k:gpd.read_file(path,layer=k) for k in ['roads','waters','waterway','railway','buildings']}
    raster=Path(path).with_name('imagery_basemap_utm51n.tif')
    if raster.exists():
        import rasterio
        with rasterio.open(raster) as src:
            rgb=src.read().astype(np.float32)/255
            gray=.2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]
            low,high=np.quantile(gray,[.02,.98]);gray=.48+.48*np.clip((gray-low)/(high-low),0,1)
            layers['_imagery']=(gray,(src.bounds.left,src.bounds.right,src.bounds.bottom,src.bounds.top))
    return layers

def draw_basemap(ax,layers,extent,detail=False):
    x0,x1,y0,y1=extent
    imagery=layers.get('_imagery')
    if imagery is not None:
        gray,extent_r=imagery
        ax.imshow(gray,cmap='gray',vmin=0,vmax=1,extent=extent_r,origin='upper',zorder=-10,rasterized=True)
    else:
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fc='#F2F4F2',ec='none',zorder=-10))
    def subset(k):
        frame=layers[k]
        return frame.cx[x0:x1,y0:y1]
    waters=subset('waters')
    if len(waters):waters.plot(ax=ax,color='#DDE5E8',edgecolor='#D5DFE1',linewidth=.18,zorder=-8,alpha=.25 if imagery is not None else 1)
    buildings=subset('buildings')
    if len(buildings) and (detail or imagery is None):buildings.plot(ax=ax,color='#E0E5E3',edgecolor='#FFFFFF' if detail else 'none',linewidth=.14 if detail else 0,zorder=-7,alpha=.22 if imagery is not None else 1)
    rivers=subset('waterway')
    if len(rivers):rivers.plot(ax=ax,color='#D5E0E4',linewidth=.7 if detail else .5,zorder=-6)
    roads=subset('roads')
    main=roads.fclass.isin(['motorway','motorway_link','trunk','trunk_link','primary','primary_link','secondary','secondary_link','tertiary','tertiary_link'])
    minor=roads[~main]
    if not detail:minor=minor[~minor.fclass.isin(['path','track','footway','steps','cycleway','bridleway'])]
    if len(minor):minor.plot(ax=ax,color='#FFFFFF' if detail else '#BFCACD',linewidth=.55 if detail else .38,zorder=-5)
    if main.any():
        roads[main].plot(ax=ax,color='#CCD4D5' if detail else '#B8C4C6',linewidth=1.15 if detail else .85,zorder=-4)
        roads[main].plot(ax=ax,color='#FFFFFF',linewidth=.68 if detail else .42,zorder=-3)
    rail=subset('railway')
    if len(rail):rail.plot(ax=ax,color='#D1D7D7',linewidth=.3,linestyle=(0,(2,1.6)),zorder=-2)
    ax.set_xlim(x0,x1);ax.set_ylim(y0,y1);ax.set_aspect('equal')

def north_arrow(ax,crs=32651,anchor=(.99,.965),height=10,outside=False):
    """Place an in-frame needle in contextual space clear of protected features."""
    ax.apply_aspect()
    if ax.get_window_extent().width/ax.figure.dpi*25.4<40:height=min(height,8)
    xx=sum(ax.get_xlim())/2;yy=sum(ax.get_ylim())/2
    togeo=Transformer.from_crs(crs,4326,always_xy=True)
    toproj=Transformer.from_crs(4326,crs,always_xy=True)
    lon,lat=togeo.transform(xx,yy);nx,ny=toproj.transform(lon,lat+.01)
    angle=np.degrees(np.arctan2(-(nx-xx),ny-yy))
    width=height*.61;h=height;w=width
    da=DrawingArea(w+3,h+12,0,0)
    shift=1.5;cx=w/2+shift
    vertices=[(cx,h+1),(shift,1),(cx,h*.30+1),(w+shift,1)]
    rot=Affine2D().rotate_deg_around(cx,h*.5+1,angle)
    left=rot.transform(np.array([vertices[0],vertices[1],vertices[2]]))
    right=rot.transform(np.array([vertices[0],vertices[2],vertices[3]]))
    da.add_artist(Polygon(left,fc='#111111',ec='#111111',lw=.5,joinstyle='miter'))
    da.add_artist(Polygon(right,fc='#FFFFFF',ec='#111111',lw=.5,joinstyle='miter'))
    letter=Text(cx,h+7.2,'N',ha='center',va='center',fontfamily='Times New Roman',fontsize=8.5,color='#111111')
    letter.set_gid('map_north_label');da.add_artist(letter)
    if outside:
        raise ValueError('North arrows must remain within the expanded map frame.')
    bbox=ax.get_window_extent();dpi=ax.figure.dpi
    width_px=(w+3)*dpi/72;height_px=(h+12)*dpi/72
    avoid=getattr(ax,'_north_avoid_geometry',None)
    candidates=[anchor,(.99,.90),(.99,.78),(.15,.965),(.21,.965),(.99,.55),(.99,.43)]
    position=None
    for candidate in candidates:
        right=bbox.x0+candidate[0]*bbox.width;top=bbox.y0+candidate[1]*bbox.height
        left=right-width_px;bottom=top-height_px
        if left<bbox.x0+1 or bottom<bbox.y0+1:continue
        xy=ax.transData.inverted().transform([[left,bottom],[right,top]])
        protected=box(xy[0,0],xy[0,1],xy[1,0],xy[1,1])
        if avoid is None or not avoid.intersects(protected):
            position=candidate;break
    if position is None:
        raise RuntimeError('No clear in-frame north-arrow position: expand the geographic frame.')
    artist=AnchoredOffsetbox(loc='upper left' if outside else 'upper right',child=da,pad=0,borderpad=0,
        bbox_to_anchor=position,bbox_transform=ax.transAxes,frameon=False)
    artist.patch.set_visible(False)
    artist.set_gid('north_arrow');artist._outside_map=outside
    artist.set_zorder(20);ax.add_artist(artist)
    return angle

def scale_bar(ax,length=6000,anchor=(.055,.085),segments=4):
    ax.apply_aspect()
    x0,x1=ax.get_xlim();y0,y1=ax.get_ylim();spanx=x1-x0;spany=y1-y0
    bbox=ax.get_window_extent();px_per_pt=ax.figure.dpi/72
    x=x0+anchor[0]*spanx;y=y0+anchor[1]*spany
    height=spany*(3.0*px_per_pt)/bbox.height
    pad=spany*(1.8*px_per_pt)/bbox.height
    artists=[]
    line=ax.plot([x,x+length],[y,y],color='black',lw=1.15,zorder=15)[0]
    line.set_gid('map_scale_segment');artists.append(line)
    for xx in [x,x+length]:
        tick=ax.plot([xx,xx],[y,y+height],color='black',lw=.65,zorder=15)[0]
        tick.set_gid('map_scale_segment');artists.append(tick)
    ticks=[(0,'0'),(length,f'{length/1000:g}')]
    for offset,label in ticks:
        artist=ax.text(x+offset,y+height+pad,label,ha='center',va='bottom',fontsize=6.2,
            fontfamily='Times New Roman',color='black',zorder=16)
        artist.set_gid('map_scale_label');artists.append(artist)
    last_label_bounds=artists[-1].get_window_extent(ax.figure.canvas.get_renderer())
    unit_start=ax.transData.inverted().transform((last_label_bounds.x1+3*px_per_pt,last_label_bounds.y0))[0]
    unit_gap=unit_start-(x+length)
    artist=ax.text(x+length+unit_gap,y+height+pad,'km',ha='left',va='bottom',fontsize=6.2,
        fontfamily='Times New Roman',color='black',zorder=16)
    artist.set_gid('map_scale_label');artists.append(artist)
    renderer=ax.figure.canvas.get_renderer()
    bounds=[artist.get_window_extent(renderer) for artist in artists]
    from matplotlib.transforms import Bbox
    combined=Bbox.union(bounds)
    corners=ax.transData.inverted().transform([[combined.x0,combined.y0],[combined.x1,combined.y1]])
    ax._scale_bar_boxes=getattr(ax,'_scale_bar_boxes',[])+[box(*corners.ravel())]
    ax._scale_bar_has_backdrop=False

def map_furniture(ax,length=6000):
    scale_bar(ax,length=length);north_arrow(ax)

def compact_graticule(ax,crs=32651):
    """Real geographic coordinates positioned outside every map frame."""
    tr=Transformer.from_crs(crs,4326,always_xy=True)
    x0,x1=ax.get_xlim();y0,y1=ax.get_ylim()
    lon,lat=tr.transform([x0,x1],[y0,y1])
    xt=np.linspace(min(lon),max(lon),4)[1:-1]
    if ax.get_window_extent().width/ax.figure.dpi*25.4<38:xt=[float(np.mean(xt))]
    yt=np.linspace(min(lat),max(lat),4)[1:-1]
    projected_graticule(ax,xt,yt,crs=crs,fontsize=7.2)

def projected_graticule(ax,lon_ticks,lat_ticks,crs=32651,fontsize=6.4,xside='top',yside='left',degree_only=False):
    """Boundary coordinate ticks; background is drawn in projected metres."""
    togeo=Transformer.from_crs(crs,4326,always_xy=True);toproj=Transformer.from_crs(4326,crs,always_xy=True)
    x0,x1=ax.get_xlim();y0,y1=ax.get_ylim()
    lon0,lat0=togeo.transform(x0,y0);lon1,lat1=togeo.transform(x1,y1)
    def deg(v,axis):
        if degree_only:return f'{v:g}°'+('E' if axis=='x' else 'N')
        degrees=int(v);minutes=round((v-degrees)*60)
        if minutes==60:degrees+=1;minutes=0
        return f'{degrees}°{minutes:02d}′'+('E' if axis=='x' else 'N')
    # Intersections with each frame edge are solved in projected coordinates.
    from scipy.optimize import brentq
    xt=[];xl=[]
    for lon in lon_ticks:
        try:
            atlat=brentq(lambda lat:toproj.transform(lon,lat)[1]-(y1 if xside=='top' else y0),min(lat0,lat1)-10,max(lat0,lat1)+10)
            xx,_=toproj.transform(lon,atlat)
            if x0<xx<x1:xt.append(xx);xl.append(deg(lon,'x'))
        except ValueError:pass
    yt=[];yl=[]
    for lat in lat_ticks:
        try:
            atlon=brentq(lambda lon:toproj.transform(lon,lat)[0]-(x0 if yside=='left' else x1),min(lon0,lon1)-20,max(lon0,lon1)+20)
            _,yy=toproj.transform(atlon,lat)
            if y0<yy<y1:yt.append(yy);yl.append(deg(lat,'y'))
        except ValueError:pass
    ax.set_axis_on();ax.set_xticks(xt,xl);ax.set_yticks(yt,yl)
    ax.tick_params(axis='x',top=xside=='top',labeltop=xside=='top',bottom=xside=='bottom',labelbottom=xside=='bottom',length=2,pad=2,labelsize=fontsize,direction='out')
    ax.tick_params(axis='y',left=yside=='left',labelleft=yside=='left',right=yside=='right',labelright=yside=='right',length=2,pad=2,labelsize=fontsize,direction='out')
    for label in ax.get_xticklabels():label.set_va('bottom' if xside=='top' else 'top');label.set_bbox(None);label.set_gid('map_coordinate_label')
    for label in ax.get_yticklabels():label.set_ha('right' if yside=='left' else 'left');label.set_bbox(None);label.set_gid('map_coordinate_label')
    for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#8A999E');spine.set_linewidth(.55)
    ax._geographic_frame=True;ax._geographic_xside=xside;ax._geographic_yside=yside

def credit(fig,text=ATTRIBUTION,x=.065,y=.002):
    fig._caption_notes=getattr(fig,'_caption_notes',[])+[text]
