"""Compact locator pair, dominant thematic map and subordinate detail strip."""
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from matplotlib.lines import Line2D
from pyproj import Transformer
from shapely.geometry import box,MultiPoint
import redraw_figures as v
import cartography_style as c

AEA='+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs'
WIDTH_MM=183
HEIGHT_MM=133

def axes_mm(fig,x,y,w,h):
    return fig.add_axes([x/WIDTH_MM,y/HEIGHT_MM,w/WIDTH_MM,h/HEIGHT_MM])

def frame(ax):
    ax.set_axis_on()
    for spine in ax.spines.values():spine.set(visible=True,color='#657174',linewidth=.45)

def panel(ax,letter,description):
    ax.text(.018,.975,letter,transform=ax.transAxes,ha='left',va='top',
        fontsize=8.5,fontweight='bold',color='black',zorder=40,
        bbox={'fc':'white','ec':'none','alpha':.9,'pad':.65})
    ax.figure._panel_descriptions=getattr(ax.figure,'_panel_descriptions',{})
    ax.figure._panel_descriptions[letter]=description

def bounds_geo(gdf,extent):
    a,b,c_,d=extent
    return gpd.clip(gdf,gpd.GeoDataFrame(geometry=[box(a,b,c_,d)],crs=4326),keep_geom_type=True)

def fit_extent(bounds,ratio,fill=.84):
    x0,y0,x1,y1=bounds;cx=(x0+x1)/2;cy=(y0+y1)/2
    sx=max((x1-x0)/fill,(y1-y0)/fill*ratio);sy=sx/ratio
    return (cx-sx/2,cx+sx/2,cy-sy/2,cy+sy/2)

def draw_study_area():
    # Discrete categories share exactly the same keys in maps, legend and bars.
    v.CAT.update({'mixed':('#87AE68','Mixed'),
        'industrial':('#BA397E','Industrial'),
        'low_activity':('#E8ECDF','Low activity'),
        'research':('#E3A8CA','Research'),
        'transport_built':('#B4CD76','Transport / built')})
    country=gpd.read_file(v.SD/'locator_context.gpkg',layer='countries')
    province=gpd.read_file(v.SD/'locator_context.gpkg',layer='provinces')
    zhejiang=province[province.name.eq('Zhejiang')].copy()
    author_province=gpd.read_file(v.SD/'china_locator_user.gpkg',layer='provinces')
    china=gpd.read_file(v.SD/'china_locator_user.gpkg',layer='national_polygons')
    outline=gpd.read_file(v.SD/'china_locator_user.gpkg',layer='national_outline')
    nine=gpd.read_file(v.SD/'china_locator_user.gpkg',layer='nine_dash')
    za=author_province[author_province.adcode.eq(330000)]
    prefectures=gpd.read_file(v.SD/'administrative_context.gpkg',layer='zhejiang_prefectures')
    districts=gpd.read_file(v.SD/'administrative_context.gpkg',layer='hangzhou_districts')
    sites=pd.read_csv(v.DATA/'validation_primary_exact_sites.csv')
    energy=pd.read_csv(v.DATA/'energy_benchmark_sites.csv')
    tr=Transformer.from_crs(4326,32651,always_xy=True)
    rxy=tr.transform(sites.longitude.to_numpy(),sites.latitude.to_numpy())
    exy=tr.transform(energy['wgs84_经度'].to_numpy(),energy['wgs84_纬度'].to_numpy())
    points=MultiPoint(np.vstack([np.column_stack(rxy),np.column_stack(exy)]))
    f=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4))

    # Two equally wide locators; actual geographic aspect is preserved.
    a=axes_mm(f,9,86,55,40);a.set_facecolor('#EAF1F3')
    bounds_geo(country,(55,0,150,62)).to_crs(AEA).plot(ax=a,color='#E7EAE3',edgecolor='#BFC7BD',linewidth=.18,zorder=0)
    cp=china.to_crs(AEA)
    cp.plot(ax=a,color='#FAFCF6',edgecolor='none',zorder=1)
    author_province.to_crs(AEA).boundary.plot(ax=a,color='#A7AEA2',linewidth=.22,zorder=2)
    za.to_crs(AEA).plot(ax=a,color=v.CORAL,edgecolor='#343434',linewidth=.4,zorder=3)
    outline.to_crs(AEA).plot(ax=a,color='#343434',linewidth=.36,zorder=4)
    mainland=cp.clip(gpd.GeoDataFrame(geometry=[box(70,17,138,56)],crs=4326).to_crs(AEA))
    ex=list(fit_extent(mainland.total_bounds,55/40,.74))
    # Reserve a southern context strip for furniture; keep the full mainland
    # and Taiwan above it. Preserve the projected geographic aspect.
    shift=(ex[3]-ex[2])*.105
    ex[2]-=shift;ex[3]-=shift
    a.set(xlim=ex[:2],ylim=ex[2:],aspect='equal')
    a._north_avoid_geometry=cp.geometry.union_all()
    c.projected_graticule(a,[80,100,120],[20,35,50],crs=AEA,fontsize=7.2)
    c.north_arrow(a,crs=AEA,height=11)
    c.scale_bar(a,length=1500000,anchor=(.065,.075))
    panel(a,'a','China provincial context, with Zhejiang highlighted')
    a.text(.48,.57,'CHINA',transform=a.transAxes,ha='center',fontsize=6.8,fontstyle='italic')
    a.annotate('Zhejiang',xy=Transformer.from_crs(4326,AEA,always_xy=True).transform(120,29),
        xytext=(.72,.36),textcoords='axes fraction',fontsize=5.8,
        arrowprops={'arrowstyle':'-','lw':.4,'color':'black'})
    sea=a.inset_axes([.795,.055,.14,.225]);sea.set_facecolor('#EFF2EF')
    bounds_geo(country,(104,2,125,25)).plot(ax=sea,color='#E4E7E2',edgecolor='#BEC7BB',linewidth=.16)
    china.plot(ax=sea,color='#FAFCF6',edgecolor='none',zorder=1)
    outline.plot(ax=sea,color='#343434',linewidth=.24,zorder=2)
    nine.plot(ax=sea,color='#343434',edgecolor='#343434',linewidth=.22,zorder=3)
    sea.set(xlim=(104,125),ylim=(2,25),aspect='equal',xticks=[],yticks=[])
    c.projected_graticule(sea,[115],[10,20],crs=4326,fontsize=5.8,xside='bottom',yside='right',degree_only=True)
    for sp in sea.spines.values():sp.set(visible=True,color='#657174',linewidth=.35)
    frame(a)
    f.canvas.draw()
    main_geometry=mainland.geometry.union_all()
    assert box(ex[0],ex[2],ex[1],ex[3]).covers(main_geometry)
    sb=sea.get_window_extent(f.canvas.get_renderer())
    corners=a.transData.inverted().transform([[sb.x0,sb.y0],[sb.x1,sb.y1]])
    sea_overlap=float(main_geometry.intersection(box(*corners.ravel())).area)
    scale_overlap=float(main_geometry.intersection(a._scale_bar_boxes[-1]).area)
    assert sea_overlap<1e-4 and scale_overlap<1e-4,(sea_overlap,scale_overlap)
    national_qa={'mainland_taiwan_hainan_inside_frame':True,
        'south_china_sea_inset':'bottom right','scale':'bottom left',
        'inset_overlap_main_geometry_m2':sea_overlap,'scale_overlap_main_geometry_m2':scale_overlap,
        'projected_extent':ex,'southern_context_extended':True}

    a=axes_mm(f,9,40,55,43);a.set_facecolor('#EAF1F3')
    bounds_geo(country,(116,26,125,33)).to_crs(32651).plot(ax=a,color='#EEF1E9',edgecolor='#A9B5AF',linewidth=.25)
    province.to_crs(32651).plot(ax=a,color='#F5F6F0',edgecolor='#9EAEA7',linewidth=.3)
    prefectures.plot(ax=a,color='#E8F0DF',edgecolor='#77836D',linewidth=.38,zorder=2)
    prefectures[prefectures.adcode.eq(330100)].plot(ax=a,color='#EBCBDD',edgecolor=v.LILAC,linewidth=.5,zorder=3)
    districts.boundary.plot(ax=a,color='#7F767B',linewidth=.23,linestyle=(0,(2,1)),zorder=4)
    prefectures.boundary.plot(ax=a,color='#66725D',linewidth=.38,zorder=5)
    exb=fit_extent(zhejiang.to_crs(32651).total_bounds,55/43,.84)
    a.set(xlim=exb[:2],ylim=exb[2:],aspect='equal');a._north_avoid_geometry=prefectures.geometry.union_all()
    c.projected_graticule(a,[118,120,122],[28,29,30,31],fontsize=7.2,xside='bottom')
    a.tick_params(axis='x',top=False,labeltop=False,bottom=True,labelbottom=True,pad=1.5)
    for lab in a.get_xticklabels():lab.set_va('top')
    c.north_arrow(a,height=11);c.scale_bar(a,length=100000,anchor=(.085,.105))
    v.boundary.plot(ax=a,color=v.INK,edgecolor=v.INK,linewidth=.3,zorder=6)
    mark=v.boundary.geometry.union_all().centroid
    a.scatter([mark.x],[mark.y],s=9,c=v.GOLD,ec='white',lw=.35,zorder=7)
    a.annotate('Hangzhou',xy=(mark.x,mark.y),xytext=(.50,.84),textcoords='axes fraction',fontsize=6.1,
        arrowprops={'arrowstyle':'-','lw':.45,'color':'black'},zorder=8)
    for code,label in [(330200,'Ningbo'),(330300,'Wenzhou'),(330700,'Jinhua')]:
        pt=prefectures.loc[prefectures.adcode.eq(code),'geometry'].iloc[0].representative_point()
        a.text(pt.x,pt.y,label,fontsize=5.5,ha='center',va='center',zorder=8,
            bbox={'fc':'white','ec':'none','alpha':.7,'pad':.4})
    a.legend(handles=[Line2D([],[],c='#66725D',lw=.55,label='City'),
        Line2D([],[],c='#7F767B',lw=.35,ls='--',label='District / county')],
        loc='lower right',bbox_to_anchor=(.98,.012),fontsize=5.2,handlelength=1.0,
        handletextpad=.4,labelspacing=.3,borderaxespad=0,frameon=True,facecolor='white',edgecolor='none',framealpha=.9)
    panel(a,'b','Zhejiang prefectures and Hangzhou district/county context');frame(a)

    # The dominant map and its compact supporting strip form one right-hand block.
    a=axes_mm(f,67,65,107,61)
    a._north_avoid_geometry=points.buffer(100)
    exc=fit_extent(v.boundary.total_bounds,107/61,.85)
    v.mapplot(a,v.grid.function_class,categories=True,extent=exc)
    c.projected_graticule(a,[120.3,120.5,120.7],[30.25,30.35,30.45],fontsize=8,yside='right')
    a.tick_params(axis='y',left=False,labelleft=False,right=True,labelright=True,pad=2)
    for lab in a.get_yticklabels():lab.set_ha('left')
    a.scatter(*rxy,s=8,facecolors='none',edgecolors=v.INK,lw=.48,zorder=8)
    a.scatter(*exy,s=8,c=v.GOLD,marker='^',edgecolors='white',lw=.2,zorder=9)
    minx,miny,maxx,maxy=v.boundary.total_bounds
    zoom=(minx+500,minx+12500,(miny+maxy)/2-2700,(miny+maxy)/2+2700)
    a.add_patch(Rectangle((zoom[0],zoom[2]),zoom[1]-zoom[0],zoom[3]-zoom[2],
        fc='none',ec=v.LILAC,lw=.7,ls=(0,(3,2)),zorder=10))
    a.text(zoom[0]+200,zoom[3]-180,'d',fontsize=7,fontweight='bold',va='top',
        bbox={'fc':'white','ec':'none','pad':.65},zorder=11)
    panel(a,'c','Functional context and withheld benchmark locations');frame(a)
    f.add_artist(Rectangle((67/WIDTH_MM,40/HEIGHT_MM),107/WIDTH_MM,25/HEIGHT_MM,
        transform=f.transFigure,fc='white',ec='#657174',lw=.45,zorder=-1))

    a=axes_mm(f,74,43.5,38,17.1);a._has_cartographic_basemap=True
    c.draw_basemap(a,v.BASEMAP,zoom,detail=True)
    v.boundary.boundary.plot(ax=a,color='#778F93',lw=.45,zorder=3)
    a.scatter(*rxy,s=7,facecolors='none',edgecolors=v.INK,lw=.4,zorder=8)
    a.scatter(*exy,s=7,c=v.GOLD,marker='^',edgecolors='white',lw=.2,zorder=9)
    a.set(xlim=zoom[:2],ylim=zoom[2:],aspect='equal',xticks=[],yticks=[])
    c.projected_graticule(a,[120.32,120.38],[30.30,30.32],fontsize=6.3,xside='bottom')
    a._north_avoid_geometry=v.boundary.geometry.union_all().boundary.buffer(80).union(points.buffer(100))
    c.scale_bar(a,length=2000,anchor=(.07,.13));c.north_arrow(a,height=8)
    panel(a,'d','Street and building detail of the marked western window');frame(a)

    legend=axes_mm(f,117,41.5,55.5,21.8);legend.axis('off')
    legend.text(.015,.98,'Functional classes',fontsize=6.4,fontweight='bold',va='top')
    cats=['mixed','industrial','low_activity','research','transport_built']
    leg=legend.legend(handles=[Patch(fc=v.CAT[k][0],label=v.CAT[k][1]) for k in cats],
        loc='upper left',bbox_to_anchor=(0,.86),ncol=2,fontsize=6.0,handlelength=.9,
        handletextpad=.4,columnspacing=.8,labelspacing=.35,borderaxespad=0)
    legend.add_artist(leg)
    handles=[Line2D([],[],marker='o',markerfacecolor='none',mec=v.INK,ls='',label='Regulatory sites (146)',ms=2.7),
        Line2D([],[],marker='^',c=v.GOLD,ls='',label='Energy-user sites (69)',ms=2.9)]
    legend.legend(handles=handles,loc='lower left',bbox_to_anchor=(0,.015),fontsize=6,
        handletextpad=.4,labelspacing=.3,borderaxespad=0)

    # Taller categorical composition panel: no interpolation between classes.
    a=axes_mm(f,36,7,138,24)
    f.text(9/WIDTH_MM,31/HEIGHT_MM,'e',fontsize=8.5,fontweight='bold',va='top')
    f._panel_descriptions['e']='Category composition across cells, regulatory sites and allocated mass'
    raw=np.array([[(v.grid.function_class==k).sum() for k in cats],
        [v.grid.loc[v.grid.function_class.eq(k),'primary_count'].sum() for k in cats],
        [v.grid.loc[v.grid.function_class.eq(k),'Full_share'].sum() for k in cats]],float)
    percent=raw/raw.sum(1,keepdims=True)*100;left=np.zeros(3)
    for j,k in enumerate(cats):
        a.barh(range(3),percent[:,j],left=left,height=.63,color=v.CAT[k][0],ec='white',lw=.4)
        for i,val in enumerate(percent[:,j]):
            if val>9:a.text(left[i]+val/2,i,f'{val:.1f}%',ha='center',va='center',fontsize=7.2)
        left+=percent[:,j]
    a.set_yticks(range(3),['Grid cells','Regulatory sites','Allocated mass'],fontsize=7.0)
    a.invert_yaxis();a.set(xlim=(0,100),xticks=[0,25,50,75,100],xlabel='Composition (%)')
    a.xaxis.label.set_size(7);a.xaxis.labelpad=2
    a.tick_params(axis='y',length=0);a.tick_params(axis='x',labelsize=6.5,length=2,pad=2)
    a.spines['left'].set_visible(False)
    c.credit(f,text=c.ATTRIBUTION+'; national boundaries, Taiwan, islands and nine-dash geometry: author-supplied data; regional background: Natural Earth; city/district context: DataV GeoAtlas')
    f._basemap_credit=True
    f._layout={'canvas_mm':[183,133],
        'hierarchy':'Stacked left locators; dominant right thematic block; inset detail; taller discrete-category composition panel',
        'composition_height_mm':24,'category_palette':'Discrete pink and green families; common map/legend/bar keys',
        'national_furniture':{'south_china_sea_inset':'bottom right','scale':'bottom left','southern_context_extended':True},
        'primary_map_frame_mm':[107,61],'locator_frames_mm':[[55,40],[55,43]],
        'left_and_right_block_top_bottom_mm':[126,40],
        'national_place_labels':['CHINA','Zhejiang'],
        'geographic_aspect_preserved':True,'locator_long_dimension_fill':[.74,.84],
        'thematic_width_fraction':.85,'map_furniture_white_backdrop':False}
    pd.DataFrame(percent,index=['grid_cells','regulatory_sites','allocated_mass'],columns=cats).to_csv(v.SD/'Analysis_category_composition.csv')
    v.save(f,'Figure_1',['allocation_500m.csv','validation_primary_exact_sites.csv','energy_benchmark_sites.csv',
        'Analysis_category_composition.csv','locator_context.gpkg','administrative_context.gpkg',
        'administrative_context_provenance.json','china_locator_user.gpkg','china_locator_user_provenance.json'],
        'China and Zhejiang locators are stacked on the left; the dominant thematic map, compact street-detail inset and shared legend form the right block. The taller composition panel uses discrete category colours consistent with the thematic map and legend. National context extends southward; the scale is at lower left and the South China Sea inset, including author-supplied islands and nine-dash geometry, is at lower right. Geographic aspect is preserved in every frame, with separate context extents adapted to each map. Zhejiang includes 11 prefecture-level units and 13 Hangzhou district/county-level units. The provincial context uses approximately inverse-transformed DataV GeoAtlas geometry; it does not replace the analytical boundary. Source imagery and OSM vectors are contextual only. Needles show local geographic north; scale bars use projected distances.')
    entry=v.manifest[-1]
    (v.QA/'study_area_layout_verification.json').write_text(json.dumps({'status':'PASS',
        'layout':entry['layout'],'north_arrows':entry['north_arrow_checks'],'national':national_qa,
        'visual_review':'Taller composition panel, discrete matched category colours; complete national locator above a furniture strip; dominant map and detail kept at their geographic aspect.'},indent=2),encoding='utf-8')

if __name__=='__main__':draw_study_area()
