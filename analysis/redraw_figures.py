"""Reproduce manuscript figures from saved scientific results.

Run: python redraw_figures.py
All task-generated files, matplotlib caches and temporary files stay on E:.
"""
from pathlib import Path
import os

HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE / 'cache')
os.environ['TMP'] = os.environ['TEMP'] = str(HERE / 'cache')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
import json
import hashlib
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm, PowerNorm
from matplotlib.patches import Patch, FancyBboxPatch, Rectangle, Circle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
from PIL import Image, ImageOps, ImageDraw, ImageFont
from pyproj import Transformer
import cartography_style as carto
import map_classification as mc
from visual_polish import complete_heatmap_legends
from figure_finish import compact_rows,compact_columns,align_matrix_bars
from refined_primitives import value_bar,emphasis_line,row_tone

ROOT = Path(os.environ.get('PGCAN_PROJECT_ROOT',HERE.parent/'data/local/project'))
WORK = ROOT / 'submission_revision_20260922'
DATA = WORK / 'corrected'
DOCS = ROOT / '论文撰写0903' / '投稿修订版_20260923'
SOURCE = DOCS / 'Supplementary_Data'
OUT = HERE / 'Figures'
QA = HERE / 'qa'
SD = HERE / 'source_data'
for d in [OUT, QA, SD, HERE/'cache']:
    d.mkdir(exist_ok=True, parents=True)

for name in ['times.ttf', 'timesbd.ttf', 'timesi.ttf', 'timesbi.ttf']:
    font_dir=Path(os.environ.get('PGCAN_FONT_DIR','C:/Windows/Fonts'))
    if (font_dir/name).exists():font_manager.fontManager.addfont(str(font_dir/name))
FONT = 'Times New Roman'
assert Path(font_manager.findfont(FONT)).name.lower() == 'times.ttf'
mpl.rcParams.update({
    'font.family': FONT, 'font.serif': [FONT], 'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 9, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.linewidth': .55,
    'axes.edgecolor': '#7b858b', 'axes.labelcolor': '#000000',
    'text.color': '#000000', 'xtick.color': '#000000', 'ytick.color': '#000000',
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.width': .5, 'ytick.major.width': .5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'legend.frameon': False, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'mathtext.fontset': 'custom', 'mathtext.rm': FONT,
    'mathtext.it': FONT+':italic', 'mathtext.bf': FONT+':bold',
    'savefig.facecolor': 'white', 'figure.dpi': 160,
})
INK='#285B1A'; TEAL='#659D3C'; GOLD='#A4C864'; LILAC='#B31B73'; CORAL='#D779AD'
LIGHT='#F3F6EF'; GRAY='#92998D'
COL={'Nightlights':GRAY,'Enterprise_POI':GOLD,'Clean_prior':LILAC,
     'Direct_mixture':'#D990B8','Full':INK,'Matched_postnorm':CORAL,
     'Matched_softmax':'#578E2D','No_context':'#D578AC','No_graph_features':'#A8005B',
     'No_graph_penalty':'#A4C864','No_graph':'#B31B73'}
LABEL={'Nightlights':'Nightlights','Enterprise_POI':'Enterprise POI','Clean_prior':'Clean prior',
       'Direct_mixture':'Direct mixture','Full':'PGCAN','Matched_postnorm':'Matched post-normalization',
       'Matched_softmax':'Matched softmax','No_context':'No functional one-hot input',
       'No_graph_features':'No graph features','No_graph_penalty':'No graph penalty',
       'No_graph':'No graph components'}
CAT={'industrial':('#BA397E','Industrial'), 'transport_built':('#B4CD76','Transport / built'),
     'research':('#E3A8CA','Research'), 'mixed':('#87AE68','Mixed'),
     'low_activity':('#E8ECDF','Low activity')}
SEQ=LinearSegmentedColormap.from_list('allocation',['#A8005B','#D86BA2','#F0BDDA','#FAFAF5','#DBECBB','#89BD49','#286B12'])
DIFF=LinearSegmentedColormap.from_list('redistribution',['#A8005B','#D86BA2','#F0BDDA','#FAFAF5','#DBECBB','#89BD49','#286B12'])
DENSITY=DIFF.reversed(name='density_green_outer_pink_core')
HEAT=LinearSegmentedColormap.from_list('readable_heat',['#D57BA9','#ECC0D9','#FCFCF9','#D5E8B1','#8FBB55'])
HEAT_DIFF=HEAT
WIDTH=183/25.4
manifest=[]
checks={}

grid=pd.read_csv(DATA/'allocation_500m.csv')
predictors=pd.read_csv(DATA/'predictors_only.csv')
runs=pd.read_csv(DATA/'metrics_runs.csv')
curves=pd.read_csv(DATA/'coverage_curves.csv')
pair=pd.read_csv(DATA/'pairwise_maps.csv')
boot=pd.read_csv(DATA/'conditional_block_resampling.csv')
ns=pd.read_csv(DATA/'new_summary.csv').set_index('configuration')
sc=pd.read_csv(DATA/'prior_scenarios_256.csv')
paired=pd.read_csv(SOURCE/'Data_S10_paired_scenarios.csv')
cat=pd.read_csv(SOURCE/'Data_S8_functional_redistribution.csv')
disc=pd.read_csv(SOURCE/'Data_S9_discordant_cells.csv')
gpkg=DOCS/'图3_ArcGIS独立地图_20260923'/'Figure_3a_Clean_Prior.gpkg'
polys=gpd.read_file(gpkg,layer='grid_values')
boundary=gpd.read_file(gpkg,layer='study_boundary')
assert polys.crs.to_epsg()==32651
assert set(polys.grid_id)==set(grid.grid_id)
polys=polys.set_index('grid_id').loc[grid.grid_id].reset_index()
assert np.allclose(polys.share_pct,grid.Clean_prior_share*100)
assert np.array_equal(grid.grid_id,predictors.grid_id)
assert len(grid)==1980 and grid.grid_id.is_unique
checks.update(grid_cells=len(grid),projection='EPSG:32651',font=FONT,
              exact_grid_alignment=True,prior_geometry_values_match=True)
for k in ['Full','Clean_prior','Direct_mixture']:
    checks[k+'_share_sum']=float(grid[k+'_share'].sum())
    assert np.isclose(grid[k+'_share'].sum(),1)

geometry=polys[['grid_id','geometry']].copy()
geometry.to_file(SD/'plotting_geometry.gpkg',layer='grid',driver='GPKG')
boundary.to_file(SD/'plotting_geometry.gpkg',layer='boundary',driver='GPKG',mode='a')

for name in ['allocation_500m.csv','predictors_only.csv','metrics_runs.csv','coverage_curves.csv',
             'pairwise_maps.csv','conditional_block_resampling.csv','new_summary.csv','new_runs.csv',
             'prior_scenarios_256.csv','sensitivity.csv','new_training_history.json','probabilities_Full.npy']:
    shutil.copyfile(DATA/name, SD/name)
for name in ['Data_S8_functional_redistribution.csv','Data_S9_discordant_cells.csv','Data_S10_paired_scenarios.csv']:
    shutil.copyfile(SOURCE/name, SD/name)
for i in [0,1]:
    shutil.copyfile(WORK/f'locator{i}.png',SD/f'locator{i}.png')

def panel(ax,letter,title,title_x=0):
    ax.set_title(title,loc='left',x=title_x,pad=10,fontweight='normal')
    ax.text(title_x-.075,1.06,letter,transform=ax.transAxes,fontweight='bold',fontsize=10,
            ha='left',va='bottom')

def quiet(ax,axis='y'):
    ax.set_axisbelow(True)
    ax.grid(axis=axis,color='#E5EBED',linewidth=.5)

def foot(fig,text):
    fig._caption_notes=getattr(fig,'_caption_notes',[])+[text]

def save(fig,name,source,caption):
    # User preference: global headings, subtitles and notes belong in the
    # manuscript caption, not inside the exported figure artwork.
    moved=[]
    # Explanatory panel titles belong in the external legend. Preserve only
    # explicitly tagged method headers required to read comparison columns.
    for axis in fig.findobj(mpl.axes.Axes):
        for title_artist in [axis.title,axis._left_title,axis._right_title]:
            title_text=title_artist.get_text()
            if not title_text or title_artist.get_gid()=='comparison_method_header':continue
            letters=[text.get_text() for text in axis.texts if len(text.get_text())==1 and text.get_text().islower() and text.get_text().isascii()]
            if letters:
                fig._panel_descriptions=getattr(fig,'_panel_descriptions',{})
                fig._panel_descriptions.setdefault(letters[0],title_text)
            else:moved.append(title_text)
            title_artist.set_text('')
    for artist in list(fig.texts):
        xx,yy=artist.get_position()
        if yy<.06 or (yy>=.93 and xx<.20):
            moved.append(artist.get_text());artist.remove()
    if getattr(fig,'_suptitle',None) is not None and fig._suptitle in fig.texts:
        moved.append(fig._suptitle.get_text());fig._suptitle.remove()
    map_axes=[ax for ax in fig.findobj(mpl.axes.Axes) if getattr(ax,'_map_classification',None) is not None]
    def signature(norm):
        return (type(norm).__name__,norm.vmin,norm.vmax,getattr(norm,'gamma',None),getattr(norm,'linthresh',None))
    for ax in list(fig.findobj(mpl.axes.Axes)):
        cb=getattr(ax,'_colorbar',None)
        if cb is None or getattr(ax,'_individual_map_bar',False) or type(cb.mappable) is not mpl.cm.ScalarMappable:continue
        matches=[a for a in map_axes if getattr(a,'_requested_map_norm',None) is not None and signature(a._requested_map_norm)==signature(cb.mappable.norm)]
        if matches:
            label=(cb.ax.xaxis.label.get_text() or cb.ax.yaxis.label.get_text()).split(';')[0]
            for a in matches:
                if label and not getattr(a,'_map_units',None):
                    a._individual_map_bar.set_label(label,fontsize=6.5,labelpad=1.5)
                    a._individual_map_bar.ax.xaxis.label.set_gid('map_colourbar_label')
            ax.remove()
    has_basemap=any(getattr(ax,'_has_cartographic_basemap',False) for ax in fig.axes)
    if has_basemap and not getattr(fig,'_basemap_credit',False):
        carto.credit(fig,text=carto.ATTRIBUTION+' | EPSG:32651')
    if has_basemap:
        source=list(dict.fromkeys([*source,'plotting_geometry.gpkg','cartographic_basemap.gpkg','basemap_provenance.json','imagery_basemap_utm51n.tif','imagery_basemap_provenance.json']))
    moved.extend(getattr(fig,'_caption_notes',[]))
    notes=list(dict.fromkeys(x for x in moved if x))
    panel_descriptions=getattr(fig,'_panel_descriptions',{})
    if panel_descriptions:
        notes.insert(0,'Panels: '+'; '.join('('+str(k)+') '+str(val) for k,val in panel_descriptions.items())+'.')
    if notes:caption+=' '+' '.join(notes)
    if map_axes:
        import re
        caption=re.sub(r'(?i)(?:shared|common|symmetric|one) (?:square-root |symmetric-log |linear )colou?r (?:scale|stretch)',
            'panel-specific empirical-quantile colour classes',caption)
        caption=caption.replace('square-root colour stretch','empirical-quantile colour classes')
        caption+=' Each quantitative map has its own colourbar with raw-unit break labels. Map fills use panel-specific empirical quantile classes (negative and positive terciles with a neutral zero class for signed data); colours show within-panel ordering, so absolute magnitudes must be read from each colourbar. No observations or prediction values were changed.'
    fig.canvas.draw()
    # User requires black lettering throughout; annotated matrices therefore
    # use pale fills so black cell labels remain legible at print size.
    all_axes=fig.findobj(mpl.axes.Axes)
    heat_replacements={}
    for ax in all_axes:
        for artist in ax.images:
            array=np.asarray(artist.get_array())
            if array.ndim==2 and max(array.shape)<=100 and not getattr(artist,'_preserve_semantic_cmap',False):
                key=(artist.cmap.name,artist.norm.vmin,artist.norm.vmax)
                replacement=HEAT_DIFF if getattr(artist.norm,'vmin',0)<0 else HEAT
                heat_replacements[key]=replacement;artist.set_cmap(replacement)
    for ax in all_axes:
        cb=getattr(ax,'_colorbar',None)
        if cb is not None:
            key=(cb.mappable.cmap.name,cb.mappable.norm.vmin,cb.mappable.norm.vmax)
            if key in heat_replacements:cb.mappable.set_cmap(heat_replacements[key])
    heatmap_checks=complete_heatmap_legends(fig)
    for artist in fig.findobj(mpl.text.Text):
        artist.set_color('#000000')
        if artist.get_text() and artist.get_gid() not in ['map_coordinate_label','map_north_label','map_scale_label','map_colourbar_label','compact_diagnostic_label']:
            artist.set_fontsize(max(8.2,min(14,artist.get_fontsize()*getattr(fig,'_font_scale',1.20))))
    compactness=compact_rows(fig,name)
    horizontal_compactness=compact_columns(fig,name)
    matrix_bar_alignment=align_matrix_bars(fig)
    if name in ['Figure_10','Figure_12'] and fig.legends:
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        top=max(a.get_tightbbox(renderer).y1 for a in fig.axes if a.get_visible())
        for legend in fig.legends:
            legend.set_loc('lower center');legend.borderaxespad=0
            legend.set_bbox_to_anchor((.5,(top+3/25.4*fig.dpi)/fig.bbox.height),transform=fig.transFigure)
    panel_alignment=[]
    for direction,groups in [('column',getattr(fig,'_aligned_axes_columns',[])),('row',getattr(fig,'_aligned_axes_rows',[]))]:
        for group in groups:
            positions=[a.get_position() for a in group]
            values=[[p.x0,p.width] if direction=='column' else [p.y0,p.height] for p in positions]
            np.testing.assert_allclose(values,np.tile(values[0],(len(values),1)),atol=1e-7)
            panel_alignment.append({'direction':direction,'matched_axes':len(group),'positions':values})
    inset_axes=[a for a in fig.findobj(mpl.axes.Axes) if a not in fig.axes and a.get_visible()]
    for a in inset_axes:a.set_in_layout(True)
    export_extra=[*inset_axes,*[t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text()],
                  *fig.legends,*[a.get_legend() for a in fig.findobj(mpl.axes.Axes) if a.get_legend() is not None]]
    fig.canvas.draw()
    north_checks=[];coordinate_checks=[];coordinate_neighbour_checks=[]
    renderer=fig.canvas.get_renderer()
    legend_clearance=[]
    if name in ['Figure_4','Figure_7','Figure_8','Figure_9','Figure_10','Figure_12']:
        for legend in fig.findobj(mpl.legend.Legend):
            if not legend.get_visible():continue
            lb=legend.get_window_extent(renderer)
            overlaps=[]
            for axis in all_axes:
                if not axis.get_visible() or not axis.axison:continue
                overlap=mpl.transforms.Bbox.intersection(lb,axis.get_window_extent(renderer))
                if overlap is not None and overlap.width>1 and overlap.height>1:overlaps.append([overlap.width,overlap.height])
            assert not overlaps,(name,'Legend overlaps a data frame',overlaps)
            legend_clearance.append({'outside_all_plot_frames':True,'labels':[t.get_text() for t in legend.get_texts()]})
    for source_axis,neighbour_title in getattr(fig,'_layout_clearance_pairs',[]):
        if not neighbour_title.get_text():continue
        title_bounds=neighbour_title.get_window_extent(renderer)
        for label in source_axis.findobj(mpl.text.Text):
            if label.get_visible() and label.get_text():
                assert not label.get_window_extent(renderer).overlaps(title_bounds), (name,'Colourbar text overlaps neighbouring panel title',label.get_text())
    for ax in all_axes:
        if getattr(ax,'_geographic_frame',False):
            ab=ax.get_window_extent(renderer);checked=0
            for axis,labels in [('x',ax.get_xticklabels()),('y',ax.get_yticklabels())]:
                for label in labels:
                    if not label.get_visible() or not label.get_text():continue
                    lb=label.get_window_extent(renderer)
                    outside=(lb.y0>=ab.y1-.1 or lb.y1<=ab.y0+.1) if axis=='x' else (lb.x0>=ab.x1-.1 or lb.x1<=ab.x0+.1)
                    assert outside,(name,'Geographic coordinate label enters its map frame',label.get_text())
                    for other in all_axes:
                        if other is ax or not getattr(other,'_geographic_frame',False):continue
                        ob=other.get_window_extent(renderer)
                        if ob.contains(ab.x0,ab.y0) and ob.contains(ab.x1,ab.y1):continue
                        if ab.contains(ob.x0,ob.y0) and ab.contains(ob.x1,ob.y1):continue
                        assert not lb.overlaps(ob),(name,'Coordinate label overlaps another map',label.get_text())
                    checked+=1
            coordinate_checks.append({'all_labels_outside_frame':True,'visible_coordinate_labels':checked,
                'x_side':ax._geographic_xside,'y_side':ax._geographic_yside})
        for artist in ax.artists:
            if artist.get_gid()=='north_arrow':
                nb=artist.get_window_extent(renderer);ab=ax.get_window_extent(renderer)
                margins=np.array([nb.x0-ab.x0,ab.x1-nb.x1,nb.y0-ab.y0,ab.y1-nb.y1])*72/fig.dpi
                assert np.all(margins>0),(name,'North arrow extends outside the map frame',margins)
                avoid=getattr(ax,'_north_avoid_geometry',None)
                overlap=0.
                if avoid is not None:
                    from shapely.geometry import box
                    xy=ax.transData.inverted().transform([[nb.x0,nb.y0],[nb.x1,nb.y1]])
                    overlap=float(avoid.intersection(box(xy[0,0],xy[0,1],xy[1,0],xy[1,1])).area)
                    assert overlap<1e-4,(name,'North arrow obscures protected map features',overlap)
                assert not artist.patch.get_visible(),(name,'North arrow has a backdrop')
                north_checks.append({'inside_map_frame':True,'minimum_margin_points':round(float(margins.min()),3),'protected_feature_overlap_m2':overlap,'white_backdrop':False})
    # Check all actual matplotlib text objects, including math labels.
    families={tuple(t.get_fontfamily()) for t in fig.findobj(mpl.text.Text) if t.get_text()}
    assert families=={(FONT,)}, (name,families)
    for ext in ['pdf','svg','png']:
        kw={'dpi':600,'bbox_inches':'tight','pad_inches':.035,'bbox_extra_artists':export_extra}
        fig.savefig(OUT/f'{name}.{ext}',**kw)
    # Lossless TIFF uses the same 600 dpi pixels; avoid rendering the identical
    # complex map a second time just to change its raster container.
    with Image.open(OUT/f'{name}.png') as raster:
        raster.save(OUT/f'{name}.tiff',dpi=(600,600),compression='tiff_lzw')
    fig.savefig(QA/f'{name}.png',dpi=180,bbox_inches='tight',pad_inches=.035,bbox_extra_artists=export_extra)
    bbox=fig.get_tightbbox(fig.canvas.get_renderer(),bbox_extra_artists=export_extra)
    import fitz
    with fitz.open(OUT/f'{name}.pdf') as exported:
        export_width=exported[0].rect.width/72*25.4
        export_height=exported[0].rect.height/72*25.4
    (OUT/f'{name}_caption.txt').write_text(caption,encoding='utf-8')
    manifest.append({'figure':name,'source_data':source,'caption':caption,
                     'width_mm':round(export_width,3),'height_mm':round(export_height,3),
                     'dimension_basis':'Actual exported PDF MediaBox; raster glyph metrics can differ slightly',
                     'agg_tight_extent_mm':[round((bbox.width+.07)*25.4,2),round((bbox.height+.07)*25.4,2)],
                     'font':FONT,'dpi':600,'global_title_inside_image':False,'notes_location':'external manuscript caption',
                     'caption_notes':notes,'panel_descriptions':panel_descriptions,'north_arrow_checks':north_checks,
                     'coordinate_label_checks':coordinate_checks,
                     'heatmap_legend_checks':heatmap_checks,
                     'density_encoding':getattr(fig,'_density_encoding',None),
                     'typography':{'statistical_text_minimum_pt':8.2,'general_size_multiplier':getattr(fig,'_font_scale',1.20),'coordinate_labels_separately_sized':True},
                     'map_furniture_white_backdrop':False,
                     'individual_map_legends':[{'units':a._individual_map_bar.ax.xaxis.label.get_text(),**a._map_classification} for a in map_axes],
                     'palette':('Pink - pale neutral - green; density: green low percentile to pink high percentile'
                         if getattr(fig,'_density_encoding',{}).get('type')=='KDE percentile' else
                         'Pink - pale neutral - green; density: green low counts to pink high counts'),
                     'layout':getattr(fig,'_layout',None)})
    manifest[-1]['compactness']=compactness
    manifest[-1]['horizontal_compactness']=horizontal_compactness
    manifest[-1]['matrix_bar_alignment']=matrix_bar_alignment
    manifest[-1]['set_overlap_checks']=[a._set_overlap_geometry for a in all_axes if hasattr(a,'_set_overlap_geometry')]
    manifest[-1]['panel_alignment']=panel_alignment
    manifest[-1]['legend_clearance']=legend_clearance
    plt.close(fig)
    print('Saved',name,flush=True)

bounds=boundary.total_bounds
cx=(bounds[0]+bounds[2])/2; cy=(bounds[1]+bounds[3])/2
sx=bounds[2]-bounds[0]; sy=bounds[3]-bounds[1]
_map_bounds=carto.context_bounds(bounds)
MAPX=(_map_bounds[0],_map_bounds[2])
MAPY=(_map_bounds[1],_map_bounds[3])
BASEMAP=carto.load_basemap(SD/'cartographic_basemap.gpkg')

def scale(ax):
    carto.map_furniture(ax,length=6000)

def mapplot(ax,values=None,norm=None,cmap=SEQ,categories=False,extent=None):
    requested_norm=norm
    classification=None
    if values is not None and not categories and not isinstance(norm,mpl.colors.BoundaryNorm):
        norm,cmap,ticks,classification=mc.classify(values,cmap)
    if extent is None and not getattr(ax,'_external_coordinate_gutter',False):
        # Reserve physical room for outside coordinates before placing the
        # fixed-size north needle or calculating colourbar positions.
        pos=ax.get_position(original=True);fw,fh=ax.figure.get_size_inches()*25.4
        ax.set_position([pos.x0+6/fw,pos.y0,pos.width-6/fw,pos.height-4/fh])
        ax._external_coordinate_gutter=True
    ax._has_cartographic_basemap=True
    protected=boundary.geometry.union_all()
    if getattr(ax,'_north_avoid_geometry',None) is not None:protected=protected.union(ax._north_avoid_geometry)
    ax._north_avoid_geometry=protected
    x0,x1,y0,y1=extent if extent is not None else (*MAPX,*MAPY)
    carto.draw_basemap(ax,BASEMAP,(x0,x1,y0,y1))
    if categories:
        colors=[CAT[v][0] for v in values]
        polys.plot(ax=ax,color=colors,edgecolor='none',linewidth=0,alpha=carto.THEME_ALPHA,zorder=2)
    elif values is not None:
        polys.assign(value=np.asarray(values)).plot(column='value',ax=ax,cmap=cmap,norm=norm,
                                                   edgecolor='none',linewidth=0,alpha=carto.THEME_ALPHA,zorder=2,
                                                   missing_kwds={'color':'#D1D1D1'})
    else:
        pass
    boundary.boundary.plot(ax=ax,color='#627B80',linewidth=.55,zorder=3)
    ax.set_xlim(x0,x1);ax.set_ylim(y0,y1);ax.set_aspect('equal');carto.compact_graticule(ax);scale(ax)
    if classification is not None:
        ax._requested_map_norm=requested_norm
        ax._display_map_norm=norm;ax._display_map_cmap=cmap
        mc.add_bar(ax.figure,ax,norm,cmap,ticks,getattr(ax,'_map_units','Value; quantile classes'),classification)

def colorbar(fig,ax,cmap,norm,label,ticks=None):
    if getattr(ax,'_individual_map_bar',None) is not None:
        ax._individual_map_bar.set_label(label.replace('; square-root stretch','').replace('; square-root',''),fontsize=6.5,labelpad=1.5)
        ax._individual_map_bar.ax.xaxis.label.set_gid('map_colourbar_label')
        return ax._individual_map_bar
    cax=ax.inset_axes([.16,-.10,.68,.045])
    cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',ticks=ticks)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2,labelsize=6.5,pad=2)
    cb.set_label(label,fontsize=7,labelpad=3)

def fig1():
    from study_area_map import draw_study_area
    return draw_study_area()

def card(ax,x,y,w,h,title,body,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009,rounding_size=0.015',
                               facecolor=color,edgecolor='#CFDCE0',lw=.7))
    ax.text(x+.022,y+h-.044,title,fontsize=8.4,fontweight='bold',va='top')
    ax.text(x+.022,y+h-.112,body,fontsize=7.1,va='top',linespacing=1.55)

def arrow(ax,p,q,label=None):
    ax.annotate('',xy=q,xytext=p,arrowprops={'arrowstyle':'-|>','lw':.85,'color':'#637F8B',
                                          'shrinkA':2,'shrinkB':2})
    if label:ax.text((p[0]+q[0])/2,(p[1]+q[1])/2+.015,label,fontsize=6.5,ha='center')

def fig2():
    f=plt.figure(figsize=(WIDTH,5.4));a=f.add_axes([.035,.07,.93,.90]);a.axis('off')
    a.set_xlim(0,1);a.set_ylim(0,1)
    a.text(.01,.98,'Prior-guided coordinate-conditioned allocation',fontsize=11,fontweight='bold',va='top')
    a.text(.01,.92,'Spatial predictors → constructed supervision → normalized shares → conditional amounts',fontsize=7.5,color='#63747D')
    card(a,.01,.63,.285,.225,'a  Spatial predictors',
         'Nightlights · enterprise POIs\nBuildings · roads · patents\nLand-cover fractions; coordinates', '#EEF4F5')
    card(a,.355,.63,.285,.225,'b  Constructed supervision',
         'Functional prior  p₀; activity  u\nTarget  t = 0.70 p₀ + 0.30 u\nPredictor-derived operational classes', '#ECF3EF')
    card(a,.70,.63,.285,.225,'c  Spatial residual network',
         'Coordinate-conditioned coefficients\nFunctional context; 8-neighbour graph\nResidual  r; graph regularization', '#F0EFF5')
    arrow(a,(.295,.745),(.355,.745));arrow(a,(.640,.745),(.700,.745))
    card(a,.355,.305,.285,.235,'d  Normalized share allocation',
         'q = log(max(p₀, ε)) + r\np = softmax(q);  ∑p = 1\nNon-negative regional shares', '#EAF2F5')
    card(a,.70,.305,.285,.235,'e  Accounting conversion',
         'Adopted regional total  C = 15 Mt\nConditional cell amount  Cᵢ = C × pᵢ\nMatched outputs and baselines', '#F7F1E7')
    card(a,.01,.305,.285,.235,'Input preparation',
         'Min–max scores for proxy design\nStandardized network predictors\nProjected coordinates (UTM 51N)', '#F6F8F8')
    # Orthogonal connectors avoid implying that site membership trains the model.
    a.plot([.842,.842,.498],[.63,.583,.583],c='#637F8B',lw=.85)
    arrow(a,(.498,.583),(.498,.54));arrow(a,(.640,.422),(.700,.422))
    a.plot([.153,.153,.330,.330],[.630,.578,.578,.422],c='#637F8B',lw=.85)
    arrow(a,(.330,.422),(.355,.422))
    card(a,.01,.018,.975,.185,'f  Evaluation and spatial diagnostics',
         '146 regulatory sites; 69 energy-user sites (59 shared) · list membership withheld from training\n'
         'Coverage · paired comparisons · spatial overlap · conditional block resampling · heuristic sensitivity', '#F7F8F6')
    arrow(a,(.842,.305),(.842,.203))
    a.text(.018,.252,'No observed grid-level emissions labels; benchmark coordinates retain shared POI geolocation.',
           fontsize=7,color='#68787E')
    foot(f,'Conceptual information flow | Exact scientific labels remain editable in SVG and PDF')
    save(f,'Figure_2',[], 'Information flow separates proxy supervision, normalized share estimation, regional accounting and location evaluation.')

def fig2_rich():
    """Illustrated technical workflow with true source-map and result thumbnails."""
    if (HERE/'Framework_API_MAX.png').is_file():
        from publish_framework import export_framework
        manifest.append(export_framework(HERE))
        print('Saved Figure_2: full API rendering, quality=max, native resolution.',flush=True)
        return
    f=plt.figure(figsize=(WIDTH,6.65))
    base=f.add_axes([0,0,1,1]);base.axis('off');base.set(xlim=(0,1),ylim=(0,1))
    base.text(.04,.978,'Technical workflow of proxy-supervised carbon allocation',fontsize=10.6,
              fontweight='bold',va='top')
    # The user's examples motivate coloured stage bands and nested scientific visuals.
    def stage(x,y,w,h,number,title,color):
        base.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.004,rounding_size=0.012',
                                      fc='#FCFDFD',ec='#CAD8DD',lw=.7))
        base.add_patch(FancyBboxPatch((x,y+h-.052),w,.052,boxstyle='round,pad=0.004,rounding_size=0.01',
                                      fc=color,ec='none'))
        base.text(x+.012,y+h-.026,f'{number}  {title}',fontsize=8.2,fontweight='bold',va='center')
    stage(.04,.51,.31,.42,'1','Multi-source predictors','#CCE1E8')
    stage(.395,.65,.235,.28,'2','Prior and supervision','#DBE7D3')
    stage(.675,.51,.285,.42,'3','Spatial neural model','#DDD8EC')
    stage(.04,.075,.435,.35,'4','Normalized allocation and accounting','#C8E0DA')
    stage(.52,.075,.44,.35,'5','Evaluation and spatial diagnostics','#EFDFC6')
    def tiny_map(rect,values,label,cmap=SEQ,norm=None):
        a=f.add_axes(rect)
        polys.assign(value=np.asarray(values)).plot(column='value',ax=a,cmap=cmap,
                                                   norm=norm,edgecolor='none',linewidth=0)
        boundary.boundary.plot(ax=a,color='#829498',linewidth=.3)
        a.set_xlim(*MAPX);a.set_ylim(*MAPY);a.set_aspect('equal');a.axis('off')
        a.text(.5,1.01,label,transform=a.transAxes,ha='center',fontsize=6.4)
        return a
    for i,(k,lab) in enumerate([('ntl_sum','Nightlights'),('enterprise_poi_density','Enterprise POIs'),
                                ('floor_area_proxy','OSM buildings'),('road_density_km_per_km2','Road network')]):
        z=np.log1p(predictors[k].to_numpy());z=z/z.max() if z.max() else z
        tiny_map([.052+(i%2)*.146,.705-(i//2)*.125,.14,.105],z,lab,norm=Normalize(0,1))
    base.text(.058,.548,'Patents · land-cover fractions · coordinates',fontsize=6.5)
    base.text(.058,.526,'Common nominal 500 m grid; 1,980 cells',fontsize=6.5,color='#667A84')
    tiny_map([.402,.768,.105,.07],grid.Clean_prior_share,'Prior p₀',norm=PowerNorm(.5,0,.009))
    tiny_map([.515,.768,.105,.07],grid.Activity_share,'Activity u',norm=PowerNorm(.5,0,.009))
    base.text(.5125,.728,'t = 0.70 p₀ + 0.30 u',ha='center',fontsize=8.4,fontweight='bold')
    base.text(.5125,.685,'Constructed target; no observed\ngrid-level emissions labels',ha='center',
              fontsize=6.7,linespacing=1.35,color='#5A7079')
    # Label-free API artwork is illustrative; equations and labels are real vector text.
    art=HERE/'Framework_AI_artwork.png'
    if art.exists():
        a=f.add_axes([.715,.655,.205,.17]);a.imshow(Image.open(art))
        a.set_xlim(535,1020);a.set_ylim(700,235);a.axis('off')
    else:
        a=f.add_axes([.715,.655,.205,.17]);a.axis('off')
        nodes=np.array([[0,0],[1,0],[2,0],[.5,1],[1.5,1],[1,2]])
        for i,j in [(0,1),(1,2),(0,3),(1,3),(1,4),(2,4),(3,4),(3,5),(4,5)]:
            a.plot(nodes[[i,j],0],nodes[[i,j],1],c=TEAL,lw=.8)
        a.scatter(nodes[:,0],nodes[:,1],c=INK,s=25)
    base.text(.8175,.642,'Coordinate-conditioned residual r',ha='center',fontsize=7,fontweight='bold')
    base.text(.692,.606,'Standardized predictors + functional context',fontsize=6.5)
    base.text(.692,.578,'8-neighbour graph features + regularization',fontsize=6.5)
    base.text(.692,.540,'Prior-guided logits: q = log(max(p₀, ε)) + r',fontsize=6.5)
    # Deliberately orthogonal arrows reproduce the reference's strong reading order.
    arrow(base,(.35,.79),(.395,.79))
    arrow(base,(.63,.79),(.675,.79))
    base.text(.652,.815,'Target t',fontsize=6,ha='center',color='#5F7781')
    base.plot([.8175,.8175,.255],[.51,.469,.469],c='#5C7A87',lw=.9)
    arrow(base,(.255,.469),(.255,.425))
    base.text(.528,.479,'Residual + prior → normalized shares',ha='center',fontsize=6.6,color='#526E7B')
    arrow(base,(.475,.25),(.52,.25))
    # Actual allocation is shown without a miniature colour bar; its source is Figure 3.
    tiny_map([.052,.165,.252,.145],grid.Full_share*100,'PGCAN mean-share map',norm=PowerNorm(.5,0,.9))
    base.text(.322,.324,'p = softmax(q)',fontsize=7.5,fontweight='bold')
    base.text(.322,.289,'pᵢ ≥ 0;  ∑pᵢ = 1',fontsize=7.2)
    base.text(.322,.240,'Adopted total',fontsize=6.4,color='#61777F')
    base.text(.322,.210,'C = 15 Mt',fontsize=8.8,fontweight='bold')
    base.text(.322,.163,'Cᵢ = C × pᵢ',fontsize=8.8,fontweight='bold')
    base.text(.058,.113,'Shares are estimated; amounts are conditional on the adopted total.',fontsize=6.3)
    # True coverage curves and Jaccard matrix are subordinate evidence thumbnails.
    a=f.add_axes([.55,.174,.18,.14])
    for k in ['Clean_prior','Direct_mixture','Full']:
        z=curves[curves.method==k].sort_values('fraction')
        a.plot(z.fraction*100,z.primary_coverage*100,c=COL[k],lw=.9 if k=='Full' else .7)
    a.set(xlim=(0,50),ylim=(0,100),xticks=[0,25,50],yticks=[0,50,100])
    a.tick_params(labelsize=5.6,length=2,pad=1);a.set_xlabel('Top-ranked cells (%)',fontsize=6,labelpad=1)
    a.set_ylabel('Coverage (%)',fontsize=6,labelpad=1);a.set_title('Location screening',fontsize=6.7,pad=4);quiet(a)
    b=f.add_axes([.785,.174,.135,.14]);keys=['Clean_prior','Direct_mixture','Full']
    mat=np.ones((3,3))
    for i,k in enumerate(keys):
        for j,l in enumerate(keys):
            if i!=j:
                r=pair[((pair['first']==k)&(pair.second==l))|((pair['first']==l)&(pair.second==k))].iloc[0]
                mat[i,j]=r.jaccard
    b.imshow(mat,cmap=SEQ,vmin=0,vmax=1)
    for i in range(3):
        for j in range(3):b.text(j,i,f'{mat[i,j]:.2f}',fontsize=5.5,ha='center',va='center',color='white')
    b.set_xticks([0,1,2],['Prior','Mix','Net'],fontsize=5.6)
    b.set_yticks([0,1,2],['Prior','Mix','Net'],fontsize=5.6);b.tick_params(length=0,pad=2)
    for sp in b.spines.values():sp.set_visible(False)
    b.set_title('Spatial disagreement',fontsize=6.7,pad=4)
    base.text(.538,.350,'146 regulatory sites; 69 energy-user sites (59 shared)',fontsize=6.4)
    base.text(.538,.108,'Paired comparisons · block resampling · ablations · scenarios',fontsize=6.3)
    foot(f,'List membership withheld from training; shared POI geolocation remains | Artwork is conceptual; inset maps / curves use saved results')
    save(f,'Figure_2',['allocation_500m.csv','predictors_only.csv','coverage_curves.csv','pairwise_maps.csv'],
         'Illustrated five-stage workflow. Insets reuse saved spatial proxies, allocation, coverage and Jaccard results. Graph artwork is conceptual; all text is editable Times New Roman.')

def fig3():
    full=grid.Full_share.to_numpy()*100; prior=grid.Clean_prior_share.to_numpy()*100
    delta=full-prior; sd=np.load(DATA/'probabilities_Full.npy').std(0,ddof=1)*100
    hi=np.ceil(max(full.max(),prior.max())*10)/10
    dh=np.ceil(abs(delta).max()*100)/100; sh=np.ceil(sd.max()*1000)/1000
    f,axs=plt.subplots(2,2,figsize=(WIDTH,5.15))
    f.subplots_adjust(left=.06,right=.975,top=.94,bottom=.14,hspace=.55,wspace=.13)
    parts=[(prior,'Clean prior',PowerNorm(.5,0,hi),SEQ,'Regional share (%; square-root stretch)'),
           (full,'PGCAN mean map',PowerNorm(.5,0,hi),SEQ,'Regional share (%; square-root stretch)'),
           (delta,'Neural redistribution',TwoSlopeNorm(0,-dh,dh),DIFF,'Mean map − prior (pp)'),
           (sd,'Across-seed variation',PowerNorm(.5,0,sh),SEQ,'Sample SD (pp; square-root stretch)')]
    for i,(a,(z,t,n,c,lab)) in enumerate(zip(axs.flat,parts)):
        ticks=[-dh,0,dh] if i==2 else [0,hi/4,hi] if i<2 else [0,sh/4,sh]
        mapplot(a,z,n,c);panel(a,chr(97+i),t);colorbar(f,a,c,n,lab,ticks)
    checks['Figure_3_shared_max_pct']=float(hi);checks['Figure_3_difference_limit_pp']=float(dh)
    foot(f,'Shared extent and grid support | Five neural seeds | SD measures seed variation, not emissions uncertainty')
    pd.DataFrame({'grid_id':grid.grid_id,'prior_pct':prior,'pgcan_pct':full,'difference_pp':delta,'seed_sd_pp':sd}).to_csv(SD/'Figure_3_values.csv',index=False)
    save(f,'Figure_3',['Figure_3_values.csv','plotting_geometry.gpkg','probabilities_Full.npy'],
         'Prior and neural mean share one square-root-stretched colour scale; seed SD uses the same stretch. Tick labels report original percentages / pp. Redistribution is linear and centred on zero.')

def forest(ax,keys,frame,labelmap=LABEL,colors=COL,column='primary_coverage',lo=0,hi=65,show_points=True):
    for i,k in enumerate(keys):
        if isinstance(frame.index,pd.MultiIndex):raise ValueError('Unexpected frame index')
        vals=frame.loc[frame.method==k,column].to_numpy()*100
        mu=vals.mean();sd=vals.std(ddof=1) if len(vals)>1 else 0
        c=colors.get(k,INK)
        if show_points and len(vals)>1:
            ax.scatter(vals,np.full(len(vals),i)+np.linspace(-.13,.13,len(vals)),s=9,
                       facecolor='white',edgecolor=c,lw=.5,zorder=3)
        ax.errorbar(mu,i,xerr=sd,fmt='o',color=c,mec='white',mew=.5,ms=4.6,
                    capsize=2,elinewidth=1.1,zorder=4)
        text=f'{mu:.2f}' if len(vals)==1 else f'{mu:.2f} ± {sd:.2f}'
        ax.text(1.025,i,text,transform=ax.get_yaxis_transform(),fontsize=6.8,va='center')
    ax.set_yticks(range(len(keys)),[labelmap[k] for k in keys],fontsize=7)
    ax.invert_yaxis();ax.set_xlim(lo,hi);ax.tick_params(axis='y',length=0,pad=5)
    ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color('#A8B6BC')
    ax.set_xlabel('Site coverage (%)');quiet(ax,'x')

def fig4():
    f=plt.figure(figsize=(WIDTH,4.1));a=f.add_axes([.08,.25,.365,.64]);b=f.add_axes([.69,.25,.21,.64])
    keys=['Nightlights','Enterprise_POI','Clean_prior','Direct_mixture','Full']
    for k in keys:
        z=curves[curves.method==k].sort_values('fraction')
        a.plot(z.fraction*100,z.primary_coverage*100,c=COL[k],lw=1.65 if k=='Full' else 1.05,
               label=LABEL[k],zorder=4 if k=='Full' else 2,
               linestyle='-' if k in ['Full','Enterprise_POI','Clean_prior'] else '--')
    a.axvline(10,color='#B9C4C9',ls=':',lw=.8);a.set(xlim=(0,50),ylim=(0,102),
               xlabel='Top-ranked cells (%)',ylabel='Regulatory-site coverage (%)');quiet(a)
    f.legend(handles=a.get_legend_handles_labels()[0],labels=a.get_legend_handles_labels()[1],
             loc='lower left',bbox_to_anchor=(.065,.075),ncol=3,columnspacing=1,handlelength=1.7)
    panel(a,'a','Coverage across thresholds')
    keys+=['Matched_postnorm','Matched_softmax']
    forest(b,keys,runs);panel(b,'b','Top-10% coverage')
    foot(f,'Curves: fixed mean maps | Points: per-run mean ± sample seed SD; open circles: individual seeds')
    save(f,'Figure_4',['coverage_curves.csv','metrics_runs.csv'],
         'Coverage versus selection threshold and Top-10% coverage; summary markers show per-run mean ± sample SD, with individual seed points.')

def fig5():
    keys=['Nightlights','Enterprise_POI','Clean_prior','Direct_mixture','Full','Matched_postnorm','Matched_softmax']
    short=['NTL','POI','Prior','Mix','PGCAN','Post','Soft']
    mat=np.ones((7,7))
    for i,k in enumerate(keys):
        for j,l in enumerate(keys):
            if i!=j:
                r=pair[((pair['first']==k)&(pair.second==l))|((pair['first']==l)&(pair.second==k))].iloc[0]
                mat[i,j]=r.jaccard
    f=plt.figure(figsize=(WIDTH,4.25));a=f.add_axes([.075,.26,.365,.61]);b=f.add_axes([.69,.26,.275,.61])
    image=a.imshow(mat,cmap=SEQ,vmin=0,vmax=1)
    for i in range(7):
        for j in range(7):
            a.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=7,
                   color='white' if mat[i,j]>.67 else INK)
    a.set(xticks=np.arange(7),yticks=np.arange(7),xticklabels=short,yticklabels=short)
    a.tick_params(length=0);a.set_xticks(np.arange(-.5,7),minor=True);a.set_yticks(np.arange(-.5,7),minor=True)
    a.grid(which='minor',color='white',lw=.6);a.tick_params(which='minor',length=0)
    for s in a.spines.values():s.set_visible(False)
    # Outline PGCAN comparisons without changing matrix values.
    a.add_patch(Rectangle((-.5,3.5),7,1,fill=False,ec=INK,lw=.8))
    panel(a,'a','Top-198 spatial overlap')
    cb=f.colorbar(image,cax=f.add_axes([.12,.17,.275,.025]),orientation='horizontal')
    cb.outline.set_visible(False);cb.set_label('Jaccard overlap',fontsize=7,labelpad=2);cb.ax.tick_params(labelsize=6.5,length=2)
    others=[k for k in keys if k!='Full']
    b.axvline(0,c='#9AABB3',lw=.7,ls='--')
    for i,k in enumerate(others):
        if i%2==0:b.axhspan(i-.45,i+.45,color='#F5F7F8',lw=0)
        for block,offset,c,marker in [(1000,-.11,INK,'o'),(2000,.11,LILAC,'s')]:
            r=boot[(boot.second==k)&(boot.block_m==block)].iloc[0]
            b.plot([r['lower_2.5'],r['upper_97.5']],[i+offset]*2,c=c,lw=1.1)
            b.scatter(r.difference_pp,i+offset,s=15,marker=marker,c=c,ec='white',lw=.3,zorder=3)
    b.set_yticks(range(6),[LABEL[k].replace('Matched post-normalization','Post-normalization') for k in others],fontsize=7)
    b.invert_yaxis();b.set_xlim(-7,25);b.set_xticks([-5,0,5,10,15,20,25]);b.tick_params(axis='y',length=0)
    b.spines['left'].set_visible(False);quiet(b,'x');b.set_xlabel('PGCAN − comparator (pp)')
    panel(b,'b','Paired coverage differences')
    b.legend(handles=[Line2D([],[],c=INK,marker='o',lw=1,label='1 km blocks'),Line2D([],[],c=LILAC,marker='s',lw=1,label='2 km blocks')],
             loc='upper center',bbox_to_anchor=(.44,-.22),ncol=2,columnspacing=1)
    foot(f,'Intervals: 95% conditional block resampling (5,000 draws); maps held fixed | Post / Soft: matched neural outputs')
    save(f,'Figure_5',['pairwise_maps.csv','conditional_block_resampling.csv'],
         'Top-198 Jaccard overlap and paired coverage differences; bars show saved 95% conditional block-resampling intervals at 1 km and 2 km.')

def fig6():
    f=plt.figure(figsize=(WIDTH,6.2))
    a=f.add_axes([.085,.63,.335,.285]);b=f.add_axes([.72,.63,.18,.285])
    c=f.add_axes([.085,.145,.335,.285]);d=f.add_axes([.65,.145,.30,.285])
    hist=json.loads((DATA/'new_training_history.json').read_text())
    for k,color,lab in [('Control',INK,'λ = 0.025'),('Prior_005',LILAC,'λ = 0.005'),('Prior_000',CORAL,'λ = 0')]:
        tr=np.array([hist[f'{k}_{s}']['trace'] for s in range(20260918,20260923)])
        vals=tr[:,:,-1];mean=vals.mean(0);sd=vals.std(0,ddof=1)
        a.fill_between(tr[0,:,0],np.maximum(mean-sd,1e-6),mean+sd,color=color,alpha=.12,lw=0)
        a.plot(tr[0,:,0],mean,c=color,lw=1.3,label=lab)
    a.set_yscale('log');a.set(xlim=(0,900),ylim=(.004,.65),xlabel='Epoch',ylabel='TV from constructed target')
    a.set_xticks([0,300,600,900]);a.set_yticks([.005,.01,.05,.1,.5],['0.005','0.01','0.05','0.1','0.5']);quiet(a)
    a.legend(loc='upper left',bbox_to_anchor=(-.02,-.23),ncol=3,handlelength=1.2,columnspacing=.7)
    panel(a,'a','Approximation to constructed target')
    rows=[('Control','Control (λ = 0.025)'),('Prior_005','λ = 0.005'),('Prior_000','λ = 0'),('No_POI_lineage','No POI / patents')]
    seed_runs=pd.read_csv(DATA/'new_runs.csv')
    for i,(k,lab) in enumerate(rows):
        r=ns.loc[k];col=[INK,LILAC,CORAL,TEAL][i]
        values=seed_runs.loc[seed_runs.configuration.eq(k),'primary_coverage'].to_numpy()*100
        np.testing.assert_allclose([values.mean(),values.std(ddof=1)],[r.coverage*100,r.sd*100],atol=1e-10)
        value_bar(b,r.coverage*100,i,row_tone(INK,i,len(rows)),height=.48,sd=r.sd*100,accent=True)
        b.text(1.04,i,f'{r.coverage*100:.3f}',transform=b.get_yaxis_transform(),fontsize=6.8,va='center')
    b.set_yticks(range(4),[v[1] for v in rows],fontsize=7);b.set(ylim=(3.6,-.6),xlim=(0,60))
    b.set_xticks([0,20,40,60]);b.set_xlabel('Coverage (%)');b.tick_params(axis='y',length=0);b.spines['left'].set_visible(False);quiet(b,'x')
    panel(b,'b','Regularization / information paths',title_x=-1)
    z=c.scatter(sc.top10_mass_share*100,sc.primary_coverage*100,c=sc.tv_target,cmap=SEQ,
                s=14,alpha=.85,edgecolors='white',linewidths=.2)
    c.set(xlabel='Top-10% mass (%)',ylabel='Regulatory-site coverage (%)');quiet(c)
    c.text(.04,.94,'256 prior scenarios',transform=c.transAxes,fontsize=7,va='top')
    cc=f.colorbar(z,cax=c.inset_axes([.10,-.27,.70,.045]),orientation='horizontal');cc.outline.set_visible(False)
    cc.ax.tick_params(labelsize=5.5,length=1);cc.set_label('TV from scenario target',fontsize=6,labelpad=1)
    panel(c,'c','Joint heuristic scenarios: priors')
    rr=paired[paired.method=='PGCAN mean map'].sort_values('scenario')
    for field,color,off,mark,lab in [('coverage_change_vs_prior_pp',INK,-.12,'o','Versus own prior'),
                                   ('coverage_change_vs_target_pp',LILAC,.12,'s','Versus own target')]:
        xx=rr.scenario+off;vv=rr[field]
        stems=d.vlines(xx,0,vv,colors=color,lw=.8,alpha=.85)
        emphasis_line(stems,color,width=3.2)
        d.scatter(xx,vv,s=20,marker=mark,c=color,edgecolors='white',lw=.4,label=lab,zorder=3)
    d.axhline(0,c='#8E9DA4',lw=.7);quiet(d);d.set_xticks([0,3,6,9,11])
    d.set(xlabel='Scenario identifier',ylabel='Coverage difference (pp)',xlim=(-.6,11.6))
    d.legend(loc='upper center',bbox_to_anchor=(.5,-.23),ncol=2,handlelength=1,columnspacing=.7)
    panel(d,'d','Paired neural-minus-baseline coverage')
    foot(f,'Training lines: five-seed mean; ribbons / bars: sample seed SD | Scenario ranges describe design sensitivity')
    save(f,'Figure_6',['new_training_history.json','new_summary.csv','prior_scenarios_256.csv','Data_S10_paired_scenarios.csv'],
         'Constructed-target approximation, regularization changes and paired heuristic scenarios. Shaded training ribbons and capped coverage intervals show sample seed SD. Bounded bars in b show five-seed mean coverage. All four bars use a decorative pink-to-green internal gradient; green endpoints progress from light to dark by row, independent of coverage magnitude. Thin translucent under-strokes in d follow the exact stems and do not represent uncertainty.')

def figs1():
    f,axs=plt.subplots(2,2,figsize=(WIDTH,5.15));f.subplots_adjust(left=.06,right=.975,top=.94,bottom=.14,hspace=.55,wspace=.13)
    names=[('ntl_sum','Nightlight sum'),('enterprise_poi_density','Enterprise POI density'),
           ('floor_area_proxy','OSM building floor-area proxy'),('road_density_km_per_km2','Road density')]
    vals={'grid_id':grid.grid_id}
    for i,(a,(k,title)) in enumerate(zip(axs.flat,names)):
        z=np.log1p(predictors[k].to_numpy());z=z/z.max() if z.max() else z;vals[k+'_display']=z
        mapplot(a,z,Normalize(0,1));panel(a,chr(97+i),title)
        colorbar(f,a,SEQ,Normalize(0,1),'log(1 + value), panel-rescaled',[0,.5,1])
    foot(f,'Common nominal support | Within-panel relative intensity; colours do not imply comparable activity units')
    pd.DataFrame(vals).to_csv(SD/'Figure_S1_display_values.csv',index=False)
    save(f,'Figure_S1',['predictors_only.csv','Figure_S1_display_values.csv','plotting_geometry.gpkg'],
         'Four spatial proxies, preserving log(1+x) and panel-maximum scaling for visualization only.')

def figs2():
    f=plt.figure(figsize=(WIDTH,3.95));a=f.add_axes([.265,.27,.255,.60]);b=f.add_axes([.69,.27,.275,.60])
    keys=['Full','No_context','No_graph_features','No_graph_penalty','No_graph']
    z=runs[runs.method.isin(keys)].primary_coverage*100
    forest(a,keys,runs,lo=np.floor(z.min())-1,hi=np.ceil(z.max())+1)
    panel(a,'a','Component ablations')
    sens=pd.read_csv(DATA/'sensitivity.csv');base=runs[(runs.method=='Full')&(runs.seed==20260918)].iloc[0]
    vals={float(r.value):(r.primary_coverage*100,r.top10_mass_share*100) for _,r in sens[sens.parameter=='strength'].iterrows()}
    vals[1]=(base.primary_coverage*100,base.top10_mass_share*100)
    xx=sorted(vals)
    for i,col,mark,lab in [(0,INK,'o','Site coverage'),(1,LILAC,'s','Top-10% mass')]:
        b.plot(xx,[vals[x][i] for x in xx],c=col,marker=mark,ms=3.6,mec='white',mew=.4,lw=1.2,label=lab)
    b.axvline(1,c='#ADBCC3',ls=':',lw=.8)
    b.text(1.025,.94,'Default',transform=b.get_xaxis_transform(),fontsize=6.5,color='#6F7D84',ha='left')
    b.set(xlabel='Functional-weight strength',ylabel='Percentage',xlim=(-.06,1.56),ylim=(0,70));b.set_xticks([0,.5,1,1.5]);quiet(b)
    b.legend(loc='upper center',bbox_to_anchor=(.5,-.23),ncol=2,columnspacing=.7,handlelength=1.3)
    panel(b,'b','Functional-weight sensitivity')
    foot(f,'Ablations: five-seed mean ± sample SD; open circles: individual seeds | Sensitivity: fixed seed 20260918')
    save(f,'Figure_S2',['metrics_runs.csv','sensitivity.csv'],
         'Component ablation means ± seed SD and individual runs; multiplier sensitivity at the original fixed seed.')

def figs3():
    f=plt.figure(figsize=(WIDTH,3.8));a=f.add_axes([.04,.27,.49,.60]);b=f.add_axes([.72,.27,.23,.60])
    mapplot(a);panel(a,'a','Discordant Top-198 selections')
    tr=Transformer.from_crs(4326,32651,always_xy=True)
    for lab,c,mark in [('PGCAN only',INK,'o'),('Prior only',LILAC,'s')]:
        z=disc[disc.selection_status==lab];xx,yy=tr.transform(z.center_lon.to_numpy(),z.center_lat.to_numpy())
        a.scatter(xx,yy,c=c,s=17,marker=mark,edgecolor='white',lw=.45,label=f'{lab} (n = {len(z)})',zorder=4)
    a.legend(loc='upper center',bbox_to_anchor=(.50,-.07),ncol=2,columnspacing=1,handlelength=1)
    order=['mixed','industrial','low_activity','transport_built','research']
    z=cat.set_index('category').loc[order]; b.axvline(0,c='#9AABB3',lw=.7)
    b.barh(np.arange(5),z.net_change_pp,color=[TEAL if v>=0 else LILAC for v in z.net_change_pp],height=.47)
    for i,(_,r) in enumerate(z.iterrows()):
        b.text(1.04,i,f'{r.net_change_pp:+.2f}',transform=b.get_yaxis_transform(),fontsize=7,va='center')
    b.set_yticks(np.arange(5),[CAT[k][1] for k in order],fontsize=7);b.invert_yaxis();b.set_xlim(-4.8,4.8)
    b.set_xticks([-4,-2,0,2,4]);b.tick_params(axis='y',length=0);b.spines['left'].set_visible(False);quiet(b,'x')
    b.set_xlabel('PGCAN − prior (pp of total)');panel(b,'b','Net redistribution by category')
    foot(f,'Standalone spatial diagnostic | Selection status and category totals are unchanged from the saved source tables')
    save(f,'Figure_S3',['Data_S8_functional_redistribution.csv','Data_S9_discordant_cells.csv','plotting_geometry.gpkg'],
         'Discordant Top-198 selections and net redistribution by predictor-derived category; standalone diagnostic.')

def montage():
    files=[QA/(m['figure']+'.png') for m in manifest]
    board=Image.new('RGB',(2100,1650),'#E9EFF0');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/timesbd.ttf',24)
    for i,p in enumerate(files):
        x=i%3*700;y=i//3*550;draw.text((x+15,y+10),p.stem,font=font,fill=INK)
        im=ImageOps.contain(Image.open(p).convert('RGB'),(675,495))
        board.paste(im,(x+int((700-im.width)/2),y+42))
    board.save(HERE/'Figure_overview.png',dpi=(180,180))

if __name__=='__main__':
    for fn in [fig1,fig2_rich,fig3,fig4,fig5,fig6,figs1,figs2,figs3]:fn()
    montage()
    (HERE/'figure_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    checks['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SD.iterdir() if p.is_file()}
    (QA/'data_checks.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding='utf-8')
    print('All figures exported; data alignment and font checks passed.',flush=True)
