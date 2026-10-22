"""Compact complete panel rows while preserving their physical plotting sizes."""
import os
import numpy as np


def compact_rows(fig, name, threshold_mm=10.0, retained_mm=6.0):
    """Remove only empty inter-row bands, never space inside an axes or its labels.

    Tight bounds include inset colourbars, tick labels, panel letters and legends.
    Axes retain their physical width/height, data limits, aspect and all artists.
    Figure 13 has an explicitly approved 42 mm scatter contract and is retained.
    """
    if name=='Figure_13':
        return {'applied':False,'reason':'Approved single-scatter layout retained'}
    fig.canvas.draw();renderer=fig.canvas.get_renderer();dpi=fig.dpi
    width,height=fig.get_size_inches()
    root_axes=list(fig.axes)
    records=[];bands=[]
    for ax in root_axes:
        bounds=ax.get_tightbbox(renderer)
        if bounds is not None and np.isfinite(bounds.extents).all():
            bands.append([bounds.y0/dpi,bounds.y1/dpi])
        p=ax.get_position()
        records.append((ax,np.array([p.x0*width,p.y0*height,p.width*width,p.height*height])))
    texts=[]
    for t in fig.texts:
        if t.get_visible() and t.get_text():
            b=t.get_window_extent(renderer);bands.append([b.y0/dpi,b.y1/dpi])
            texts.append((t,np.array(t.get_position())*np.array([width,height])))
    legends=[]
    for legend in fig.legends:
        b=legend.get_window_extent(renderer)
        bands.append([b.y0/dpi,b.y1/dpi]);legends.append((legend,b.x0/dpi,b.y0/dpi))
    if not bands:return {'applied':False,'reason':'No panel bounds'}
    merged=[]
    for bottom,top in sorted(bands):
        if merged and bottom<=merged[-1][1]:merged[-1][1]=max(top,merged[-1][1])
        else:merged.append([bottom,top])
    removals=[]
    for lower,upper in zip(merged,merged[1:]):
        gap=upper[0]-lower[1]
        if gap*25.4>threshold_mm:
            removals.append((lower[1],upper[0],gap-retained_mm/25.4))
    if not removals:return {'applied':False,'removed_mm':0,'gaps':[]}
    def shift(y):return y-sum(delta for bottom,top,delta in removals if y>=top-1e-7)
    removed=sum(x[2] for x in removals);new_height=height-removed
    fig.set_size_inches(width,new_height,forward=True)
    for ax,(x,y,w,h) in records:
        ax.set_position([x/width,shift(y)/new_height,w/width,h/new_height])
    for text,(x,y) in texts:text.set_position([x/width,shift(y)/new_height])
    for legend,x,y in legends:
        legend.set_loc('lower left');legend.borderaxespad=0
        legend.set_bbox_to_anchor((x/width,shift(y)/new_height),transform=fig.transFigure)
    for ax,(_,_,w,h) in records:
        pos=ax.get_position()
        actual=np.array([pos.width*width,pos.height*new_height])
        np.testing.assert_allclose(actual,[w,h],rtol=0,atol=1e-5,
                                   err_msg='Panel physical dimensions changed during compaction')
    return {'applied':True,'original_canvas_height_mm':round(height*25.4,2),
        'canvas_height_mm':round(new_height*25.4,2),'removed_mm':round(removed*25.4,2),
        'retained_gap_mm':retained_mm,'panel_physical_dimensions_preserved':True,
        'gaps':[{'before_mm':round((top-bottom)*25.4,2),'after_mm':retained_mm} for bottom,top,_ in removals]}


def compact_columns(fig,name,threshold_mm=9.,retained_mm=4.):
    if name in ['Figure_13','Figure_1']:return {'applied':False,'removed_mm':0}
    fig.canvas.draw();renderer=fig.canvas.get_renderer();dpi=fig.dpi
    width,height=fig.get_size_inches();records=[];bands=[]
    for ax in fig.axes:
        b=ax.get_tightbbox(renderer)
        if b is not None:bands.append([b.x0/dpi,b.x1/dpi])
        pos=ax.get_position();records.append((ax,[pos.x0*width,pos.y0*height,pos.width*width,pos.height*height]))
    extras=[]
    for artist in [*fig.texts,*fig.legends]:
        if not artist.get_visible():continue
        b=artist.get_window_extent(renderer);bands.append([b.x0/dpi,b.x1/dpi]);extras.append((artist,b))
    merged=[]
    for left,right in sorted(bands):
        if merged and left<=merged[-1][1]:merged[-1][1]=max(right,merged[-1][1])
        else:merged.append([left,right])
    remove=[(a[1],b[0],b[0]-a[1]-retained_mm/25.4) for a,b in zip(merged,merged[1:]) if (b[0]-a[1])*25.4>threshold_mm]
    if not remove:return {'applied':False,'removed_mm':0}
    shift=lambda x:x-sum(d for a,b,d in remove if x>=b-1e-7)
    new_width=width-sum(d for _,_,d in remove);fig.set_size_inches(new_width,height,forward=True)
    for ax,(x,y,w,h) in records:ax.set_position([shift(x)/new_width,y/height,w/new_width,h/height])
    for artist,b in extras:
        if artist in fig.legends:
            artist.set_loc('lower left');artist.borderaxespad=0
            artist.set_bbox_to_anchor((shift(b.x0/dpi)/new_width,b.y0/dpi/height),transform=fig.transFigure)
        else:
            x,y=artist.get_position();artist.set_position([shift(x*width)/new_width,y])
    return {'applied':True,'removed_mm':round((width-new_width)*25.4,2),'retained_gap_mm':retained_mm}


def align_matrix_bars(fig):
    """Match a heatmap bar's quantitative long edge to the matrix itself."""
    fig.canvas.draw();checks=[]
    for ax in fig.findobj(__import__('matplotlib').axes.Axes):
        if getattr(ax,'_geographic_frame',False) or getattr(ax,'_colorbar',None):continue
        for im in ax.images:
            array=np.asarray(im.get_array())
            if array.ndim!=2 or max(array.shape)>100:continue
            cb=next((getattr(a,'_colorbar',None) for a in fig.findobj(__import__('matplotlib').axes.Axes)
                if getattr(a,'_colorbar',None) is not None and a._colorbar.mappable is im),None)
            if cb is None:continue
            p=ax.get_position();c=cb.ax.get_position();cb.ax.set_axes_locator(None)
            if cb.orientation=='horizontal':cb.ax.set_position([p.x0,c.y0,p.width,c.height])
            else:cb.ax.set_position([c.x0,p.y0,c.width,p.height])
            cb.ax.set_in_layout(True)
            fig.canvas.draw();actual=cb.ax.get_position();p=ax.get_position()
            delta=([actual.x0-p.x0,actual.x1-p.x1] if cb.orientation=='horizontal'
                   else [actual.y0-p.y0,actual.y1-p.y1])
            assert max(abs(v) for v in delta)<1e-8,('Matrix/bar alignment',delta)
            checks.append({'shape':list(array.shape),'orientation':cb.orientation,
                'long_edge_aligned':True,'edge_delta_in_figure_coordinates':delta})
    return checks
