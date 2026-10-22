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
