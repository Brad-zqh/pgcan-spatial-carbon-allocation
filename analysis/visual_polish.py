"""Explicit count-density scaling and complete quantitative heatmap legends."""
import os
import numpy as np
from matplotlib.colors import FuncNorm, Normalize


def density_norm(counts):
    """Raw count 1 is green; count 2 enters pink, higher counts darken.

    The two logarithmic segments retain the actual maximum and every bin.
    This intentionally expands low-count differences rather than changing bins.
    """
    maximum=float(np.max(counts))
    if maximum<=2:return Normalize(1,max(maximum,1.001))
    knots=np.log1p([1.,2.,maximum]);positions=np.array([0.,.72,1.])
    def forward(x):return np.interp(np.log1p(np.maximum(x,0)),knots,positions)
    def inverse(y):return np.expm1(np.interp(y,positions,knots))
    return FuncNorm((forward,inverse),vmin=1,vmax=maximum)


def tag_small_text(cb):
    cb.ax.tick_params(labelsize=6.2,length=1.5,pad=1)
    for label in [*cb.ax.get_xticklabels(),*cb.ax.get_yticklabels(),cb.ax.xaxis.label,cb.ax.yaxis.label]:
        label.set_gid('map_colourbar_label')


def complete_heatmap_legends(fig):
    """Add a legend only to a scalar matrix that has none; never to imagery."""
    records=[]
    axes=list(fig.findobj(__import__('matplotlib').axes.Axes))
    for ax in axes:
        if getattr(ax,'_geographic_frame',False) or getattr(ax,'_colorbar',None):continue
        for im in ax.images:
            array=np.asarray(im.get_array())
            if array.ndim!=2 or max(array.shape)>100:continue
            existing=[getattr(a,'_colorbar',None) for a in fig.findobj(__import__('matplotlib').axes.Axes)]
            cb=next((c for c in existing if c is not None and c.mappable is im),None)
            if cb is None:
                binary=getattr(ax,'_heatmap_binary',False)
                cax=ax.inset_axes([.30,1.13,.65,.025] if binary else [1.035,.03,.035,.94])
                cb=fig.colorbar(im,cax=cax,orientation='horizontal' if binary else 'vertical')
                cb.outline.set_visible(False)
                units=getattr(ax,'_heatmap_units','Scaled value')
                if not binary:cb.set_label(units,fontsize=6.2,labelpad=2)
                if binary:
                    cb.set_ticks([0,1],labels=['Off','On'])
                    cb.ax.xaxis.set_ticks_position('top')
                else:cb.set_ticks(np.linspace(im.norm.vmin,im.norm.vmax,3))
            tag_small_text(cb)
            if getattr(ax,'_heatmap_ticks',None) is not None:
                cb.set_ticks(ax._heatmap_ticks)
                if getattr(ax,'_heatmap_ticklabels',None) is not None:cb.set_ticklabels(ax._heatmap_ticklabels)
                tag_small_text(cb)
            records.append({'shape':list(array.shape),'units':cb.ax.xaxis.label.get_text() or cb.ax.yaxis.label.get_text(),
                            'limits':[float(im.norm.vmin),float(im.norm.vmax)],'has_colourbar':True})
    return records
