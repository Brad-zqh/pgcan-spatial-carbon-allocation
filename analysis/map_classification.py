"""Explicit panel-specific quantile classes, retaining raw values and zero sign."""
import os
import numpy as np
from matplotlib.colors import BoundaryNorm,ListedColormap

def classify(values,cmap):
    z=np.asarray(values,float);z=z[np.isfinite(z)]
    lo=float(z.min());hi=float(z.max())
    if lo==hi:
        eps=max(abs(lo)*1e-6,1e-10);edges=np.array([lo-eps,hi+eps]);positions=[.5]
        scheme='Constant value; no spatial variation'
    elif lo<0<hi:
        eps=float(np.min(np.abs(z[z!=0])))*.25
        negative=np.unique(np.r_[lo,np.quantile(z[z<0],[1/3,2/3]),-eps])
        positive=np.unique(np.r_[eps,np.quantile(z[z>0],[1/3,2/3]),hi])
        edges=np.r_[negative,positive]
        positions=np.r_[np.linspace(.04,.36,len(negative)-1),.5,np.linspace(.64,.96,len(positive)-1)]
        scheme='Separate negative/positive terciles; zero sign boundary'
    elif lo==0 and np.any(z==0):
        positive=z[z>0];eps=float(positive.min())*.5
        levels=np.unique(np.quantile(positive,np.linspace(0,1,7)))
        if len(levels)==1:
            edges=np.array([-eps,eps,hi]);positions=[.04,.96]
        else:
            edges=np.r_[-eps,eps,levels[1:]]
            positions=np.r_[.5,np.linspace(.04,.96,len(levels)-1)]
        scheme='Zero separated; positive-value sextiles; duplicate boundaries collapsed'
    else:
        edges=np.unique(np.quantile(z,np.linspace(0,1,8)))
        positions=np.linspace(.04,.96,len(edges)-1)
        scheme='Empirical septiles; duplicate boundaries collapsed'
    norm=BoundaryNorm(edges,len(edges)-1,clip=True)
    cm=ListedColormap(cmap(positions),name='panel_empirical_pink_green')
    ticks=[edges[0],edges[len(edges)//2],edges[-1]]
    if lo==0:ticks=[0,float(np.median(z[z>0])) if hi>0 else 0,hi]
    if lo<0<hi:ticks=[edges[0],0,edges[-1]]
    return norm,cm,np.unique(ticks),{'scheme':scheme,'boundaries':edges.tolist(),'class_counts':np.bincount(norm(z),minlength=len(edges)-1).tolist(),'minimum':lo,'maximum':hi,
        'negative_cells':int((z<0).sum()),'positive_cells':int((z>0).sum()),'zero_cells':int((z==0).sum())}

def format_tick(value,span):
    if value==0:return '0'
    if abs(value)<.001:return f'{value:.1e}'
    digits=min(6,max(2,int(np.ceil(-np.log10(max(span,1e-12))))+1))
    formatted=f'{value:.{digits}f}'.rstrip('0').rstrip('.')
    return f'{value:.2g}' if formatted in ('0','-0','') else formatted

def add_bar(fig,ax,norm,cm,ticks,label,description):
    import matplotlib as mpl
    cax=ax.inset_axes([.13,-.14,.78,.045]);cax._individual_map_bar=True
    cb=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cm),cax=cax,orientation='horizontal',ticks=ticks,spacing='uniform')
    labels=[format_tick(float(x),float(norm.vmax-norm.vmin)) for x in ticks]
    # A narrow entropy/gate range may otherwise print both median and maximum
    # as "1". Increase precision until distinct numeric ticks remain distinct.
    for digits in range(3,9):
        if len(set(labels))==len(ticks):break
        labels=[f'{float(x):.{digits}f}'.rstrip('0').rstrip('.') if x else '0' for x in ticks]
    assert len(set(labels))==len(ticks),(label,ticks,labels)
    cb.set_ticklabels(labels)
    cb.outline.set_visible(False);cb.ax.tick_params(length=2,pad=1,labelsize=6.5)
    cb.set_label(label,fontsize=6.5,labelpad=1.5)
    for text in [*cb.ax.get_xticklabels(),cb.ax.xaxis.label]:text.set_gid('map_colourbar_label')
    ax._individual_map_bar=cb;ax._map_classification=description
    return cb
