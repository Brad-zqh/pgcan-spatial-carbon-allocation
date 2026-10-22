"""Scientific marks whose positions, shapes and colours encode real quantities."""
import os
import numpy as np
from matplotlib.colors import to_rgba, Normalize,LinearSegmentedColormap
from matplotlib.path import Path
from scipy.stats import gaussian_kde,rankdata
from scipy.optimize import brentq
from matplotlib.patches import Circle,Rectangle
import matplotlib.patheffects as pe

def emphasis_line(artist,color,width=3.0):
    artist.set_path_effects([pe.Stroke(linewidth=width,foreground=color,alpha=.13),pe.Normal()])

def value_bar(ax,value,y,color,height=.46,sd=None,accent=False):
    patch=Rectangle((0,y-height/2),value,height,facecolor='none' if accent else color,edgecolor=color,lw=.4,zorder=2)
    ax.add_patch(patch)
    if accent:
        cmap=LinearSegmentedColormap.from_list('model_accent',['#D992B9','#F6F5EE',color])
        rgba=np.tile(cmap(np.linspace(0,1,160))[None,:,:],(2,1,1))
        _image(ax,rgba,(0,value,y-height/2,y+height/2),zorder=1,clip=patch.get_path().transformed(patch.get_patch_transform()))
        ax._accent_bar_rows=getattr(ax,'_accent_bar_rows',[])+[float(y)]
    if sd is not None and sd>0:ax.errorbar(value,y,xerr=sd,fmt='none',ecolor='#3E4741',lw=.8,capsize=2,zorder=3)
    return patch

def row_tone(color,index,total):
    """Uniform marks, with a light-to-dark progression across ordered rows."""
    rgb=np.array(to_rgba(color)[:3]);strength=.32+.62*index/max(total-1,1)
    return tuple(1-strength*(1-rgb))

def pink_green_row(index,total):
    """One solid colour per row: deep pink, pale pink, pale green, deep green."""
    cmap=LinearSegmentedColormap.from_list('pink_green_rows',
        [(0,'#AD2C74'),(1/3,'#E7AFCE'),(.5,'#F5F4EE'),(2/3,'#C5DEA6'),(1,'#3F742E')],N=256)
    return cmap(index/max(total-1,1))

def _image(ax,rgba,extent,zorder=0,clip=None):
    limits=(ax.get_xlim(),ax.get_ylim());aspect=ax.get_aspect()
    autoscale=(ax.get_autoscalex_on(),ax.get_autoscaley_on())
    image=ax.imshow(rgba,extent=extent,origin='lower',aspect='auto',interpolation='bilinear',zorder=zorder)
    if clip is not None:image.set_clip_path(clip,ax.transData)
    ax.set_xlim(limits[0]);ax.set_ylim(limits[1]);ax.set_aspect(aspect)
    ax.set_autoscalex_on(autoscale[0]);ax.set_autoscaley_on(autoscale[1])
    return image

def signed_violin(ax,values,position,cmap,width=.66):
    values=np.asarray(values)
    violin=ax.violinplot([values],positions=[position],orientation='horizontal',widths=width,showextrema=False)
    body=violin['bodies'][0];body.set(facecolor='none',edgecolor='#8E9A90',linewidth=.5,alpha=.7)
    path=body.get_paths()[0];lo,hi=values.min(),values.max()
    if hi>lo:
        maximum=max(abs(lo),abs(hi),1e-12)
        colours=cmap(Normalize(-maximum,maximum)(np.linspace(lo,hi,256)))
        rgba=np.tile(colours[None,:,:],(48,1,1));rgba[:,:,3]=.60
        _image(ax,rgba,(lo,hi,position-width/2,position+width/2),zorder=1,clip=path)
    q=np.quantile(values,[.25,.5,.75])
    ax.plot(q[[0,2]],[position,position],color='#47534B',lw=1.3,zorder=3)
    ax.scatter(q[1],position,s=15,facecolor='white',edgecolor='#47534B',lw=.65,zorder=4)
    return violin

def overlap_ribbons(ax,left,shared,right,labels=('Regulatory','Energy-user'),compact=False):
    """One equal-area square per member of the union; split squares are shared."""
    counts=[int(left),int(shared),int(right)];cols=16 if compact else 13
    rows=int(np.ceil(sum(counts)/cols));green='#739A59';pink='#BF5796'
    ax.set(xlim=(-.2,cols+.2),ylim=(-7.0 if compact else -5.0,rows+.2),aspect='equal');ax.axis('off')
    categories=np.repeat(np.arange(3),counts)
    for index,kind in enumerate(categories):
        x=index%cols;y=rows-1-index//cols
        if kind==1:
            ax.add_patch(Rectangle((x,y),.45,.90,fc=green,ec='none'))
            ax.add_patch(Rectangle((x+.45,y),.45,.90,fc=pink,ec='none'))
        else:ax.add_patch(Rectangle((x,y),.90,.90,fc=green if kind==0 else pink,ec='none'))
    for row,(name,kind) in enumerate([(labels[0]+' only',0),('Shared',1),(labels[1]+' only',2)]):
        y=-1.5-row*(2.0 if compact else 1.25)
        if kind==1:
            ax.add_patch(Rectangle((0,y-.34),.42,.68,fc=green,ec='none'))
            ax.add_patch(Rectangle((.42,y-.34),.42,.68,fc=pink,ec='none'))
        else:ax.add_patch(Rectangle((0,y-.34),.84,.68,fc=green if kind==0 else pink,ec='none'))
        text=ax.text(1.35,y,name,ha='left',va='center',fontsize=6.5)
        text.set_gid('compact_diagnostic_label')
    ax._set_overlap_geometry={'left_only':counts[0],'shared':counts[1],'right_only':counts[2],
        'first_total':left+shared,'second_total':right+shared,'unit_count':len(categories),
        'encoding':'one equal-area square per union member; shared members use split green/pink squares',
        'numeric_annotations':False,'unit_categories':categories.tolist()}

def density_scatter(ax,x,y,cmap,size=3.4,alpha=.76):
    x=np.asarray(x);y=np.asarray(y);sample=np.vstack([x,y])
    density=gaussian_kde(sample,bw_method='scott')(sample)
    percentile=(rankdata(np.round(np.log(density),10),method='average')-1)/(len(x)-1)*100
    order=np.argsort(density,kind='stable')
    artist=ax.scatter(x[order],y[order],c=percentile[order],cmap=cmap,norm=Normalize(0,100),
        s=size,alpha=alpha,lw=0,rasterized=True)
    return artist,density,percentile
