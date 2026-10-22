"""Scientific marks whose positions, shapes and colours encode real quantities."""
import os
import numpy as np
from matplotlib.colors import to_rgba, Normalize
from matplotlib.path import Path
from scipy.stats import gaussian_kde,rankdata

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
    total=left+shared+right
    ax.set(xlim=(0,total),ylim=(-.08,1.1));ax.axis('off')
    # Aligned set intervals: lengths encode exact set cardinalities.
    ax.fill_between([left,left+shared],.30,.70,color='#A9B2AD',alpha=.14,lw=0,zorder=0)
    for x0,x1,y,color in [(0,left+shared,.78,'#658C55'),(left,total,.22,'#B64686')]:
        ax.fill_between([x0,x1],y-.105,y+.105,color=color,alpha=.32,lw=.7,edgecolor=color)
    for xx in [left,left+shared]:ax.plot([xx,xx],[.08,.92],color='#8D9691',lw=.5,ls=':',zorder=2)
    for y,name,count in [(.78,labels[0],left+shared),(.22,labels[1],right+shared)]:
        ax.text(-.035,y,name,transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=7.0)
        ax.text(1.02,y,str(count),transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=7.0)
    for x,value in [(left/2,left),(left+shared/2,shared),(left+shared+right/2,right)]:
        ax.text(x,1.03,str(value),ha='center',va='bottom',fontweight='bold',fontsize=8.3)
    if not compact:
        for x,label in [(left/2,'Regulatory only'),(left+shared/2,'Shared'),(left+shared+right/2,'Energy only')]:
            ax.text(x,-.065,label,ha='center',va='top',fontsize=6.6)

def density_scatter(ax,x,y,cmap,size=3.4,alpha=.76):
    x=np.asarray(x);y=np.asarray(y);sample=np.vstack([x,y])
    density=gaussian_kde(sample,bw_method='scott')(sample)
    percentile=(rankdata(np.round(np.log(density),10),method='average')-1)/(len(x)-1)*100
    order=np.argsort(density,kind='stable')
    artist=ax.scatter(x[order],y[order],c=percentile[order],cmap=cmap,norm=Normalize(0,100),
        s=size,alpha=alpha,lw=0,rasterized=True)
    return artist,density,percentile
