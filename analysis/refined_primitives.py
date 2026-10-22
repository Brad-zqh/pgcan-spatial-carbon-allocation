"""Scientific marks whose positions, shapes and colours encode real quantities."""
import os
import numpy as np
from matplotlib.colors import to_rgba, Normalize
from matplotlib.path import Path
from scipy.stats import gaussian_kde,rankdata
from scipy.optimize import brentq
from matplotlib.patches import Circle,Rectangle
import matplotlib.patheffects as pe

def emphasis_line(artist,color,width=3.0):
    artist.set_path_effects([pe.Stroke(linewidth=width,foreground=color,alpha=.13),pe.Normal()])

def value_bar(ax,value,y,color,height=.46,sd=None):
    base=np.array(to_rgba(color));left=base.copy();right=base.copy()
    left[:3]=.62+.38*base[:3];right[:3]=.28+.72*base[:3]
    rgba=np.tile(np.linspace(left,right,128)[None,:,:],(2,1,1))
    patch=Rectangle((0,y-height/2),value,height,facecolor='none',edgecolor=color,lw=.4,zorder=2)
    ax.add_patch(patch)
    _image(ax,rgba,(0,value,y-height/2,y+height/2),zorder=1,clip=patch.get_path().transformed(patch.get_patch_transform()))
    if sd is not None and sd>0:ax.errorbar(value,y,xerr=sd,fmt='none',ecolor='#3E4741',lw=.8,capsize=2,zorder=3)
    return patch

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
    n1,n2=left+shared,right+shared;r1,r2=np.sqrt([n1,n2])
    def intersection(d):
        t1=np.arccos(np.clip((d*d+r1*r1-r2*r2)/(2*d*r1),-1,1))
        t2=np.arccos(np.clip((d*d+r2*r2-r1*r1)/(2*d*r2),-1,1))
        return r1*r1*t1+r2*r2*t2-.5*np.sqrt(max(0,(-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2)))
    d=brentq(lambda x:intersection(x)-shared*np.pi,abs(r1-r2)+1e-8,r1+r2-1e-8)
    np.testing.assert_allclose(intersection(d)/np.pi,shared,atol=1e-8)
    margin=.32*r1
    ax.set(xlim=(-r1-margin,d+r2+margin),ylim=(-1.43*r1,1.20*r1),aspect='equal');ax.axis('off')
    for x,r,c in [(0,r1,'#6A9251'),(d,r2,'#BE4D90')]:
        ax.add_patch(Circle((x,0),r,fc=to_rgba(c,.21),ec=c,lw=.9))
    ax.text((d-r2+r1)/2,0,str(shared),ha='center',va='center',fontsize=11,fontweight='bold')
    for side,count,mid,rim in [('left',left,(-r1+d-r2)/2,-r1),('right',right,(r1+d+r2)/2,d+r2)]:
        narrow=count/(n1 if side=='left' else n2)<.22
        if narrow:
            out=rim+(-.23 if side=='left' else .23)*r1
            ax.annotate(str(count),xy=(mid,0),xytext=(out,.48*r1),ha='center',va='center',fontsize=8.3,
                arrowprops={'arrowstyle':'-','lw':.6,'color':'#657067','connectionstyle':'angle,angleA=90,angleB=0,rad=0'})
        else:ax.text(mid,0,str(count),ha='center',va='center',fontsize=9,fontweight='bold')
    for x,name,n,c in [(-.45*r1,labels[0],n1,'#456B36'),(d+.45*r2,labels[1],n2,'#A52C75')]:
        text=ax.text(x,-1.13*r1,f'{name}\n(n = {n})',ha='center',va='top',fontsize=6.5,color=c)
        text.set_gid('compact_diagnostic_label')
    ax._set_overlap_geometry={'left_only':left,'shared':shared,'right_only':right,
        'first_total':n1,'second_total':n2,'area_per_count':float(np.pi),'intersection_area':float(intersection(d)),
        'radii':[float(r1),float(r2)],'center_distance':float(d),'area_proportional':True}

def density_scatter(ax,x,y,cmap,size=3.4,alpha=.76):
    x=np.asarray(x);y=np.asarray(y);sample=np.vstack([x,y])
    density=gaussian_kde(sample,bw_method='scott')(sample)
    percentile=(rankdata(np.round(np.log(density),10),method='average')-1)/(len(x)-1)*100
    order=np.argsort(density,kind='stable')
    artist=ax.scatter(x[order],y[order],c=percentile[order],cmap=cmap,norm=Normalize(0,100),
        s=size,alpha=alpha,lw=0,rasterized=True)
    return artist,density,percentile
