"""Hierarchical allocation atlas from saved maps: overview, local detail, statistics."""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm,TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
from shapely.geometry import box
from refined_primitives import signed_violin,overlap_ribbons
from matplotlib.colors import ListedColormap,BoundaryNorm
import redraw_figures as v
import enhanced_analysis as e
import cartography_style as c


def draw():
    W,H=183,241
    f=plt.figure(figsize=(W/25.4,H/25.4));f._font_scale=1.0
    f._layout='14-panel hierarchical atlas: dominant overview, shared-window detail triptych, six statistical diagnostics'
    def axes(x,y,w,h):return f.add_axes([x/W,y/H,w/W,h/H])
    def label(a,l,text):
        a.text(-.025,1.07,l,transform=a.transAxes,fontsize=10,fontweight='bold',va='bottom')
        f._panel_descriptions=getattr(f,'_panel_descriptions',{});f._panel_descriptions[l]=text
    def bar(x,y,w,norm,cm,text,ticks):
        cb=f.colorbar(v.mpl.cm.ScalarMappable(norm=norm,cmap=cm),cax=axes(x,y,w,1.8),orientation='horizontal',ticks=ticks)
        cb.outline.set_visible(False);cb.ax.tick_params(labelsize=8.2,pad=1,length=2)
        cb.set_label(text,fontsize=8.2,labelpad=2)
    prior=e.prior*100;full=e.full*100;delta=e.delta;sd=e.Q.std(0,ddof=1)*100
    hi=np.ceil(max(prior.max(),full.max())*10)/10
    dh=np.ceil(abs(delta).max()*100)/100;sh=np.ceil(sd.max()*1000)/1000
    share=PowerNorm(.5,0,hi);diff=TwoSlopeNorm(0,-dh,dh);std=PowerNorm(.5,0,sh)
    xmin,ymin,xmax,ymax=v.boundary.total_bounds
    window=(xmin+500,xmin+12500,(ymin+ymax)/2-2700,(ymin+ymax)/2+2700)
    a=axes(8,181,113,51);v.mapplot(a,full,share,v.SEQ);label(a,'a','Mean PGCAN allocation and common western detail window')
    a.add_patch(Rectangle((window[0],window[2]),window[1]-window[0],window[3]-window[2],
        fc='none',ec='black',lw=.8,ls=(0,(3,2)),zorder=12))
    a.text(window[0]+200,window[3]-200,'f–h',fontsize=8.5,fontweight='bold',va='top',zorder=14)
    bar(28,175,81,share,v.SEQ,'Share (%; square-root stretch)',[0,hi/4,hi])
    b=axes(138,186,34,37);positive=(prior>0)&(full>0)
    lx=np.log10(prior[positive]);ly=np.log10(full[positive])
    hb=b.hexbin(lx,ly,gridsize=27,mincnt=1,bins='log',cmap=v.DENSITY,linewidths=0,rasterized=True)
    lo=min(lx.min(),ly.min());up=max(lx.max(),ly.max())
    b.plot([lo,up],[lo,up],c='black',lw=.65,ls='--')
    b.set(xlabel='Prior (log₁₀ %)',ylabel='PGCAN (log₁₀ %)',xticks=[-4,-2,0],yticks=[-4,-2,0])
    b.text(.04,.96,f'ρ = {spearmanr(prior,full).statistic:.3f}',transform=b.transAxes,va='top',fontsize=8.2)
    label(b,'b','Cellwise agreement; 1,959 positive-prior cells; correlation across all cells')
    ht=b.inset_axes([0,1.02,1,.12]);ht.hist(lx,bins=26,color=v.TEAL,edgecolor='none');ht.axis('off')
    hr=b.inset_axes([1.02,0,.12,1]);hr.hist(ly,bins=26,orientation='horizontal',color=v.LILAC,edgecolor='none');hr.axis('off')
    cb=f.colorbar(hb,cax=axes(139,175,32,1.8),orientation='horizontal',ticks=[1,10,100]);cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.2,length=2,pad=1);cb.set_label('Cells / hexagon',fontsize=8.2,labelpad=2)
    specs=[(prior,share,v.SEQ,'c','Clean prior'),(delta,diff,v.DIFF,'d','Mean map minus prior'),(sd,std,v.SEQ,'e','Across-seed standard deviation')]
    for j,(z,norm,cm,letter,desc) in enumerate(specs):
        x=8+58*j;a=axes(x,135,49,28);v.mapplot(a,z,norm,cm);label(a,letter,desc)
        bar(x+9,129,39,norm,cm,['Share (%)','Difference (pp)','Seed SD (pp)'][j],
            [[0,hi/4,hi],[-dh,0,dh],[0,sh/4,sh]][j])
    # Three detail panels use exactly the marked common geographic window.
    # Only boundary lines are protected here: the window intentionally lies in
    # the thematic region, unlike the overview's contextual needle placement.
    for j,(z,norm,cm,letter,desc) in enumerate([(full,share,v.SEQ,'f','Local mean allocation'),
            (delta,diff,v.DIFF,'g','Local redistribution'),(sd,std,v.SEQ,'h','Local seed variation')]):
        a=axes(14+58*j,91,43,26);a._has_cartographic_basemap=True
        original_norm=norm
        inside=v.polys.geometry.intersection(box(window[0],window[2],window[1],window[3])).area>0
        norm,cm,ticks,description=v.mc.classify(np.asarray(z)[inside],cm)
        c.draw_basemap(a,v.BASEMAP,window,detail=True)
        v.polys.assign(value=z).plot(column='value',ax=a,norm=norm,cmap=cm,
            edgecolor='none',linewidth=0,alpha=.79,zorder=2)
        v.boundary.boundary.plot(ax=a,color='#536363',lw=.45,zorder=3)
        a.set(xlim=window[:2],ylim=window[2:],aspect='equal')
        c.compact_graticule(a);a._north_avoid_geometry=v.boundary.geometry.union_all().boundary.buffer(80)
        c.scale_bar(a,length=2000,anchor=(.045,.14));c.north_arrow(a,height=8)
        a._requested_map_norm=original_norm
        v.mc.add_bar(f,a,norm,cm,ticks,['Share (%)','Difference (pp)','Seed SD (pp)'][j],description)
        label(a,letter,desc+' in the same marked western window; scale as overview counterpart')
    for j in range(3):
        # Deliberate main-to-detail associations without repeated headings.
        x=(35+58*j)/W
        # Panel meanings are given in the external legend, leaving room for individual colourbars.
    i=axes(18,47,35,29)
    for z,col,name in [(prior,v.LILAC,'Prior'),(full,v.INK,'PGCAN')]:
        i.step(np.sort(z),np.arange(1,len(z)+1)/len(z)*100,c=col,lw=1,label=name)
    i.set_xscale('symlog',linthresh=.005);i.set(xlabel='Share (%)',ylabel='Cells (%)',ylim=(0,102),xticks=[0,.01,.1,1])
    i.set_xticklabels(['0','.01','.1','1']);i.legend(fontsize=8.2,loc='lower right',handlelength=1)
    label(i,'i','Full-cell allocation ECDFs; all zeros retained')
    j=axes(77,47,35,29)
    for z,col in [(prior,v.LILAC),(full,v.INK)]:
        j.plot(np.linspace(0,100,len(z)+1),np.r_[0,np.cumsum(np.sort(z))]/z.sum()*100,c=col,lw=1)
    lx=np.linspace(0,100,len(full)+1);lp=np.r_[0,np.cumsum(np.sort(prior))]/prior.sum()*100;lf=np.r_[0,np.cumsum(np.sort(full))]/full.sum()*100
    j.fill_between(lx,lp,lf,where=lf>=lp,color=v.TEAL,alpha=.21,lw=0)
    j.fill_between(lx,lp,lf,where=lf<lp,color=v.LILAC,alpha=.21,lw=0)
    j.plot([0,100],[0,100],c='black',ls=':',lw=.6)
    j.set(xlabel='Cells, ascending (%)',ylabel='Cumulative share (%)',xlim=(0,100),ylim=(0,100),xticks=[0,50,100])
    label(j,'j','Lorenz concentration curves on the same allocation support')
    k=axes(137,47,35,29);cats=e.CATORDER
    values=[delta[e.D.function_class.eq(key)] for key in cats]
    for pos,z in enumerate(values):signed_violin(k,z,pos,v.DIFF,width=.66)
    k.axvline(0,c='black',lw=.5,ls=':');k.set_yticks(range(5),['Mixed','Industrial','Low','Research','Transport'],fontsize=8.2)
    k.invert_yaxis();k.set_xlabel('Change (pp)');label(k,'k','Redistribution by functional category; median and IQR')
    l=axes(18,7,35,29);old=np.minimum((e.rp-1)//198,9);new=np.minimum((e.rf-1)//198,9)
    matrix=np.zeros((10,10),int);np.add.at(matrix,(old,new),1)
    migration_levels=[0,.5,5,15,30,50,75,100.001]
    migration_cmap=ListedColormap(['#F3F4F0',*v.HEAT(np.linspace(0,1,6))])
    im=l.imshow(matrix/198*100,cmap=migration_cmap,norm=BoundaryNorm(migration_levels,7),origin='lower',aspect='auto')
    im._preserve_semantic_cmap=True
    l.set(xticks=[0,4,9],xticklabels=[1,5,10],yticks=[0,4,9],yticklabels=[1,5,10],xlabel='PGCAN rank decile',ylabel='Prior rank decile')
    cb=f.colorbar(im,cax=l.inset_axes([1.03,0,.055,1]),ticks=[0,50,100]);cb.outline.set_visible(False);cb.ax.tick_params(labelsize=8.2,length=1,pad=1)
    label(l,'l','Rank-migration matrix; percentages within each prior decile')
    m=axes(77,7,35,29);groups=[e.pm&e.fm,e.pm&~e.fm,~e.pm&e.fm]
    counts=[int(z.sum()) for z in groups]
    overlap_ribbons(m,counts[1],counts[0],counts[2],labels=('Prior','PGCAN'),compact=True)
    label(m,'m','Overlap of prior and mean-map top-198 cells; method-only sets exclude shared cells')
    m.texts[-1].set_position((-.18,1.0));m.texts[-1].set_va('top')
    n=axes(137,7,35,29);freq=np.bincount(e.freq,minlength=6)
    for xi,(count,colour) in enumerate(zip(freq,[v.SEQ(t) for t in np.linspace(.08,.92,6)])):
        n.vlines(xi,1,max(count,1),color=colour,lw=1.6,alpha=.75)
        n.scatter(xi,max(count,1),s=23,c=[colour],ec='white',lw=.5,zorder=3)
        n.text(xi,max(count,1)*1.30,str(count),ha='center',fontsize=7)
    n.set(xticks=range(6),xlabel='Seeds selecting a cell',ylabel='Cells',yscale='log',ylim=(.8,10000),xlim=(-.55,5.5))
    label(n,'n','Cell selection frequency in top-198 across five training seeds; logarithmic count axis')
    for ax in [i,j,k,m,n]:v.quiet(ax)
    pd.DataFrame({'prior_decile':np.repeat(np.arange(1,11),10),'mean_decile':np.tile(np.arange(1,11),10),
        'cell_count':matrix.ravel(),'within_prior_pct':matrix.ravel()/198*100}).to_csv(v.SD/'Analysis_atlas_rank_migration.csv',index=False)
    (v.QA/'allocation_atlas_contract.json').write_text(json.dumps({'backend':'Python',
        'conclusion':'Allocation remains strongly prior-aligned, with localized redistribution and seed-dependent top-cell selection.',
        'archetype':'asymmetric mixed-modality figure','panels':14,'canvas_mm':[W,H],
        'hero_map_width_mm':113,'subordinate_map_width_mm':49,'detail_window_utm51n':list(window),
        'evidence':['Overview plus joint distribution','Same-support prior, change and seed SD maps',
            'Three maps of one declared western window','ECDF, concentration, category distributions, rank migration, overlap and stability'],
        'data_policy':'Saved allocation and five seed maps only; no new measured observations or training',
        'reference_adaptation':'Hierarchical overview/detail/statistics and controlled comparison scales; no reference values copied'},indent=2),encoding='utf-8')
    v.save(f,'Figure_3',['Figure_3_values.csv','allocation_500m.csv','probabilities_Full.npy','Analysis_atlas_rank_migration.csv'],
        'Hierarchical allocation atlas with 14 panels. The mean allocation overview identifies the common western detail window. '
        'Prior and mean allocation share one square-root colour stretch; seed SD also uses a square-root colour stretch, and differences use a symmetric linear scale. '
        'Panels f–h use the exact same geographic window and the scales of their corresponding overview maps. '
        'The joint density uses 1,959 positive-prior cells; Spearman correlation uses all 1,980 cells. '
        'ECDFs and concentration curves retain every cell, including zero shares. Category violins show empirical distributions, with black median dots and interquartile intervals. '
        'Rank migration uses deterministic grid-ID tie breaking and within-prior-decile percentages; the neutral first class represents zero (positive cells are at least 0.505%). The unit-square membership diagram represents the union of the two Top-198 selections: one square per cell, with 184 split green/pink squares for shared selections and 14 green-only and 14 pink-only squares. Square positions group membership categories and do not encode geography. '
        'Seed frequency counts all 1,980 cells across five original training seeds and uses a logarithmic count axis. '
        'Signed violin gradients encode negative/pink and positive/green changes, with median/IQR marks. The map legend gives seed standard deviation with ddof=1; no confidence intervals or measured emissions accuracies are implied.')

if __name__=='__main__':draw()
