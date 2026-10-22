"""Substantive visual and descriptive analyses of the saved allocation experiment.

Adapted design ideas: map-plus-distribution composites, joint plots with marginal
histograms, density overlays, and annotated comparison matrices. No example data,
example significance labels or fitted relationships are copied from references.
"""
import os
from pathlib import Path
import json, shutil
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import Normalize, TwoSlopeNorm, PowerNorm, ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
import redraw_figures as v

P=v.HERE;SD=v.SD;QA=v.QA
D=v.grid;X=v.predictors;Q=np.load(v.DATA/'probabilities_Full.npy')
FEATURES=['ntl_sum','enterprise_poi_density','industrial_poi_density','floor_area_proxy',
          'road_density_km_per_km2','lowcarbon_patent_count','landuse_class1_ratio',
          'landuse_class2_ratio','landuse_class3_ratio']
FLABEL=['Nightlights','Enterprise POI','Industrial POI','Floor area','Road density','Patents','Land cover 1','Land cover 2','Land cover 3']
SHORT=['NTL','E-POI','I-POI','Floor','Road','Patent','LC1','LC2','LC3']
METHODS=['Nightlights','Enterprise_POI','Clean_prior','Direct_mixture','Full','Matched_postnorm','Matched_softmax']
SEEDS=list(range(20260918,20260923))
CATORDER=['mixed','industrial','low_activity','research','transport_built']
energy=pd.read_csv(v.DATA/'energy_benchmark_sites.csv')
ens=pd.read_csv(v.DATA/'metrics_ensemble.csv').set_index('method')
em=pd.read_csv(v.DATA/'energy_benchmark_metrics.csv')
for fn in ['energy_benchmark_metrics.csv','energy_benchmark_sites.csv','metrics_ensemble.csv']:
    shutil.copyfile(v.DATA/fn,SD/fn)
assert len(energy)==69 and energy.id.is_unique
indexer=pd.Index(D.grid_id).get_indexer(energy.grid_id)
assert np.all(indexer>=0)
ec=np.bincount(indexer,minlength=len(D));assert ec.sum()==69
def topflag(x):
    ids=np.lexsort((D.grid_id.to_numpy(str),-np.asarray(x)))[:198]
    mask=np.zeros(len(D),bool);mask[ids]=True;return mask
select=np.array([topflag(q) for q in Q]);freq=select.sum(0)
prior=D.Clean_prior_share.to_numpy();full=D.Full_share.to_numpy();delta=(full-prior)*100
pm=topflag(prior);fm=topflag(full)
def ranks(x):
    order=np.lexsort((D.grid_id.to_numpy(str),np.asarray(x)))
    r=np.empty(len(x),int);r[order]=np.arange(1,len(x)+1);return r
def gini(x):
    x=np.sort(np.asarray(x));n=len(x)
    return 2*np.dot(np.arange(1,n+1),x)/(n*x.sum())-(n+1)/n
rp=ranks(prior);rf=ranks(full)

def tickstyle(a):
    a.tick_params(labelsize=7)
    v.quiet(a)

def density_joint(fig,rect,letter,title):
    a=fig.add_axes(rect);mask=(prior>0)&(full>0)
    x=np.log10(prior[mask]*100);y=np.log10(full[mask]*100)
    hb=a.hexbin(x,y,gridsize=33,mincnt=1,cmap=v.DENSITY,bins='log',linewidths=0)
    lo=min(x.min(),y.min());hi=max(x.max(),y.max())
    a.plot([lo,hi],[lo,hi],c=v.GOLD,ls='--',lw=.8)
    a.set(xlabel='Prior share (log₁₀ %)',ylabel='PGCAN share (log₁₀ %)')
    a.text(.04,.95,f'n = {mask.sum():,} positive-prior cells',transform=a.transAxes,fontsize=6.3,va='top')
    a.text(.04,.84,f'ρ = {spearmanr(prior,full).statistic:.3f} (all cells)',transform=a.transAxes,fontsize=6.3,va='top')
    ht=a.inset_axes([0,1.02,1,.15]);hr=a.inset_axes([1.02,0,.14,1])
    ht.hist(x,bins=32,color=v.TEAL,alpha=.7,edgecolor='white',lw=.15)
    hr.hist(y,bins=32,orientation='horizontal',color=v.INK,alpha=.7,edgecolor='white',lw=.15)
    ht.axis('off');hr.axis('off')
    cb=fig.colorbar(hb,cax=a.inset_axes([.15,-.36,.70,.035]),orientation='horizontal')
    cb.outline.set_visible(False);cb.ax.tick_params(labelsize=5.3,length=1.4,pad=1)
    cb.set_label('Cells per hexagon',fontsize=5.8,labelpad=1)
    a.set_title(title,loc='left',y=1.22,pad=4)
    a.text(-.075,1.24,letter,transform=a.transAxes,fontweight='bold',fontsize=10,va='bottom')
    return a

def enriched_main3():
    f=plt.figure(figsize=(v.WIDTH,119/25.4))
    hi=np.ceil(max(prior.max(),full.max())*1000)/10
    sd=Q.std(0,ddof=1)*100;dh=np.ceil(abs(delta).max()*100)/100;sh=np.ceil(sd.max()*1000)/1000
    parts=[([.04,.63,.29,.29],prior*100,'a','Clean prior',PowerNorm(.5,0,hi),v.SEQ,'Share (%; square-root stretch)',[0,hi/4,hi]),
           ([.365,.63,.29,.29],full*100,'b','PGCAN mean map',PowerNorm(.5,0,hi),v.SEQ,'Share (%; square-root stretch)',[0,hi/4,hi]),
           ([.04,.14,.29,.29],delta,'c','Mean map − prior',TwoSlopeNorm(0,-dh,dh),v.DIFF,'Difference (pp)',[-dh,0,dh]),
           ([.365,.14,.29,.29],sd,'d','Across-seed variation',PowerNorm(.5,0,sh),v.SEQ,'Seed SD (pp; square-root)',[0,sh/4,sh])]
    for rect,z,l,t,n,cm,unit,ticks in parts:
        a=f.add_axes(rect);v.mapplot(a,z,n,cm);v.panel(a,l,t);v.colorbar(f,a,cm,n,unit,ticks)
    density_joint(f,[.76,.65,.185,.245],'e','Cellwise agreement')
    a=f.add_axes([.76,.14,.185,.28])
    for values,c,lab in [(prior*100,v.LILAC,'Prior'),(full*100,v.INK,'PGCAN')]:
        xx=np.sort(values);a.step(xx,np.arange(1,len(xx)+1)/len(xx)*100,c=c,lw=1.1,label=lab)
    a.set_xscale('symlog',linthresh=.005);a.set(xlabel='Regional share (%)',ylabel='Cumulative cells (%)',ylim=(0,102))
    a.set_xticks([0,.01,.1,1],['0','0.01','0.1','1']);a.legend(fontsize=6.5,loc='lower right');tickstyle(a);v.panel(a,'f','Allocation distribution')
    v.foot(f,'Maps share projected support | Joint plot excludes 21 zero-prior cells; Spearman ρ uses all 1,980 cells | SD: five training seeds')
    v.save(f,'Figure_3',['Figure_3_values.csv','allocation_500m.csv','probabilities_Full.npy'],
           'Four comparable maps plus a density joint plot with marginal distributions and allocation ECDFs. No grid-level emissions observations are used.')

def enriched_main1():
    from study_area_map import draw_study_area
    return draw_study_area()

def enriched_proxies():
    f=plt.figure(figsize=(v.WIDTH,6.1))
    specs=[('ntl_sum','Nightlight sum'),('enterprise_poi_density','Enterprise POI density'),
           ('floor_area_proxy','OSM building floor-area proxy'),('road_density_km_per_km2','Road density')]
    cb=f.colorbar(v.mpl.cm.ScalarMappable(norm=Normalize(0,1),cmap=v.SEQ),cax=f.add_axes([.34,.95,.40,.012]),orientation='horizontal',ticks=[0,.5,1]);cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6,length=2);cb.set_label('Within-panel log(1 + value), rescaled to [0, 1]',fontsize=6.7,labelpad=1)
    for i,(k,title) in enumerate(specs):
        x=.045+(i%2)*.49;y=.65-(i//2)*.44
        z=np.log1p(X[k].to_numpy());z=z/z.max() if z.max() else z
        a=f.add_axes([x,y,.42,.23]);v.mapplot(a,z,Normalize(0,1));v.panel(a,chr(97+i),title)
        a.text(.02,.91,f'{(X[k]>0).mean()*100:.1f}% non-zero',transform=a.transAxes,fontsize=6.3,color='#536D78')
        h=f.add_axes([x+.065,y-.135,.31,.09]);counts,edges=np.histogram(z,bins=np.linspace(0,1,31))
        h.bar(edges[:-1],counts,width=np.diff(edges),align='edge',color=v.SEQ((edges[:-1]+edges[1:])/2),edgecolor='white',lw=.15)
        h.set_yscale('log');h.set(xlim=(0,1),ylim=(.8,4000),xticks=[0,.5,1],yticks=[1,100,1000])
        h.tick_params(labelsize=6,length=2,pad=1);h.set_ylabel('Cells (log)',fontsize=6,labelpad=1);h.set_xlabel('Displayed relative intensity',fontsize=6.3,labelpad=1)
    v.foot(f,'Spatial pattern plus full-cell distributions | Zero prevalence is retained | Colours compare relative intensity within a panel, not activity units across sources')
    v.save(f,'Figure_S1',['predictors_only.csv','Figure_S1_display_values.csv'],
           'Four proxy maps paired with their full-cell value distributions and non-zero prevalence. The shared colour scale is panel-specific log(1+x) rescaling; histogram counts use log axes.')

def enriched_main4():
    f=plt.figure(figsize=(v.WIDTH,6.2))
    a=f.add_axes([.085,.62,.34,.285]);b=f.add_axes([.70,.62,.20,.285])
    c=f.add_axes([.085,.16,.34,.285]);d=f.add_axes([.70,.16,.20,.285])
    keys=METHODS[:5]
    for k in keys:
        z=v.curves[v.curves.method==k].sort_values('fraction')
        kw=dict(color=v.COL[k],lw=1.5 if k=='Full' else .9,ls='-' if k in ['Full','Enterprise_POI','Clean_prior'] else '--')
        a.plot(z.fraction*100,z.primary_coverage*100,label=v.LABEL[k],**kw)
        c.plot(z.fraction*100,z.mass_share*100,**kw)
    for ax in [a,c]:ax.axvline(10,c='#A9B9C1',ls=':',lw=.75);ax.set(xlim=(0,50),ylim=(0,102),xlabel='Top-ranked cells (%)');tickstyle(ax)
    a.set_ylabel('Regulatory-site coverage (%)');c.set_ylabel('Captured regional mass (%)')
    v.panel(a,'a','Coverage across thresholds');v.panel(c,'c','Mass concentration across thresholds')
    from refined_primitives import value_bar,row_tone,pink_green_row
    for i,k in enumerate(METHODS):
        values=v.runs.loc[v.runs.method.eq(k),'primary_coverage'].to_numpy()*100
        mu=values.mean();sd=values.std(ddof=1) if len(values)>1 else 0
        value_bar(b,mu,i,pink_green_row(i,len(METHODS)),height=.52,sd=sd)
        txt=f'{mu:.3f}' if len(values)==1 else f'{mu:.3f} ± {sd:.3f}'
        b.text(1.02,i,txt,transform=b.get_yaxis_transform(),fontsize=6.8,va='center')
    b.set(yticks=range(7),xlim=(0,60),ylim=(6.6,-.6),xlabel='Site coverage (%)')
    b.tick_params(axis='y',length=0);b.spines['left'].set_visible(False);v.quiet(b,'x')
    v.panel(b,'b','Top-10% coverage',title_x=-.65)
    short=lambda k:v.LABEL[k].replace('Matched post-normalization','Post-normalized')
    b.set_yticklabels([short(k) for k in METHODS],fontsize=7)
    for i,k in enumerate(METHODS):
        m=topflag(D[k+'_share']);reg=D.primary_count.to_numpy()[m].sum()/146*100;en=ec[m].sum()/69*100
        value_bar(d,reg,i-.16,row_tone(v.INK,i,len(METHODS)),height=.28)
        value_bar(d,en,i+.16,row_tone(v.LILAC,i,len(METHODS)),height=.28)
    d.set_yticks(range(7),[short(k) for k in METHODS],fontsize=6.8);d.set(ylim=(6.6,-.6),xlim=(0,65),xlabel='Fixed-map site coverage (%)')
    d.tick_params(axis='y',length=0);d.spines['left'].set_visible(False);v.quiet(d,'x');v.panel(d,'d','Two location cohorts',title_x=-.65)
    d.legend(handles=[v.Patch(fc=v.INK,label='Regulatory (146)'),v.Patch(fc=v.LILAC,label='Energy-user (69)')],
             loc='upper center',bbox_to_anchor=(.5,-.22),ncol=1,fontsize=6.4,handlelength=1,columnspacing=.8)
    f.legend(*a.get_legend_handles_labels(),loc='lower left',bbox_to_anchor=(.07,.04),ncol=3,fontsize=6.5,handlelength=1.5,columnspacing=.8)
    v.foot(f,'Curves / cohort bars: fixed mean maps; panel b: per-run mean ± seed SD | The two cohorts share 59 sites and POI geolocation')
    v.save(f,'Figure_4',['coverage_curves.csv','metrics_runs.csv','energy_benchmark_sites.csv','allocation_500m.csv'],
           'Coverage, captured mass, per-seed variation and paired regulatory / energy-user cohort coverage. Shared sites and geolocation limit cohort independence. Bars in b show per-run mean coverage, with capped sample-SD intervals; numerical labels retain three decimals. Each of the seven bars in b has a uniform fill; the rows progress from deep pink through pale pink and pale green to deep green. Colours distinguish row order and do not encode coverage magnitude or another statistic. Paired green/pink bars in d show the two exact fixed-map cohort coverages. Each bounded bar runs from zero to its actual value. In d the green and pink series each follow their own light-to-dark row progression.')

def input_structure():
    corr=X[FEATURES].corr(method='spearman').to_numpy()
    dist=np.clip(1-np.abs(corr),0,1);np.fill_diagonal(dist,0)
    tree=linkage(squareform(dist,checks=False),method='average');order=leaves_list(tree)
    f=plt.figure(figsize=(v.WIDTH,6.8))
    a=f.add_axes([.09,.50,.36,.35]);at=f.add_axes([.09,.87,.36,.075])
    dendrogram(tree,ax=at,no_labels=True,color_threshold=0,above_threshold_color='#73949E');at.axis('off')
    mat=corr[np.ix_(order,order)];im=a.imshow(mat,cmap=v.DIFF,vmin=-1,vmax=1)
    for i in range(9):
        for j in range(9):
            if i>=j:a.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=5.5,color='white' if abs(mat[i,j])>.65 else v.INK)
    a.set_xticks(range(9),[SHORT[i] for i in order],rotation=45,ha='right',fontsize=6)
    a.set_yticks(range(9),[SHORT[i] for i in order],fontsize=6);a.tick_params(length=0)
    a.set_title('Clustered predictor correlations',loc='left',y=1.34,pad=4)
    a.text(-.075,1.35,'a',transform=a.transAxes,fontweight='bold',fontsize=10,va='bottom')
    cb=f.colorbar(im,cax=f.add_axes([.46,.54,.012,.25]));cb.outline.set_visible(False);cb.ax.tick_params(labelsize=6,length=2)
    b=f.add_axes([.68,.50,.275,.35])
    z=np.log1p(X[FEATURES].to_numpy(float));z=(z-z.mean(0))/z.std(0)
    vp=b.violinplot([z[:,i] for i in range(9)],positions=np.arange(9),orientation='horizontal',widths=.8,showextrema=False)
    for i,body in enumerate(vp['bodies']):
        body.set_facecolor([v.TEAL,v.LILAC,v.GOLD][i%3]);body.set_alpha(.65);body.set_edgecolor('white');body.set_linewidth(.4)
    q=np.percentile(z,[25,50,75],axis=0)
    b.hlines(np.arange(9),q[0],q[2],color=v.INK,lw=1.3);b.scatter(q[1],np.arange(9),s=10,c='white',edgecolor=v.INK,lw=.5,zorder=3)
    b.set_yticks(range(9),FLABEL,fontsize=6.5);b.invert_yaxis();b.set_xlim(-2.7,9.5);b.axvline(0,c='#B8C6CC',ls=':',lw=.6)
    for i,k in enumerate(FEATURES):b.text(1.02,i,f'{(X[k]==0).mean()*100:.0f}%',transform=b.get_yaxis_transform(),fontsize=6,va='center')
    b.text(1.02,1.025,'Zeros',transform=b.transAxes,fontsize=6,color='#697F88')
    b.set_xlabel('z score of log(1 + predictor)');b.spines['left'].set_visible(False);b.tick_params(axis='y',length=0)
    v.panel(b,'b','Distribution and zero prevalence')
    c=f.add_axes([.16,.145,.72,.235]);cz=pd.DataFrame(z,columns=FEATURES).assign(category=D.function_class).groupby('category').mean().loc[CATORDER].to_numpy()
    lim=max(abs(cz.min()),abs(cz.max()));im=c.imshow(cz,cmap=v.DIFF,norm=TwoSlopeNorm(0,-lim,lim),aspect='auto')
    c.set_xticks(range(9),SHORT,fontsize=6.8);c.set_yticks(range(5),[f'{v.CAT[k][1]} (n={int((D.function_class==k).sum())})' for k in CATORDER],fontsize=6.8)
    for i in range(5):
        for j in range(9):c.text(j,i,f'{cz[i,j]:+.1f}',ha='center',va='center',fontsize=6.3,color='white' if abs(cz[i,j])>lim*.55 else v.INK)
    c.tick_params(length=0);v.panel(c,'c','Operational-category predictor profiles',title_x=0)
    cb=f.colorbar(im,cax=f.add_axes([.90,.145,.012,.235]));cb.outline.set_visible(False);cb.ax.tick_params(labelsize=6,length=2);cb.set_label('Mean z score',fontsize=6.5)
    v.foot(f,'Descriptive Spearman associations; clustering uses 1 − |ρ| | Violin / category panels standardize log-transformed inputs, not emissions')
    pd.DataFrame(corr,index=FEATURES,columns=FEATURES).to_csv(SD/'Analysis_predictor_spearman.csv')
    pd.DataFrame(cz,index=CATORDER,columns=FEATURES).to_csv(SD/'Analysis_category_predictor_profiles.csv')
    pd.DataFrame({'predictor':FEATURES,'zero_fraction':[(X[k]==0).mean() for k in FEATURES]}).to_csv(SD/'Analysis_predictor_zero_prevalence.csv',index=False)
    v.save(f,'Figure_S4',['predictors_only.csv','Analysis_predictor_spearman.csv','Analysis_category_predictor_profiles.csv','Analysis_predictor_zero_prevalence.csv'],
           'Input structure: absolute-correlation hierarchical clustering, predictor distribution / zero prevalence and category-specific mean standardized log profiles. Correlations are descriptive, not causal effects.')

def allocation_anatomy():
    f=plt.figure(figsize=(v.WIDTH,6.5));a=f.add_axes([.085,.60,.34,.28]);b=f.add_axes([.67,.60,.285,.28])
    c=density_joint(f,[.085,.18,.30,.245],'c','Allocation density and marginals')
    d=f.add_axes([.655,.16,.285,.275])
    concentration=[]
    for k in ['Enterprise_POI','Clean_prior','Direct_mixture','Full']:
        vals=np.sort(D[k+'_share'].to_numpy());xx=np.arange(len(vals)+1)/len(vals)*100;yy=np.r_[0,np.cumsum(vals)]*100
        a.plot(xx,yy,c=v.COL[k],lw=1.4 if k=='Full' else .95,label=f'{v.LABEL[k]}  (G={gini(vals):.3f})')
        concentration.append({'method':k,'gini':gini(vals),'top10_mass_share':float(D.loc[topflag(D[k+'_share']),k+'_share'].sum())})
    a.plot([0,100],[0,100],c='#A9B9C1',ls=':',lw=.7);a.set(xlim=(0,100),ylim=(0,100),xlabel='Cells, ascending allocation (%)',ylabel='Cumulative regional share (%)')
    a.legend(fontsize=6.4,loc='upper left',handlelength=1.3);v.panel(a,'a','Allocation concentration');tickstyle(a)
    data=[delta[D.function_class.eq(k)] for k in CATORDER]
    vp=b.violinplot(data,positions=range(5),orientation='horizontal',widths=.7,showextrema=False)
    for k,body in zip(CATORDER,vp['bodies']):body.set_facecolor(v.CAT[k][0]);body.set_alpha(.8);body.set_edgecolor('white')
    q=[np.percentile(z,[25,50,75]) for z in data]
    for i,(lo,mid,hi) in enumerate(q):b.plot([lo,hi],[i,i],c=v.INK,lw=1.2);b.scatter(mid,i,c='white',edgecolor=v.INK,s=14,lw=.5,zorder=3)
    b.axvline(0,c='#8EA3AD',ls='--',lw=.7);b.set_yticks(range(5),[v.CAT[k][1] for k in CATORDER],fontsize=6.8);b.invert_yaxis();b.set_xlabel('Cell share change (pp)');b.spines['left'].set_visible(False);b.tick_params(axis='y',length=0)
    v.panel(b,'b','Within-category redistribution')
    old=np.minimum((rp-1)//198,9);new=np.minimum((rf-1)//198,9);transition=np.zeros((10,10),int)
    np.add.at(transition,(old,new),1);assert np.all(transition.sum(0)==198) and np.all(transition.sum(1)==198)
    im=d.imshow(transition/198*100,cmap=v.SEQ,norm=PowerNorm(.4,0,100),origin='lower')
    for i in range(10):
        for j in range(10):
            val=transition[i,j]/198*100
            if val>=2:d.text(j,i,f'{val:.0f}',ha='center',va='center',fontsize=5.8,color='white' if val>20 else v.INK)
    d.set_xticks(range(10),np.arange(1,11),fontsize=6);d.set_yticks(range(10),np.arange(1,11),fontsize=6)
    d.set(xlabel='PGCAN rank decile (low → high)',ylabel='Prior rank decile (low → high)');d.tick_params(length=0)
    v.panel(d,'d','Rank migration (% within prior decile)')
    cb=f.colorbar(im,cax=d.inset_axes([1.04,0,.045,1]),ticks=[0,10,50,100]);cb.outline.set_visible(False);cb.ax.tick_params(labelsize=6,length=2)
    v.foot(f,'Lorenz / Gini describe allocation concentration, not inequality of measured emissions | Deterministic grid-ID tie breaking for ranks')
    pd.DataFrame(concentration).to_csv(SD/'Analysis_concentration.csv',index=False)
    pd.DataFrame(transition,index=np.arange(1,11),columns=np.arange(1,11)).to_csv(SD/'Analysis_rank_decile_transitions.csv')
    pd.DataFrame({'grid_id':D.grid_id,'prior_rank':rp,'pgcan_rank':rf,'rank_change':rf-rp,'share_change_pp':delta,'category':D.function_class}).to_csv(SD/'Analysis_cell_rank_changes.csv',index=False)
    v.save(f,'Figure_S5',['allocation_500m.csv','Analysis_concentration.csv','Analysis_rank_decile_transitions.csv','Analysis_cell_rank_changes.csv'],
           'Allocation anatomy: Lorenz curves / Gini, category-wise share-change distributions, cellwise density with marginal histograms and row-normalized rank-decile migration. No emissions accuracy claim is implied.')

def seed_stability():
    f=plt.figure(figsize=(v.WIDTH,6.15));a=f.add_axes([.045,.58,.47,.31]);b=f.add_axes([.665,.60,.265,.29])
    cols=[v.mpl.colors.to_hex(v.SEQ(t)) for t in np.linspace(0,1,6)];cm=ListedColormap(cols);norm=BoundaryNorm(np.arange(-.5,6.5),6)
    v.mapplot(a,freq,norm,cm);v.panel(a,'a','Hotspot selection frequency across five seeds')
    v.colorbar(f,a,cm,norm,'Times selected in the Top-198 cells',[0,1,2,3,4,5])
    jac=np.array([[(s&t).sum()/(s|t).sum() for t in select] for s in select])
    jac_min=np.floor(jac.min()*1000)/1000
    im=b.imshow(jac,cmap=v.SEQ,vmin=jac_min,vmax=1)
    for i in range(5):
        for j in range(5):b.text(j,i,f'{jac[i,j]:.3f}',ha='center',va='center',fontsize=6.7,color='white' if jac[i,j]>.95 else v.INK)
    b.set_xticks(range(5),[f'S{i+1}' for i in range(5)]);b.set_yticks(range(5),[f'S{i+1}' for i in range(5)]);b.tick_params(length=0);v.panel(b,'b','Seed-to-seed Top-198 Jaccard')
    cb=f.colorbar(im,cax=b.inset_axes([1.04,0,.045,1]),ticks=[jac_min,(1+jac_min)/2,1]);cb.outline.set_visible(False);cb.ax.tick_params(labelsize=6,length=2)
    cb.ax.set_yticklabels([f'{x:.3f}' for x in [jac_min,(1+jac_min)/2,1]])
    cb.set_label('Jaccard',fontsize=6.5,labelpad=2)
    c=f.add_axes([.09,.16,.345,.26]);d=f.add_axes([.635,.16,.31,.26])
    n=np.bincount(freq,minlength=6);c.bar(range(6),n,color=cols,edgecolor='#73929B',lw=.45,width=.64);c.set_yscale('log');c.set(xticks=range(6),xlabel='Number of seeds selecting a cell',ylabel='Number of cells (log scale)',ylim=(.8,4000))
    for i,count in enumerate(n):c.text(i,count*1.17,f'{count:,}',ha='center',fontsize=7)
    c.text(.96,.94,f'{int((freq==5).sum())} shared core cells\n{int((freq>0).sum())} cells in the union',transform=c.transAxes,fontsize=7,va='top',ha='right')
    v.panel(c,'c','Core and boundary selections');v.quiet(c)
    reg=np.array([D.primary_count.to_numpy()[s].sum()/146*100 for s in select]);eng=np.array([ec[s].sum()/69*100 for s in select]);xx=np.arange(5)
    d.plot(xx,reg,'o-',c=v.INK,lw=1.1,ms=4,mec='white',mew=.4,label='Regulatory (146)')
    d.plot(xx,eng,'s-',c=v.LILAC,lw=1.1,ms=4,mec='white',mew=.4,label='Energy-user (69)')
    d.set(xticks=xx,xticklabels=[f'S{i+1}' for i in range(5)],xlabel='Training seed',ylabel='Top-10% site coverage (%)');tickstyle(d)
    d.legend(loc='upper center',bbox_to_anchor=(.5,-.24),ncol=2,fontsize=6.5,handlelength=1.2,columnspacing=.8)
    v.panel(d,'d','Screening variability across seeds')
    v.foot(f,'S1–S5: seeds 20260918–20260922 | Frequency measures algorithmic repeatability; it is not an emissions confidence probability')
    pd.DataFrame({'grid_id':D.grid_id,'selection_frequency':freq,**{f'S{i+1}':select[i].astype(int) for i in range(5)}}).to_csv(SD/'Analysis_seed_selection_frequency.csv',index=False)
    pd.DataFrame(jac,index=SEEDS,columns=SEEDS).to_csv(SD/'Analysis_seed_jaccard.csv')
    pd.DataFrame({'seed':SEEDS,'regulatory_coverage_pct':reg,'energy_coverage_pct':eng}).to_csv(SD/'Analysis_seed_cohort_coverage.csv',index=False)
    v.save(f,'Figure_S6',['probabilities_Full.npy','Analysis_seed_selection_frequency.csv','Analysis_seed_jaccard.csv','Analysis_seed_cohort_coverage.csv'],
           'Seed stability: spatial selection frequency, pairwise Top-198 Jaccard, frequency counts and cohort coverage across the five recorded seeds. The Jaccard colour scale spans the observed lower bound (rounded down to three decimals) to 1; printed cell labels retain the raw overlaps. Site lists share geolocation; stability does not quantify emissions uncertainty.')

def benchmark_profiles():
    keys=METHODS;metrics=['primary_coverage','primary_rho','primary_ap','primary_auc','top10_mass_share']
    vals=ens.loc[keys,metrics].to_numpy();scaled=(vals-vals.min(0))/np.maximum(np.ptp(vals,axis=0),1e-12)
    f=plt.figure(figsize=(v.WIDTH,6.35));a=f.add_axes([.26,.59,.655,.30])
    im=a.imshow(scaled,cmap=v.SEQ,vmin=0,vmax=1,aspect='auto');a._heatmap_units='Relative score'
    for i in range(7):
        for j in range(5):
            val=vals[i,j]*100 if j in [0,4] else vals[i,j]
            a.text(j,i,f'{val:.2f}' if j in [0,4] else f'{val:.3f}',ha='center',va='center',fontsize=7,color='white' if scaled[i,j]>.55 else v.INK)
    a.set_xticks(range(5),['Coverage (%)','Spearman ρ','AP','ROC AUC','Top-10% mass (%)'],fontsize=6.7)
    a.set_yticks(range(7),[v.LABEL[k] for k in keys],fontsize=7);a.tick_params(length=0);v.panel(a,'a','Fixed-map method profiles (raw metric labels)',title_x=-.27)
    b=f.add_axes([.085,.16,.34,.27]);c=f.add_axes([.64,.16,.30,.27]);cohort=[]
    for k in ['Clean_prior','Direct_mixture','Full']:
        order=np.lexsort((D.grid_id.to_numpy(str),-D[k+'_share'].to_numpy()))
        qs=np.arange(.01,.501,.01);coverage=[]
        for q in qs:
            ids=order[:max(1,int(q*len(D)))];coverage.append(ec[ids].sum()/69*100)
        b.plot(qs*100,coverage,c=v.COL[k],lw=1.4 if k=='Full' else 1,label=v.LABEL[k])
        m=topflag(D[k+'_share']);expected=em[(em.method==k)&(em.subset=='all_energy')].iloc[0].coverage
        assert np.isclose(ec[m].sum()/69,expected)
        cohort.extend({'method':k,'fraction':q,'energy_coverage_pct':cv} for q,cv in zip(qs,coverage))
    b.axvline(10,c='#AABBC2',ls=':',lw=.7);b.set(xlim=(0,50),ylim=(0,102),xlabel='Top-ranked cells (%)',ylabel='Energy-user coverage (%)');tickstyle(b)
    b.legend(fontsize=6.7,loc='lower right');v.panel(b,'b','Energy-user screening curves (n = 69)')
    for k in keys:
        r=ens.loc[k];c.scatter(r.top10_mass_share*100,r.primary_coverage*100,s=35,c=v.COL[k],edgecolor='white',lw=.5,zorder=3)
    positions={'Nightlights':(42,31.7),'Enterprise_POI':(59,39.5),'Clean_prior':(46,41.5),'Direct_mixture':(36.5,48),'Full':(49,46.6)}
    for k,pos in positions.items():
        r=ens.loc[k];c.annotate(v.LABEL[k],(r.top10_mass_share*100,r.primary_coverage*100),xytext=pos,textcoords='data',fontsize=6.5,
                                arrowprops={'arrowstyle':'-','color':'#97ABB4','lw':.5})
    c.set(xlabel='Top-10% regional mass (%)',ylabel='Regulatory-site coverage (%)',xlim=(35,74),ylim=(30,49));tickstyle(c);v.panel(c,'c','Concentration–screening trade-off')
    v.foot(f,'Heatmap colours are normalized within each metric; labels retain raw values | AP/AUC describe location screening, not emissions accuracy')
    ens.loc[keys,metrics].to_csv(SD/'Analysis_fixed_map_profiles.csv')
    pd.DataFrame(cohort).to_csv(SD/'Analysis_energy_coverage_curves.csv',index=False)
    v.save(f,'Figure_S7',['metrics_ensemble.csv','Analysis_fixed_map_profiles.csv','Analysis_energy_coverage_curves.csv','energy_benchmark_metrics.csv'],
           'Method profiles with raw metric labels and within-column colour normalization, energy-user coverage curves, and concentration versus regulatory-site screening. The energy benchmark shares 59 sites with the regulatory list.')

def write_analysis():
    core=int((freq==5).sum());union=int((freq>0).sum());shared=int((pm&fm).sum());tv=float(abs(full-prior).sum()/2)
    corr=X[FEATURES].corrwith(pd.Series(full),method='spearman').sort_values(ascending=False)
    summary={'n_cells':len(D),'n_training_seeds':5,'prior_zero_cells':int((prior==0).sum()),
             'all_seed_core_cells':core,'any_seed_selected_cells':union,'variable_selection_cells':union-core,
             'core_fraction_of_union':core/union,'core_fraction_of_each_top198':core/198,
             'prior_neural_shared_top198':shared,'prior_only_top198':int((pm&~fm).sum()),'neural_only_top198':int((fm&~pm).sum()),
             'tv_neural_vs_prior':tv,'gini_prior':gini(prior),'gini_neural':gini(full),'gini_direct_mixture':gini(D.Direct_mixture_share),
             'descriptive_predictor_allocation_spearman':corr.to_dict(),
             'limitations':['Predictors and constructed targets share information. Correlations are descriptive.','Five seeds quantify algorithmic variability, not emissions uncertainty.','Location benchmarks share POI coordinates and 59 sites.']}
    (P/'Analysis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    report=f'''# Additional descriptive analyses\n\nThe revised figures add data analysis and change the visual structure; no model is retrained and no observed emissions are invented.\n\n- **Seed stability.** The five Top-198 selections share {core} cells, while their union contains {union} cells. Thus {union-core} cells change selection across runs; the shared core is {core/union*100:.2f}% of the union and {core/198*100:.2f}% of each run's selected cells. This describes repeatability of the algorithm, not accuracy of cell emissions.\n- **Spatial disagreement.** The prior and PGCAN mean map share {shared} of their 198 selected cells. Each selects {198-shared} cells absent from the other selection. Coverage summaries alone therefore omit spatial changes.\n- **Allocation concentration.** The Gini coefficient changes from {gini(prior):.4f} for the prior to {gini(full):.4f} for PGCAN; the direct target has Gini {gini(D.Direct_mixture_share):.4f}. The neural allocation is less concentrated than the prior on this saved grid, while remaining close to the explicitly constructed target. These are allocation concentration statistics, not measured emissions inequality.\n- **Redistribution.** Total variation between the mean neural allocation and the prior is {tv:.6f}; equivalently {tv*100:.2f}% of regional mass is reallocated in total-variation terms. Category distributions and rank-decile transitions show where these changes occur.\n- **Input structure.** The most strongly associated predictor in a descriptive Spearman comparison with neural shares is {corr.index[0]} (ρ={corr.iloc[0]:.3f}). Predictors also help construct the target, so this is not an independently identified effect. The figures report zero prevalence, correlations and category profiles directly from all 1,980 cells.\n- **Screening versus concentration.** The new method-profile panel reports coverage, rank association, AP, ROC AUC and selected mass separately. Energy-user coverage curves are recomputed from 69 geolocated sites and checked against the saved Top-10% benchmark values. The 59 shared sites and shared POI coordinates prevent treating the two cohorts as independent emissions validation.\n\n## Figure guide\n\n- Figure 3: four maps + cellwise density / marginal distributions + ECDF.\n- Figure 4: regulatory coverage + seed-level comparison + captured mass + two-cohort comparison.\n- Figure S4: clustered correlations + violin / quartile distributions + category profiles.\n- Figure S5: Lorenz / Gini + within-category redistribution + density joint plot + rank-decile transitions.\n- Figure S6: spatial selection frequency + seed Jaccard matrix + shared-core counts + cohort variability.\n- Figure S7: annotated method profiles + energy coverage curves + screening–concentration trade-off.\n\nAll new numerical statements are reproduced by enhanced_analysis.py and their source tables are saved under source_data/. No p-values or causal claims are introduced.\n'''
    (P/'Additional_Analysis.md').write_text(report,encoding='utf-8')
    return summary

if __name__=='__main__':
    old=json.loads((P/'figure_manifest.json').read_text(encoding='utf-8'))
    for fn in [enriched_main1,enriched_main3,enriched_main4,enriched_proxies,input_structure,allocation_anatomy,seed_stability,benchmark_profiles]:fn()
    additions={x['figure']:x for x in v.manifest}
    final=[additions.pop(x['figure'],x) for x in old]
    final.extend(additions.values());v.manifest=final
    (P/'figure_manifest.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
    summary=write_analysis()
    print('ENRICHED_ANALYSIS_COMPLETE',json.dumps({k:summary[k] for k in ['all_seed_core_cells','any_seed_selected_cells','prior_neural_shared_top198','gini_prior','gini_neural']}),flush=True)
