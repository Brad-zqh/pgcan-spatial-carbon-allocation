"""Evidence-led model attribution, structural sensitivity and source dependence.

Run deep_model_analysis.py with the existing E: torch environment first, then
run this script with the plotting Python. All displayed quantities are saved
as source data. Ribbons/ranges describe actual draws and are not confidence
intervals. The AI schematic is never used as a quantitative source.
"""
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, PowerNorm, ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont
import redraw_figures as v
from visual_polish import density_norm,tag_small_text

P=v.HERE; SD=v.SD; D=v.grid
CS=pd.read_csv(SD/'Deep_channel_summary.csv')
CC=pd.read_csv(SD/'Deep_channel_category_changes.csv')
CV=pd.read_csv(SD/'Deep_channel_cell_values.csv')
RR=pd.read_csv(SD/'Deep_residual_response.csv')
SS=pd.read_csv(SD/'Deep_structural_sensitivity_cells.csv')
CD=pd.read_csv(SD/'Deep_category_design_vs_seed.csv')
SC=pd.read_csv(SD/'Deep_shared_source_screening.csv').set_index('map')
AUD=json.loads((P/'Deep_analysis_audit.json').read_text())
CAT=['industrial','transport_built','research','mixed','low_activity']
CO=v.LILAC; CT=v.TEAL
GROUPCOL=['#E8ECDF','#92B768','#E1A5C6','#AA146A']
GROUPLAB=['Below both cutoffs','Adjustment only','Design IQR only','Both above cutoff']

def heading(fig,title,subtitle):
    fig._caption_notes=getattr(fig,'_caption_notes',[])+[title,subtitle]

def footer(fig,lines):
    fig._caption_notes=getattr(fig,'_caption_notes',[])+[' '.join(lines)]

def title(a,letter,text,letter_x=-.08):
    a.set_title(text,loc='left',fontsize=8.8,pad=11)
    a.text(letter_x,1.06,letter,transform=a.transAxes,fontweight='bold',fontsize=11,va='bottom')

def style(a):
    v.quiet(a);a.tick_params(labelsize=7)

def response(a,metric,color,label):
    g=RR.groupby('residual_multiplier')[metric]
    x=g.mean().index.to_numpy();mean=g.mean().to_numpy()
    a.fill_between(x,g.min(),g.max(),color=color,alpha=.15,lw=0)
    a.plot(x,mean,color=color,lw=1.5,label=label)

def figure7():
    f=plt.figure(figsize=(v.WIDTH,7.65))
    heading(f,'What the network changes, and through which channel',
            'Exact decomposition of the fitted allocation · five saved checkpoints · 1,980 cells')
    z=[CV.coordinate_phi_pp,CV.context_phi_pp,CV.total_change_pp]
    lim=np.ceil(max(abs(x).max() for x in z)*1000)/1000
    norm=TwoSlopeNorm(0,-lim,lim)
    for i,(vals,lab) in enumerate(zip(z,['Coordinate contribution','Context contribution','Combined adjustment'])):
        a=f.add_axes([.055+i*.335,.688,.265,.222]);v.mapplot(a,vals,norm,v.DIFF)
        title(a,chr(97+i),lab)
    cb=f.colorbar(v.mpl.cm.ScalarMappable(norm=norm,cmap=v.DIFF),
        cax=f.add_axes([.29,.665,.42,.012]),orientation='horizontal',ticks=[-lim,0,lim])
    cb.outline.set_visible(False);cb.ax.tick_params(labelsize=7,length=2)
    cb.set_label('Attributed change in regional share (pp); shared linear scale',fontsize=7,labelpad=2)
    a=f.add_axes([.17,.40,.39,.18]);title(a,'d','Redistribution by functional category')
    yy=np.arange(5)
    for offset,key,c,label in [(-.19,'coordinate_change_pp',CO,'Coordinate'),(0,'context_change_pp',CT,'Context'),(.19,'total_change_pp',v.INK,'Combined')]:
        mean=CC.groupby('category')[key].mean().reindex(CAT)
        lo=CC.groupby('category')[key].min().reindex(CAT)
        hi=CC.groupby('category')[key].max().reindex(CAT)
        a.hlines(yy+offset,lo,hi,color=c,lw=1.4,alpha=.65)
        a.scatter(mean,yy+offset,c=c,s=17,zorder=3,label=label,edgecolors='white',lw=.4)
    a.axvline(0,c='#8E989E',lw=.65,ls='--')
    a.set_yticks(yy,[v.CAT[k][1] for k in CAT],fontsize=7);a.invert_yaxis()
    a.set_xlabel('Category share change (pp)',fontsize=7.5);style(a)
    a.legend(loc='upper left',bbox_to_anchor=(-.32,-.28),ncol=3,fontsize=6.7,handletextpad=.4,columnspacing=1)
    a=f.add_axes([.67,.40,.285,.18]);title(a,'e','Channel stability across seeds')
    for j,row in CS.iterrows():
        c=v.LILAC if row.cancellation_fraction>.4 else '#BFC9CC'
        a.plot([0,1,2],np.array([row.coordinate_tv,row.context_tv,row.net_tv])*100,c=c,lw=.8,alpha=.9,zorder=1)
    means=CS[['coordinate_tv','context_tv','net_tv']].mean().to_numpy()*100
    a.scatter([0,1,2],means,c=[CO,CT,v.INK],s=34,edgecolors='white',lw=.6,zorder=3)
    a.set_xticks([0,1,2],['Coordinate','Context','Net'],fontsize=7)
    a.set(ylabel='Attributed redistribution (TV, %)',ylim=(0,9))
    a.text(.02,.98,f'Mean within-seed cancellation: {CS.cancellation_fraction.mean()*100:.1f}%',
        transform=a.transAxes,fontsize=6.8,va='top')
    style(a)
    a=f.add_axes([.10,.105,.365,.17]);title(a,'f','Response to residual strength')
    response(a,'tv_prior',v.LILAC,'Distance from prior')
    response(a,'tv_target',v.INK,'Distance from target')
    a.axvline(1,c='#79858C',ls='--',lw=.7);a.text(1.025,.98,'Fitted',transform=a.get_xaxis_transform(),va='top',fontsize=6.7)
    a.set(xlabel='Residual multiplier τ',ylabel='Total variation distance',xlim=(0,2),ylim=(0,.18),xticks=[0,.5,1,1.5,2]);style(a)
    a.legend(loc='upper left',fontsize=6.7)
    a=f.add_axes([.62,.105,.335,.17]);title(a,'g','Location screening along the same response')
    response(a,'regulatory_coverage_pct',v.INK,'Regulatory (146)')
    response(a,'energy_coverage_pct',v.LILAC,'All energy (69)')
    a.axvline(1,c='#79858C',ls='--',lw=.7)
    a.set(xlabel='Residual multiplier τ',ylabel='Top-10% site coverage (%)',xlim=(0,2),ylim=(30,52),xticks=[0,.5,1,1.5,2]);style(a)
    a.legend(loc='upper left',fontsize=6.7)
    footer(f,['Two-channel Shapley values sum exactly to full allocation − epsilon-clipped prior. TV = half the sum of absolute share changes.',
              'Maps: five-seed mean; bars/ribbons: five-seed range, not confidence intervals. τ changes the fitted algorithm, not an environmental exposure.'])
    v.save(f,'Figure_7',['Deep_channel_cell_values.csv','Deep_channel_category_changes.csv','Deep_channel_summary.csv','Deep_residual_response.csv'],
        'Exact two-channel fitted-model decomposition, category heterogeneity, per-seed channel stability, and conditional response to residual scaling. Functional attributions are not causal effects.')

def figure8():
    f=plt.figure(figsize=(v.WIDTH,7.3))
    heading(f,'Structural assumptions dominate the spatial stability check',
            '256 original prior-design draws compared with five fitted training seeds; the ensembles have different meanings')
    hi=np.ceil(max(SS.prior_design_sd_pp.max(),SS.seed_sd_pp.max())*1000)/1000
    norm=PowerNorm(.5,0,hi)
    for i,(col,lab) in enumerate([('prior_design_sd_pp','Prior-design variation (256 draws)'),('seed_sd_pp','Training variation (five seeds)')]):
        a=f.add_axes([.035+i*.49,.66,.44,.20]);v.mapplot(a,SS[col],norm,v.SEQ);title(a,chr(97+i),lab)
    cb=f.colorbar(v.mpl.cm.ScalarMappable(norm=norm,cmap=v.SEQ),cax=f.add_axes([.29,.635,.42,.012]),orientation='horizontal',ticks=[0,hi/4,hi])
    cb.outline.set_visible(False);cb.ax.tick_params(labelsize=7,length=2)
    cb.set_label('Grid-share SD (pp); shared square-root colour stretch',fontsize=7,labelpad=2)
    a=f.add_axes([.17,.395,.43,.17]);title(a,'c','Category response to the two ensembles')
    prior=D.groupby('function_class').Clean_prior_share.sum()*100
    rng=np.random.default_rng(20261020)
    for i,cat in enumerate(CAT):
        b=CD[(CD.category==cat)&(CD.source=='Prior_design')].mass_pct.to_numpy()-prior[cat]
        s=CD[(CD.category==cat)&(CD.source=='Neural_seeds')].mass_pct.to_numpy()-prior[cat]
        vio=a.violinplot([b],positions=[i-.12],orientation='horizontal',widths=.55,showextrema=False)
        for body in vio['bodies']:body.set(facecolor=CT,edgecolor=CT,alpha=.35,linewidth=.5)
        a.scatter([np.median(b)],[i-.12],s=12,c=CT,zorder=3)
        a.scatter(s,np.full(len(s),i+.20)+rng.uniform(-.025,.025,len(s)),s=13,c=v.LILAC,edgecolors='white',lw=.3,zorder=3)
    a.axvline(0,c='#8E989E',ls='--',lw=.7)
    a.set_yticks(range(5),[v.CAT[k][1] for k in CAT],fontsize=7);a.invert_yaxis()
    a.set_xlabel('Category mass − reference-prior mass (pp)',fontsize=7.5);style(a)
    f.legend(handles=[Patch(fc=CT,alpha=.45,label='Prior designs (256)'),Line2D([],[],marker='o',ls='',color=v.LILAC,label='Fitted seeds (5)',ms=3)],
             loc='center',bbox_to_anchor=(.395,.321),ncol=2,fontsize=6.2,handlelength=1,columnspacing=.7)
    a=f.add_axes([.70,.395,.25,.17]);title(a,'d','Where variability comes from')
    xx=SS.seed_sd_pp.to_numpy();yy=SS.prior_design_sd_pp.to_numpy()
    positive=(xx>0)&(yy>0);xx=xx[positive];yy=yy[positive]
    hb=a.hexbin(np.log10(xx),np.log10(yy),gridsize=32,mincnt=1,cmap=v.DENSITY,linewidths=0)
    hb.set_norm(density_norm(hb.get_array()))
    maximum=float(hb.get_array().max());cb=f.colorbar(hb,cax=a.inset_axes([1.035,.05,.04,.9]),ticks=[1,2,10,maximum])
    cb.set_ticklabels(['1','2','10',f'{maximum:g}']);cb.ax.minorticks_off()
    cb.set_label('Cells',fontsize=6.2);cb.outline.set_visible(False);tag_small_text(cb)
    limits=(np.floor(np.log10(min(xx.min(),yy.min()))),np.ceil(np.log10(max(xx.max(),yy.max()))))
    a.plot(limits,limits,c=v.GOLD,ls='--',lw=.8)
    a.set(xlabel='log₁₀ seed SD (pp)',ylabel='log₁₀ design SD (pp)',xlim=limits,ylim=limits)
    a.text(.04,.95,f'Median SD ratio\n{AUD["median_design_sd_to_seed_sd_ratio"]:.1f}× (all cells)',transform=a.transAxes,va='top',fontsize=7.5,color=v.INK)
    a.text(.97,.035,f'n = {positive.sum():,} positive-SD cells',transform=a.transAxes,ha='right',fontsize=6.3,color='#52636B')
    style(a)
    a=f.add_axes([.025,.105,.51,.19]);title(a,'e','Candidates for an independent-data audit')
    cm=ListedColormap(GROUPCOL);bn=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4)
    v.mapplot(a,SS.diagnostic_group,bn,cm)
    a=f.add_axes([.64,.105,.31,.19]);title(a,'f','Two distinct diagnostic axes')
    # No shaded mask or low-opacity background: every cell is an opaque mark.
    colors=np.where(SS.prior_design_iqr_pp>=AUD['diagnostic_cutoffs']['design_iqr_pp_q80'],v.LILAC,v.TEAL)
    a.scatter(SS.neural_adjustment_abs_pp,SS.prior_design_iqr_pp,s=5,c=colors,alpha=1,lw=0,rasterized=False)
    xc=AUD['diagnostic_cutoffs']['absolute_adjustment_pp_q80'];yc=AUD['diagnostic_cutoffs']['design_iqr_pp_q80']
    a.axvline(xc,c='#777777',ls='--',lw=.65);a.axhline(yc,c='#777777',ls='--',lw=.65)
    a.legend(handles=[Line2D([],[],marker='o',ls='',color=v.TEAL,label='IQR below cutoff',ms=3),Line2D([],[],marker='o',ls='',color=v.LILAC,label='IQR at/above cutoff',ms=3)],
             loc='upper left',bbox_to_anchor=(.01,.99),fontsize=6.2,handletextpad=.3)
    a.set_xscale('symlog',linthresh=.0001);a.set_yscale('symlog',linthresh=.0001)
    a.set(xlabel='Absolute neural adjustment (pp)',ylabel='Prior-design IQR (pp)')
    a.set_xticks([0,.001,.01,.1],['0','.001','.01','.1']);a.set_yticks([0,.001,.01,.1],['0','.001','.01','.1']);style(a)
    handles=[Patch(fc=GROUPCOL[g],label=f'{GROUPLAB[g]} ({int((SS.diagnostic_group==g).sum()):,})') for g in range(4)]
    f.legend(handles=handles,loc='lower left',bbox_to_anchor=(.055,.055),ncol=2,fontsize=6.5,handlelength=1,columnspacing=1)
    footer(f,['Audit flags use the upper quintile of |neural adjustment| and prior-design IQR; cutoffs are not optimized against benchmark sites.',
              'Design draws are assumption perturbations; seed variability is algorithm repeatability. Neither is calibrated predictive uncertainty.'])
    v.save(f,'Figure_8',['Deep_structural_sensitivity_cells.csv','Deep_category_design_vs_seed.csv','Deep_prior_design_metrics.csv'],
        'Structural sensitivity compared with seed repeatability, category-specific responses, and bivariate independent-data audit flags. Descriptive ensembles are not probability distributions of true emissions.')

def figure9():
    f=plt.figure(figsize=(v.WIDTH,7.05))
    heading(f,'Screening depends strongly on the shared source package',
            'Saved source-removal experiments · exact location counts · explicit overlap between benchmark cohorts')
    # Measured cohort overlap replaces the redundant box-and-arrow sketch.
    a=f.add_axes([.20,.745,.69,.135]);title(a,'a','Benchmark cohort intersections')
    counts_overlap=np.array([146-59,59,69-59])
    a.bar(range(3),counts_overlap,color=[v.TEAL,v.LILAC,v.CORAL],width=.54)
    for i,value in enumerate(counts_overlap):a.text(i,value+2,str(value),ha='center',va='bottom',fontsize=8)
    a.set(ylim=(0,105),ylabel='Distinct sites',xticks=[],xlim=(-.6,2.6));style(a)
    dots=f.add_axes([.20,.665,.69,.065]);dots.set(xlim=(-.6,2.6),ylim=(-.5,1.5))
    membership=np.array([[1,0],[1,1],[0,1]])
    for i,col in enumerate([v.TEAL,v.LILAC,v.CORAL]):
        dots.scatter([i,i],[1,0],c=[col if b else '#E3E3E3' for b in membership[i]],s=24,zorder=3)
        if membership[i].all():dots.plot([i,i],[0,1],color=col,lw=1.2)
    dots.set_yticks([1,0],['Regulatory','Energy-user'],fontsize=7)
    dots.set_xticks(range(3),['Regulatory only','Shared','Energy only'],fontsize=7)
    dots.tick_params(length=0);dots.spines[:].set_visible(False)
    # Actual site multiplicities, not occupied-cell counts.
    energy=pd.read_csv(SD/'energy_benchmark_sites.csv')
    ei=pd.Index(D.grid_id).get_indexer(energy.grid_id)
    ecount=np.bincount(ei,minlength=len(D))
    counts=D.primary_count.to_numpy()
    shares=pd.read_csv(v.DATA/'comparison_shares.csv')
    curve=[]
    for k in ['Clean_prior','Direct_mixture','Full','No_POI_prior','No_POI_mixture','No_POI_neural']:
        order=np.lexsort((D.grid_id.to_numpy(str),-shares[k].to_numpy()))
        for pct in np.arange(1,51):
            top=order[:int(np.ceil(len(D)*pct/100))]
            curve.append({'map':k,'ranked_cells_pct':pct,'regulatory_pct':counts[top].sum()/146*100,'energy_pct':ecount[top].sum()/69*100})
    curves=pd.DataFrame(curve);curves.to_csv(SD/'Deep_source_removal_coverage_curves.csv',index=False)
    for x,letter,key,lab in [(.10,'b','regulatory_pct','Regulatory-site screening (n = 146)'),(.60,'c','energy_pct','Energy-user screening (n = 69)')]:
        a=f.add_axes([x,.398,.35,.21]);title(a,letter,lab)
        for base,removed,c in [('Clean_prior','No_POI_prior',v.TEAL),('Direct_mixture','No_POI_mixture',v.CORAL),('Full','No_POI_neural',v.INK)]:
            for k,ls in [(base,'-'),(removed,'--')]:
                s=curves[curves['map']==k];a.plot(s.ranked_cells_pct,s[key],c=c,lw=1.2,ls=ls)
        a.axvline(10,c='#B8C1C5',ls=':',lw=.8);a.set(xlim=(0,50),ylim=(0,105),xlabel='Top-ranked cells (%)',ylabel='Site coverage (%)');style(a)
    f.legend(handles=[Line2D([],[],c=v.TEAL,label='Prior'),Line2D([],[],c=v.CORAL,label='Direct mixture'),Line2D([],[],c=v.INK,label='PGCAN'),
        Line2D([],[],c=v.GRAY,ls='-',label='Original package'),Line2D([],[],c=v.GRAY,ls='--',label='Source removed')],
        loc='upper center',bbox_to_anchor=(.52,.352),ncol=5,fontsize=6.8,columnspacing=1,handlelength=1.8)
    a=f.add_axes([.19,.102,.34,.175]);title(a,'d','Top-198 screening: exact hit counts')
    methods=['Clean_prior','Direct_mixture','Full','No_POI_prior','No_POI_mixture','No_POI_neural']
    vals=SC.loc[methods,['regulatory_coverage_pct','energy_coverage_pct','unique_energy_hits']].to_numpy()
    hit=np.column_stack([np.rint(vals[:,0]*146/100),np.rint(vals[:,1]*69/100),vals[:,2]]).astype(int)
    fractions=hit/np.array([146,69,10])*100
    a.imshow(fractions,cmap=v.SEQ,vmin=0,vmax=60,aspect='auto');a._heatmap_units='Coverage (%)'
    for i in range(6):
        for j in range(3):
            a.text(j,i,f'{hit[i,j]}/{[146,69,10][j]}',ha='center',va='center',fontsize=7,color='white' if fractions[i,j]>35 else v.INK)
    a.set_yticks(range(6),['Prior','Mixture','PGCAN','Prior: removed','Mixture: removed','PGCAN: removed'],fontsize=6.8)
    a.set_xticks(range(3),['Regulatory','All energy','Energy-only'],fontsize=6.5);a.tick_params(length=0)
    a.axhline(2.5,c='white',lw=2);a.spines[['left','bottom']].set_visible(False)
    a=f.add_axes([.61,.105,.345,.175]);title(a,'e','Source removal changes the allocation map')
    diff=(shares.No_POI_neural-shares.Full)*100
    lim=np.ceil(abs(diff).max()*100)/100
    norm=TwoSlopeNorm(0,-lim,lim);v.mapplot(a,diff,norm,v.DIFF)
    cb=f.colorbar(v.mpl.cm.ScalarMappable(norm=norm,cmap=v.DIFF),
        cax=a.inset_axes([.29,-.055,.68,.04]),orientation='horizontal',ticks=[-lim,0,lim])
    cb.outline.set_visible(False);cb.ax.tick_params(labelsize=6.7,length=2)
    cb.set_label('Source-removed PGCAN − original (pp)',fontsize=7,labelpad=2)
    footer(f,['Top-198 = 10% of cells. Benchmark membership is withheld from training; POI geolocation is shared and 59 sites occur in both cohorts.',
              'The source package includes POI and linked patent/innovation fields. This perturbation measures source dependence, not emissions accuracy.'])
    v.save(f,'Figure_9',['Deep_shared_source_screening.csv','Deep_source_removal_coverage_curves.csv','Deep_source_removed_prior.npy','Deep_no_poi_seed_maps.npy','energy_benchmark_sites.csv'],
        'Benchmark cohort intersections (87 regulatory-only, 59 shared, 10 energy-only sites), full-threshold source-removal screening curves, exact counts on overlapping and energy-only cohorts, and spatial redistribution. The source package includes POI and linked innovation variables; removal changes both the prior/target construction and network inputs. The comparison does not isolate a single feature effect.')

def overview():
    # Current plots only; no chronological backup accumulation.
    names=['Figure_2','Figure_7','Figure_8','Figure_9']
    canvas=Image.new('RGB',(2200,2460),'#E9EEF0');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/timesbd.ttf',28)
    for i,n in enumerate(names):
        im=Image.open(v.QA/f'{n}.png').convert('RGB');im.thumbnail((1060,1150))
        x=20+(i%2)*1100;y=18+(i//2)*1230
        draw.text((x,y),n,font=font,fill=v.INK)
        canvas.paste(im,(x+(1060-im.width)//2,y+48))
    canvas.save(P/'Deep_analysis_overview.png')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--only',choices=['7','8','9'])
    selected=parser.parse_args().only
    for n,func in [('7',figure7),('8',figure8),('9',figure9)]:
        if selected is None or selected==n:func()
    manifest=json.loads((P/'figure_manifest.json').read_text(encoding='utf-8'))
    for entry in v.manifest:
        manifest=[m for m in manifest if m['figure']!=entry['figure']]+[entry]
    def sortkey(e):
        s=e['figure'].removeprefix('Figure_');return (s.startswith('S'),int(s.removeprefix('S')))
    manifest.sort(key=sortkey)
    (P/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    overview()
