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
from refined_primitives import signed_violin,overlap_ribbons,density_scatter,emphasis_line

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
    line,=a.plot(x,mean,color=color,lw=1.5,label=label)
    emphasis_line(line,color,width=3.8)
    a.plot(x[::5],mean[::5],ls='',marker='o',ms=2.5,color=color,mec='white',mew=.35)

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
        intervals=a.hlines(yy+offset,lo,hi,color=c,lw=1.25,alpha=.85)
        emphasis_line(intervals,c,width=3.4)
        a.scatter(mean,yy+offset,c=c,s=17,zorder=3,label=label,edgecolors='white',lw=.4)
    a.axvline(0,c='#8E989E',lw=.65,ls='--')
    a.set_yticks(yy,[v.CAT[k][1] for k in CAT],fontsize=7);a.invert_yaxis()
    a.set_xlabel('Category share change (pp)',fontsize=7.5);style(a)
    a.legend(loc='upper left',bbox_to_anchor=(-.32,-.28),ncol=3,fontsize=6.7,handletextpad=.4,columnspacing=1)
    a=f.add_axes([.67,.40,.285,.18]);title(a,'e','Channel stability across seeds')
    for j,row in CS.iterrows():
        c=['#AAC19E','#D7AAC5','#6F9958','#B85B91','#B5C2B8'][j%5]
        a.plot([0,1,2],np.array([row.coordinate_tv,row.context_tv,row.net_tv])*100,c=c,lw=.8,alpha=.65,zorder=1)
    means=CS[['coordinate_tv','context_tv','net_tv']].mean().to_numpy()*100
    a.scatter([0,1,2],means,c=[CO,CT,v.INK],s=34,edgecolors='white',lw=.6,zorder=3)
    lo=CS[['coordinate_tv','context_tv','net_tv']].min().to_numpy()*100
    hi=CS[['coordinate_tv','context_tv','net_tv']].max().to_numpy()*100
    a.fill_between([0,1,2],lo,hi,color='#ADBEA4',alpha=.15,lw=0,zorder=.3)
    a.plot([0,1,2],means,color='#536653',lw=1.25,zorder=2)
    a.set_xticks([0,1,2],['Coordinate','Context','Net'],fontsize=7)
    a.set(ylabel='Attributed redistribution (TV, %)',ylim=(0,9))
    a.text(.02,.98,f'Mean within-seed cancellation: {CS.cancellation_fraction.mean()*100:.1f}%',
        transform=a.transAxes,fontsize=6.8,va='top')
    style(a)
    a=f.add_axes([.10,.105,.365,.17]);title(a,'f','Response to residual strength')
    response(a,'tv_prior',v.LILAC,'Distance from prior')
    response(a,'tv_target',v.INK,'Distance from target')
    xx=RR.groupby('residual_multiplier').tv_target.mean().index.to_numpy()
    a.axvline(1,c='#79858C',ls='--',lw=.7);a.text(1.025,.98,'Fitted',transform=a.get_xaxis_transform(),va='top',fontsize=6.7)
    a.set(xlabel='Residual multiplier τ',ylabel='Total variation distance',xlim=(0,2),ylim=(0,.18),xticks=[0,.5,1,1.5,2]);style(a)
    a.legend(loc='lower left',bbox_to_anchor=(0,1.025),fontsize=6.7,ncol=1,borderaxespad=0)
    a=f.add_axes([.62,.105,.335,.17]);title(a,'g','Location screening along the same response')
    response(a,'regulatory_coverage_pct',v.INK,'Regulatory (146)')
    response(a,'energy_coverage_pct',v.LILAC,'All energy (69)')
    a.axvline(1,c='#79858C',ls='--',lw=.7)
    a.set(xlabel='Residual multiplier τ',ylabel='Top-10% site coverage (%)',xlim=(0,2),ylim=(30,52),xticks=[0,.5,1,1.5,2]);style(a)
    a.legend(loc='lower left',bbox_to_anchor=(0,1.025),fontsize=6.7,ncol=1,borderaxespad=0)
    footer(f,['Two-channel Shapley values sum exactly to full allocation − epsilon-clipped prior. TV = half the sum of absolute share changes.',
              'Maps: five-seed mean; bars/ribbons: five-seed range, not confidence intervals. τ changes the fitted algorithm, not an environmental exposure.'])
    v.save(f,'Figure_7',['Deep_channel_cell_values.csv','Deep_channel_category_changes.csv','Deep_channel_summary.csv','Deep_residual_response.csv'],
        'Exact two-channel fitted-model decomposition, category heterogeneity, per-seed channel stability, and conditional response to residual scaling. Thin translucent under-strokes trace the exact interval/response lines for emphasis; shaded response ribbons retain the actual seed min–max range. Functional attributions are not causal effects.')

def figure8():
    f=plt.figure(figsize=(v.WIDTH,7.3))
    heading(f,'Structural assumptions dominate the spatial stability check',
            '256 original prior-design draws compared with five fitted training seeds; the ensembles have different meanings')
    hi=np.ceil(max(SS.prior_design_sd_pp.max(),SS.seed_sd_pp.max())*1000)/1000
    norm=PowerNorm(.5,0,hi)
    aligned_maps=[]
    for i,(col,lab) in enumerate([('prior_design_sd_pp','Prior-design variation (256 draws)'),('seed_sd_pp','Training variation (five seeds)')]):
        a=f.add_axes([.15+i*.50,.73,.33,.17]);a._external_coordinate_gutter=True
        v.mapplot(a,SS[col],norm,v.SEQ);title(a,chr(97+i),lab);aligned_maps.append(a)
    frame_height=aligned_maps[0].get_position().height
    cb=f.colorbar(v.mpl.cm.ScalarMappable(norm=norm,cmap=v.SEQ),cax=f.add_axes([.29,.635,.42,.012]),orientation='horizontal',ticks=[0,hi/4,hi])
    cb.outline.set_visible(False);cb.ax.tick_params(labelsize=7,length=2)
    cb.set_label('Grid-share SD (pp); shared square-root colour stretch',fontsize=7,labelpad=2)
    a=f.add_axes([.15,.415,.33,frame_height]);category_axis=a;title(a,'c','Category response to the two ensembles')
    prior=D.groupby('function_class').Clean_prior_share.sum()*100
    rng=np.random.default_rng(20261020)
    for i,cat in enumerate(CAT):
        b=CD[(CD.category==cat)&(CD.source=='Prior_design')].mass_pct.to_numpy()-prior[cat]
        s=CD[(CD.category==cat)&(CD.source=='Neural_seeds')].mass_pct.to_numpy()-prior[cat]
        signed_violin(a,b,i-.11,v.DIFF,width=.57)
        a.scatter(s,i+.20+np.linspace(-.05,.05,len(s)),s=19,marker='D',c='#444A45',edgecolors='white',lw=.4,zorder=4)
    a.axvline(0,c='#8E989E',ls='--',lw=.7)
    a.set_yticks(range(5),[v.CAT[k][1] for k in CAT],fontsize=7);a.invert_yaxis()
    a.set_xlabel('Category mass − reference-prior mass (pp)',fontsize=7.5);style(a)
    f.legend(handles=[Patch(fc='#D8DCCF',alpha=.75,label='Prior designs (256)'),Line2D([],[],marker='D',ls='',color='#444A45',label='Fitted seeds (5)',ms=3)],
             loc='center',bbox_to_anchor=(.315,.35),ncol=2,fontsize=6.2,handlelength=1,columnspacing=.7)
    a=f.add_axes([.65,.415,.33,frame_height]);density_axis=a;title(a,'d','Where variability comes from')
    xx=SS.seed_sd_pp.to_numpy();yy=SS.prior_design_sd_pp.to_numpy()
    positive=(xx>0)&(yy>0);xx=xx[positive];yy=yy[positive]
    hb,_,_=density_scatter(a,np.log10(xx),np.log10(yy),v.DENSITY,size=3.0)
    cb=f.colorbar(hb,cax=a.inset_axes([1.035,0,.035,1]),ticks=[0,50,100]);cb.ax.minorticks_off()
    cb.set_label('Density percentile (%)',fontsize=6.2);cb.outline.set_visible(False);tag_small_text(cb)
    limits=(np.floor(np.log10(min(xx.min(),yy.min()))),np.ceil(np.log10(max(xx.max(),yy.max()))))
    a.plot(limits,limits,c=v.GOLD,ls='--',lw=.8)
    a.set(xlabel='log₁₀ seed SD (pp)',ylabel='log₁₀ design SD (pp)',xlim=limits,ylim=limits)
    a.text(.04,.95,f'Median SD ratio\n{AUD["median_design_sd_to_seed_sd_ratio"]:.1f}× (all cells)',transform=a.transAxes,va='top',fontsize=7.5,color=v.INK)
    a.text(.97,.035,f'n = {positive.sum():,} positive-SD cells',transform=a.transAxes,ha='right',fontsize=6.3,color='#52636B')
    style(a)
    a=f.add_axes([.15,.10,.33,frame_height]);a._external_coordinate_gutter=True;flag_axis=a;title(a,'e','Candidates for an independent-data audit')
    cm=ListedColormap(GROUPCOL);bn=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4)
    v.mapplot(a,SS.diagnostic_group,bn,cm)
    a=f.add_axes([.65,.10,.33,frame_height]);adjustment_axis=a;title(a,'f','Two distinct diagnostic axes')
    tx=lambda value:np.arcsinh(np.asarray(value)/.0001)
    hb,_,_=density_scatter(a,tx(SS.neural_adjustment_abs_pp),tx(SS.prior_design_iqr_pp),v.DENSITY,size=3.2)
    xc=AUD['diagnostic_cutoffs']['absolute_adjustment_pp_q80'];yc=AUD['diagnostic_cutoffs']['design_iqr_pp_q80']
    a.axvline(tx(xc),c='#777777',ls='--',lw=.65);a.axhline(tx(yc),c='#777777',ls='--',lw=.65)
    cb=f.colorbar(hb,cax=a.inset_axes([1.035,0,.035,1]),ticks=[0,50,100]);cb.outline.set_visible(False)
    cb.set_label('Density percentile (%)',fontsize=6.2);tag_small_text(cb)
    a.set(xlabel='Absolute neural adjustment (pp)',ylabel='Prior-design IQR (pp)')
    a.set(xlim=(0,tx(SS.neural_adjustment_abs_pp.max())*1.03),ylim=(0,tx(SS.prior_design_iqr_pp.max())*1.03))
    a.set_xticks(tx([0,.001,.01,.1]),['0','.001','.01','.1']);a.set_yticks(tx([0,.001,.01,.1]),['0','.001','.01','.1']);style(a)
    handles=[Patch(fc=GROUPCOL[g],label=f'{GROUPLAB[g]} ({int((SS.diagnostic_group==g).sum()):,})') for g in range(4)]
    f.legend(handles=handles,loc='lower left',bbox_to_anchor=(.04,.015),ncol=2,fontsize=6.5,handlelength=1,columnspacing=1)
    f._aligned_axes_columns=[[aligned_maps[0],category_axis,flag_axis],[aligned_maps[1],density_axis,adjustment_axis]]
    f._aligned_axes_rows=[[aligned_maps[0],aligned_maps[1]],[category_axis,density_axis],[flag_axis,adjustment_axis]]
    footer(f,['Audit flags use the upper quintile of |neural adjustment| and prior-design IQR; cutoffs are not optimized against benchmark sites.',
              'Design draws are assumption perturbations; seed variability is algorithm repeatability. Neither is calibrated predictive uncertainty.'])
    f._density_encoding={'type':'KDE percentile','panels':['d','f'],'bandwidth':'Scott',
        'point_positions':'Actual cell pairs: log10 in d and asinh(pp/0.0001) in f; no jitter'}
    v.save(f,'Figure_8',['Deep_structural_sensitivity_cells.csv','Deep_category_design_vs_seed.csv','Deep_prior_design_metrics.csv'],
        'Structural sensitivity compared with seed repeatability, category-specific responses, and bivariate independent-data audit flags. Category violins use a pink-negative/green-positive signed gradient with median and IQR; dark diamonds show all five fitted seeds, separated only in the categorical direction. Density panels use actual cell locations and within-panel Scott-KDE percentiles. Panel f uses asinh(pp/0.0001) coordinates, labelled in raw pp, with nonnegative limits; dashed lines retain the original upper-quintile cutoffs. Descriptive ensembles are not probability distributions of true emissions.')

def figure9():
    f=plt.figure(figsize=(v.WIDTH,5.2))
    heading(f,'Screening depends strongly on the shared source package',
            'Saved source-removal experiments · exact location counts · explicit overlap between benchmark cohorts')
    a=f.add_axes([.035,.61,.27,.28]);title(a,'a','Benchmark cohort intersections')
    overlap_ribbons(a,87,59,10)
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
    for x,letter,key,lab in [(.41,'b','regulatory_pct','Regulatory-site screening (n = 146)'),(.76,'c','energy_pct','Energy-user screening (n = 69)')]:
        a=f.add_axes([x,.61,.22,.28]);title(a,letter,lab)
        for base,removed,c in [('Clean_prior','No_POI_prior',v.TEAL),('Direct_mixture','No_POI_mixture',v.CORAL),('Full','No_POI_neural',v.INK)]:
            for k,ls in [(base,'-'),(removed,'--')]:
                s=curves[curves['map']==k];a.plot(s.ranked_cells_pct,s[key],c=c,lw=1.2,ls=ls)
        a.axvline(10,c='#B8C1C5',ls=':',lw=.8);a.set(xlim=(0,50),ylim=(0,105),xlabel='Top-ranked cells (%)',ylabel='Site coverage (%)');style(a)
    f.legend(handles=[Line2D([],[],c=v.TEAL,label='Prior'),Line2D([],[],c=v.CORAL,label='Direct mixture'),Line2D([],[],c=v.INK,label='PGCAN'),
        Line2D([],[],c=v.GRAY,ls='-',label='Original package'),Line2D([],[],c=v.GRAY,ls='--',label='Source removed')],
        loc='upper center',bbox_to_anchor=(.62,.505),ncol=3,fontsize=6.4,columnspacing=.8,handlelength=1.3)
    a=f.add_axes([.16,.12,.30,.28]);title(a,'d','Top-198 screening: exact hit counts')
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
    a=f.add_axes([.57,.12,.40,.28]);title(a,'e','Source removal changes the allocation map')
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
        'Benchmark cohort intersections (87 regulatory-only, 59 shared, 10 energy-only sites), full-threshold source-removal screening curves, exact counts on overlapping and energy-only cohorts, and spatial redistribution. Panel a is an area-proportional two-set diagram: circle areas represent the 146 and 69 sites, and their intersection represents 59 shared sites; outside callouts label narrow exclusive regions. The source package includes POI and linked innovation variables; removal changes both prior/target construction and network inputs. The comparison does not isolate a single feature effect.')

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
