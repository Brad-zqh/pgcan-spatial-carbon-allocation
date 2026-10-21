"""Evidence-led dense model-comparison, ablation and graph-audit plates."""
import os
from pathlib import Path
import json,string,textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,TwoSlopeNorm,ListedColormap,BoundaryNorm,SymLogNorm
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch,Patch
from matplotlib.ticker import LogLocator,NullFormatter
from sklearn.metrics import precision_recall_curve,r2_score
import redraw_figures as v
from visual_polish import density_norm

P=v.HERE;E=P/'experiments';S=v.SD
protocol=json.loads((E/'protocol.json').read_text());assert protocol['completed']
assert json.loads((E/'direct_baseline_protocol.json').read_text())['completed']
fold=pd.concat([pd.read_csv(S/'Benchmark_fold_metrics.csv'),pd.read_csv(S/'Benchmark_direct_fold_metrics.csv')],ignore_index=True)
maps=pd.concat([pd.read_csv(S/'Benchmark_map_metrics.csv'),pd.read_csv(S/'Benchmark_direct_map_metrics.csv')],ignore_index=True)
cells=pd.read_csv(S/'Benchmark_cell_manifest.csv');target=cells.target_share.to_numpy()
names=['Ridge_direct','Local_ridge_direct','Random_forest_direct','Extra_trees_direct','Hist_gradient_direct','XGBoost_direct','MLP_direct','PGCAN','Multi_relation']
labels={'Ridge_direct':'Ridge','Local_ridge_direct':'Local ridge','Random_forest_direct':'Random forest','Extra_trees_direct':'ExtraTrees',
 'Hist_gradient_direct':'Hist. boosting','XGBoost_direct':'XGBoost','MLP_direct':'MLP','PGCAN':'PGCAN refit','Multi_relation':'Multi-relation',
 'No_coordinate':'No coordinate','No_context_branch':'No context branch','No_function_input':'No function input',
 'No_graph_input':'No graph input','No_graph_penalty':'No graph penalty','No_graph':'No graph',
 'No_prior_anchor':'No prior anchor','No_prior_penalty':'No prior penalty',
 'MR_no_spatial':'No spatial paths','MR_no_function':'No function relation','MR_no_built':'No built relation',
 'MR_fixed_gate':'Uniform gate','MR_rewired':'Rewired relations'}
colors=dict(zip(names,['#47782B','#7CA554','#D796BB','#B84E8A','#BDCD8B','#679C45','#9F286E','#2E5D1B','#B01572']))
colors.update({k:('#B01572' if k.startswith('MR') else '#6B9C49') for k in labels if k not in colors})
colors['Multi_relation']='#B01572';colors['No_graph']='#83AD59';colors['No_coordinate']='#DEAAC8'
colors.update({'No_context_branch':'#B31B73','No_graph_penalty':'#D779AD','No_prior_anchor':'#A8005B'})
rc=v.grid.primary_count.to_numpy();cats=v.grid.function_class.to_numpy()
xy=np.column_stack(v.Transformer.from_crs(4326,32651,always_xy=True).transform(cells.center_lon,cells.center_lat))
gates=np.array([np.load(E/f'gates_Multi_relation_{seed}.npy') for seed in protocol['config']['seeds']])
G=gates.mean(0)
REL=['Spatial proximity','Functional similarity','Built/activity similarity']
RELC=['#32651D','#B11B76','#99BD60']

def load(name,scope='oof'):return np.load(E/f'{scope}_{name}.npy')
def mean(name,scope='oof'):return load(name,scope).mean(0)
def heading(fig,title,subtitle):
 fig._caption_notes=getattr(fig,'_caption_notes',[])+[title,subtitle]
def axes_grid(title,subtitle,rows=5):
 fig=plt.figure(figsize=(v.WIDTH,8.75 if rows==5 else 7.5));heading(fig,title,subtitle)
 gs=fig.add_gridspec(rows,3,left=.105,right=.975,bottom=.055,top=.945,hspace=.72,wspace=.74)
 return fig,[fig.add_subplot(gs[i,j]) for i in range(rows) for j in range(3)]
def title(ax,letter,text):
 fig=ax.figure;fig._panel_descriptions=getattr(fig,'_panel_descriptions',{});fig._panel_descriptions[letter]=text
 short={'Fixed spatial test folds':'Spatial folds','Covariate spatial dependence':'Spatial dependence','Input sparsity audit':'Zero prevalence',
 'Held-out proxy RMSE':'Proxy RMSE','Held-out proxy MAE':'Proxy MAE','Method profiles: raw values':'Method profiles',
 'PGCAN refit: held-out shares':'PGCAN','Multi-relation: held-out shares':'Multi-relation','Ridge: held-out shares':'Ridge',
 'Distribution of held-out errors':'Error distribution','Category-specific MAE (pp)':'Category MAE','Spatial-support heterogeneity':'Spatial support',
 'OOF site-screening curves':'Site coverage','Site-membership precision–recall':'Precision–recall','Recorded CPU fitting cost':'Fitting time',
 'Component inclusion':'Components','Paired reconstruction changes':'Paired error changes','Full-map screening changes':'Coverage changes',
 'Inner spatial learning curves':'Learning curves','Spatial early-stopping budget':'Selected epochs','Does removing the graph hurt?':'Graph removal',
 'Category mass redistribution':'Category changes','Top-198 seed repeatability':'Selection stability','Complexity–error trade-off':'Cost and error',
 'Error-distribution controls':'Error distribution','Refitted-map site screening':'Site coverage','How strongly each model adjusts':'Prior adjustment',
 'Relation-edge Jaccard overlap':'Edge overlap','How far each relation reaches':'Edge distance','Functional homophily audit':'Functional homophily',
 'Gate allocation by category':'Category gates','Normalized gate entropy [0, 1]':'Gate entropy','Largest learned relation gate':'Dominant relation',
 'Relation and topology controls':'Relation controls','Gate dispersion: 5–95% range':'Gate dispersion','Do semantic edges aid screening?':'Site coverage'}
 text=short.get(text,text)
 if ax.get_position().width<.3:text=textwrap.fill(text,width=27,break_long_words=False)
 ax.set_title(text,loc='left',fontsize=7.5,pad=8,fontweight='bold')
 letter_x=-.045 if ax.get_position().width>.3 else -.17
 ax.text(letter_x,1.10,letter,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')
 ax.tick_params(labelsize=6.2);ax.xaxis.label.set_size(6.7);ax.yaxis.label.set_size(6.7)
def footer(fig,text):
 fig._caption_notes=getattr(fig,'_caption_notes',[])+[text.replace('\n',' ')]
def forest(ax,field,keys=names,log=False,reference=None):
 for i,key in enumerate(keys):
  a=fold.loc[fold.method.eq(key),field].to_numpy()
  ax.hlines(i,a.min(),a.max(),color=colors.get(key,v.TEAL),lw=1.1,alpha=.75)
  ax.scatter(a.mean(),i,s=17,c=colors.get(key,v.TEAL),ec='white',lw=.4,zorder=3)
 ax.set_yticks(range(len(keys)),[labels.get(k,k) for k in keys],fontsize=5.9);ax.invert_yaxis()
 if log:
  ax.set_xscale('log');ax.xaxis.set_major_locator(LogLocator(base=10,subs=(1,),numticks=4));ax.xaxis.set_minor_formatter(NullFormatter())
 if reference is not None:ax.axvline(reference,c='#8B9885',lw=.6,ls=':')
 v.quiet(ax,'x');ax.tick_params(axis='y',length=0)
def curve(ax,x,label,color):
 z=np.sort(x);ax.step(z,np.arange(1,len(z)+1)/len(z),where='post',label=label,c=color,lw=1)
def joint(ax,key):
 pred=mean(key);mask=(target>0)&(pred>0)
 x=np.log10(target[mask]*100);y=np.log10(pred[mask]*100)
 h=ax.hexbin(x,y,gridsize=50,mincnt=1,cmap=v.DENSITY,bins='log',linewidths=0)
 h.set_norm(density_norm(h.get_array()))
 maximum=float(h.get_array().max());cb=ax.figure.colorbar(h,cax=ax.inset_axes([1.02,.08,.035,.84]),ticks=[1,maximum])
 cb.set_ticklabels(['1',f'{maximum:g}']);cb.outline.set_visible(False);cb.ax.minorticks_off();cb.ax.tick_params(labelsize=6.2,length=1,pad=1)
 for t in cb.ax.get_yticklabels():t.set_gid('map_colourbar_label')
 lim=[-4.1,.1];ax.plot(lim,lim,ls='--',c=v.LILAC,lw=.7)
 ax.set(xlim=lim,ylim=lim,xlabel='Proxy target (log₁₀ %)',ylabel='OOF prediction (log₁₀ %)')
 ax.text(.03,.96,f'Pooled R² = {r2_score(target,pred):.3f}',transform=ax.transAxes,va='top',fontsize=6)
 ax.text(.03,.82,f'Positive target cells: {mask.sum():,}',transform=ax.transAxes,va='top',fontsize=5.7)
def coverage(ax,keys,scope='oof'):
 for key in keys:
  p=mean(key,scope);order=np.lexsort((cells.grid_id.to_numpy(str),-p));q=np.arange(.01,.501,.01)
  cov=[rc[order[:int(len(p)*a)]].sum()/146*100 for a in q]
  ax.plot(q*100,cov,c=colors.get(key,v.TEAL),lw=1.1,label=labels.get(key,key))
 ax.axvline(10,c='#8B9885',lw=.6,ls=':');ax.set(xlim=(0,50),ylim=(0,102),xlabel='Top-ranked cells (%)',ylabel='Regulatory coverage (%)')
def delta_map(ax,key,baseline='PGCAN',lim=.1):
 vals=(mean(key,'full')-mean(baseline,'full'))*100
 v.mapplot(ax,vals,SymLogNorm(linthresh=lim/100,vmin=-lim,vmax=lim,base=10),v.DIFF)
 ax.figure._caption_notes=getattr(ax.figure,'_caption_notes',[])+[f'{labels.get(key,key)} minus {labels.get(baseline,baseline)}: full-map TV = {abs(vals).sum()/2:.2f}%.']

def benchmark_plate():
 f,axs=axes_grid('Model comparison under spatial separation','10 km blocks · 1 km exclusion buffer · five folds · three seeds for stochastic models')
 a=axs[0];cm=ListedColormap(['#517F32','#8DB860','#C5DCA2','#DEA3C4','#B12078'])
 v.mapplot(a,cells.fold,BoundaryNorm(np.arange(.5,6),5),cm);title(a,'a','Fixed spatial test folds')
 a.legend(handles=[Patch(fc=cm(i),label=str(i+1)) for i in range(5)],loc='upper center',bbox_to_anchor=(.5,-.48),ncol=5,fontsize=6.8,handlelength=.7,columnspacing=.5,borderaxespad=0)
 a=axs[1];q=pd.read_csv(S/'Benchmark_covariate_correlogram.csv')
 a.plot((q.distance_lower_m+500)/1000,q.ntl_standardized_product,c=v.INK,lw=1.1,marker='o',ms=2)
 a.axhline(0,c='#9BA58F',lw=.6);a.axvline(10,c=v.LILAC,ls='--',lw=.8)
 a.set(xlabel='Pair separation (km)',ylabel='Standardized NTL product');title(a,'b','Covariate spatial dependence')
 a=axs[2];audit=pd.read_csv(S/'Benchmark_input_audit.csv')
 a.barh(range(9),audit.zero_fraction*100,color=[v.LILAC,v.CORAL,v.LILAC,v.TEAL,v.INK,v.CORAL,v.TEAL,v.GOLD,v.TEAL],height=.65)
 a.set_yticks(range(9),['NTL','Enterprise','Industrial','Floor area','Roads','Patents','Land 1','Land 2','Land 3'],fontsize=5.7)
 a.invert_yaxis();a.set(xlim=(0,100),xlabel='Zero-valued cells (%)');title(a,'c','Input sparsity audit')
 for a,field,l,t in [(axs[3],'proxy_rmse_pp','d','Held-out proxy RMSE'),(axs[4],'proxy_mae_pp','e','Held-out proxy MAE')]:
  forest(a,field,log=True);a.set_xlabel('Share error (pp; log scale)');title(a,l,t)
 a=axs[5];rows=[]
 for key in names:
  rows.append([fold.loc[fold.method.eq(key),c].mean() for c in ['proxy_rmse_pp','proxy_mae_pp','proxy_r2']])
 vals=np.array(rows);normalized=(vals-vals.min(0))/np.maximum(np.ptp(vals,axis=0),1e-12);normalized[:,:2]=1-normalized[:,:2]
 a.imshow(normalized,cmap=v.SEQ,vmin=0,vmax=1,aspect='auto');a._heatmap_units='Relative score'
 for i,row in enumerate(vals):
  for j,value in enumerate(row):a.text(j,i,f'{value:.3g}' if j<2 else f'{value:.3f}',ha='center',va='center',fontsize=5.8,color='white' if normalized[i,j]>.65 else '#24351B')
 a.set_yticks(range(9),[labels[k] for k in names],fontsize=5.7);a.set_xticks(range(3),['RMSE','MAE','R²'],fontsize=6.2);a.tick_params(length=0);title(a,'f','Method profiles: raw values')
 for a,key,l in zip(axs[6:9],['PGCAN','Multi_relation','Ridge_direct'],'ghi'):
  joint(a,key);title(a,l,labels[key]+': held-out shares')
 a=axs[9]
 for key in ['Ridge_direct','Random_forest_direct','PGCAN','Multi_relation']:
  curve(a,abs(mean(key)-target)*100,labels[key],colors[key])
 a.set_xscale('symlog',linthresh=1e-5);a.set(xlabel='Absolute share error (pp)',ylabel='Cumulative cells',ylim=(0,1.02));title(a,'j','Distribution of held-out errors')
 a.legend(fontsize=5.7,loc='lower right',handlelength=1)
 a=axs[10];category_order=['industrial','transport_built','research','mixed','low_activity']
 val=np.array([[np.mean(abs(mean(k)[cats==c]-target[cats==c]))*100 for c in category_order] for k in names])
 norm=Normalize(0,float(val.max()));image=a.imshow(val,cmap=v.HEAT,norm=norm,aspect='auto')
 # Dense five-column values belong in the source table. At the requested
 # larger type size they cannot fit these cells without colliding.
 pd.DataFrame(val,index=names,columns=category_order).to_csv(S/'Benchmark_category_mae_pp.csv',index_label='method')
 cb=f.colorbar(image,cax=a.inset_axes([.15,-.29,.80,.04]),orientation='horizontal',ticks=[0,.025,.05])
 cb.outline.set_visible(False);cb.ax.tick_params(labelsize=6.8,length=2,pad=1)
 cb.ax.text(-.07,.5,'pp',transform=cb.ax.transAxes,ha='right',va='center',fontsize=6.8)
 a.set_yticks(range(9),[labels[k] for k in names],fontsize=5.6);a.set_xticks(range(5),['Ind.','Trans.','Res.','Mixed','Low'],fontsize=5.5);a.tick_params(length=0);title(a,'k','Category-specific MAE (pp)')
 a=axs[11];strata=pd.read_csv(S/'Benchmark_support_strata.csv')
 for k in ['PGCAN','Multi_relation','No_graph']:
  gg=strata[strata.method.eq(k)].groupby('support').proxy_rmse_pp.mean().reindex(['dense','middle','sparse'])
  a.plot(range(3),gg,c=colors[k],marker='o',ms=3,lw=1,label=labels[k])
 a.set_xticks(range(3),['Dense','Middle','Sparse'],fontsize=6);a.set(ylabel='Proxy RMSE (pp)',xlabel='Relative training support');a.legend(fontsize=5.7);title(a,'l','Spatial-support heterogeneity')
 a=axs[12];coverage(a,['Ridge_direct','Random_forest_direct','PGCAN','Multi_relation']);title(a,'m','OOF site-screening curves')
 a=axs[13]
 for k in ['Ridge_direct','Random_forest_direct','PGCAN','Multi_relation']:
  pr,re,_=precision_recall_curve(rc>0,mean(k));a.plot(re,pr,c=colors[k],lw=1,label=labels[k])
 a.axhline((rc>0).mean(),c='#A0A890',ls=':',lw=.6);a.set(xlabel='Recall of occupied cells',ylabel='Precision',xlim=(0,1),ylim=(0,1));title(a,'n','Site-membership precision–recall')
 a=axs[14];forest(a,'elapsed_seconds',log=True);a.set_xlabel('Fit + selection time (s)');title(a,'o','Recorded CPU fitting cost')
 f._layout_clearance_pairs=[(cb.ax,axs[13]._left_title)]
 footer(f,'Intervals in d–e and o: observed fold/seed ranges; dots: means. Colours in f are scaled separately by metric.\nAll regression metrics evaluate a constructed proxy, not measured emissions. Site curves use 146 regulatory sites; no site labels enter training.')
 v.save(f,'Figure_10',['Benchmark_fold_metrics.csv','Benchmark_direct_fold_metrics.csv','Benchmark_cell_manifest.csv','Benchmark_input_audit.csv','Benchmark_covariate_correlogram.csv','Benchmark_support_strata.csv','Benchmark_category_mae_pp.csv'],
  'Fifteen-panel spatial benchmark. Direct-share baselines receive the declared prior and the same train-only neighbourhood covariates. Proxy RMSE/MAE/R2 use unrenormalized held-out predictions calibrated on training mass; OOF screening uses separately normalized maps. Error and timing ranges are actual fold/seed ranges. Spatial support is defined by held-out query-to-training distance tertiles. Category MAE uses one common linear colour scale; the exact nine-by-five values are supplied in Benchmark_category_mae_pp.csv instead of overlapping in-cell numbers. No measured emissions are available.')

def paired_difference(keys,field,baseline='PGCAN',scope='fold'):
 table=fold if scope=='fold' else maps[maps.scope.eq('full_domain_refit')]
 join=['seed','fold'] if scope=='fold' else ['seed']
 reference=table[table.method.eq(baseline)][join+[field]].rename(columns={field:'baseline'})
 result=[]
 for key in keys:
  q=table[table.method.eq(key)].merge(reference,on=join)
  result.append((key,q[field].to_numpy()-q.baseline.to_numpy()))
 return result

def ablation_plate():
 keys=['No_coordinate','No_context_branch','No_function_input','No_graph_input','No_graph_penalty','No_graph','No_prior_anchor','No_prior_penalty','Multi_relation']
 f,axs=axes_grid('Which components earn their complexity?','Matched spatial folds and seeds · refitted ablations · full-domain maps shown separately')
 a=axs[0];fields=['C','M','F','GX','GL','P','PL']
 for axis in axs[:3]:
  pos=axis.get_position();axis.set_position([pos.x0,.79,pos.width,.155])
 f._caption_notes=getattr(f,'_caption_notes',[])+['Component keys: C coordinate branch; M context MLP; F functional one-hot; GX graph inputs; GL graph loss; P prior anchor; PL prior penalty.']
 matrix=np.ones((len(keys)+1,7));allkeys=['PGCAN']+keys
 for i,k in enumerate(allkeys):
  for j,field in enumerate(['coordinate','context','function','graph','penalty','anchor','prior_penalty']):
   matrix[i,j]=protocol['variants'].get(k,{}).get(field,True)
 a.imshow(matrix,cmap=ListedColormap(['#F2D9E7','#6B9B4B']),vmin=0,vmax=1,aspect='auto');a._heatmap_units='Component';a._heatmap_binary=True
 for i in range(len(allkeys)):
  for j in range(7):a.text(j,i,'+' if matrix[i,j] else '–',ha='center',va='center',fontsize=6)
 a.set_yticks(range(len(allkeys)),[labels[k] for k in allkeys],fontsize=5.6)
 a.set_xticks(range(7),fields,fontsize=7);a.tick_params(length=0);title(a,'a','Component inclusion')
 a=axs[1]
 for i,(key,values) in enumerate(paired_difference(keys,'proxy_rmse_pp')):
  a.hlines(i,values.min(),values.max(),color=colors[key],lw=1);a.scatter(values.mean(),i,c=colors[key],s=16,ec='white',lw=.3)
 a.axvline(0,c='#7A8873',ls='--',lw=.7);a.set_yticks(range(len(keys)),[labels[k] for k in keys],fontsize=5.7);a.invert_yaxis()
 a.set_xlabel('Δ proxy RMSE (pp); lower is better');title(a,'b','Paired reconstruction changes')
 a=axs[2]
 for i,(key,values) in enumerate(paired_difference(keys,'regulatory_coverage_pct',scope='full')):
  a.hlines(i,values.min(),values.max(),color=colors[key],lw=1);a.scatter(values.mean(),i,c=colors[key],s=16,ec='white',lw=.3)
 a.axvline(0,c='#7A8873',ls='--',lw=.7);a.set_yticks(range(len(keys)),[labels[k] for k in keys],fontsize=5.7);a.invert_yaxis()
 a.set_xlabel('Δ regulatory coverage (pp)');title(a,'c','Full-map screening changes')
 mapkeys=['No_coordinate','No_graph','Multi_relation']
 lim=max(abs((mean(k,'full')-mean('PGCAN','full'))*100).max() for k in mapkeys)
 for a,k,l in zip(axs[3:6],mapkeys,'def'):
  a.set_position([.055+mapkeys.index(k)*.335,.602,.265,.17]);a._map_units='Difference (pp)'
  delta_map(a,k,lim=lim);title(a,l,labels[k]+' − PGCAN')
 cax=axs[4].inset_axes([.04,-.25,.92,.07]);cb=f.colorbar(v.mpl.cm.ScalarMappable(norm=SymLogNorm(linthresh=lim/100,vmin=-lim,vmax=lim,base=10),cmap=v.DIFF),cax=cax,orientation='horizontal')
 cb.set_ticks([-lim,0,lim]);cb.set_ticklabels([f'{-lim:.3g}','0',f'{lim:.3g}'])
 cb.outline.set_visible(False);cb.ax.tick_params(labelsize=6.8,length=1.5);cb.set_label('Full-map difference (pp); shared symmetric-log scale',fontsize=6,labelpad=1)
 a=axs[6];hist=json.loads((E/'training_histories.json').read_text())
 for k in ['PGCAN','No_graph','Multi_relation']:
  q=pd.DataFrame(hist[f'{k}_20260918_fold1']['selection'])
  a.plot(q.epoch,q.train_ce,c=colors[k],lw=.9,ls='--')
  a.plot(q.epoch,q.validation_ce,c=colors[k],lw=1.1,label=labels[k])
 a.set(xlabel='Selection epoch',ylabel='Conditional cross entropy');a.legend(fontsize=5.5);title(a,'g','Inner spatial learning curves')
 a=axs[7]
 k=['PGCAN','No_coordinate','No_graph','Multi_relation']
 data=[fold[fold.method.eq(z)].epochs.to_numpy() for z in k]
 vp=a.violinplot(data,showextrema=False,widths=.7)
 for body,col in zip(vp['bodies'],[colors[z] for z in k]):body.set_facecolor(col);body.set_alpha(.35)
 for i,z in enumerate(data,1):a.scatter(np.full(len(z),i)+np.linspace(-.12,.12,len(z)),z,s=7,c=colors[k[i-1]],lw=0)
 a.set_xticks(range(1,5),['PGCAN','No\ncoord.','No\ngraph','Multi-\nrel.'],fontsize=6.8);a.set_ylabel('Selected refit epochs');title(a,'h','Spatial early-stopping budget')
 a=axs[8];ref=fold[fold.method.eq('PGCAN')][['seed','fold','proxy_rmse_pp']]
 q=fold[fold.method.eq('No_graph')].merge(ref,on=['seed','fold'],suffixes=('','_full'))
 a.scatter(q.proxy_rmse_pp_full,q.proxy_rmse_pp,c=cm_values(q.fold),s=18,edgecolors='white',lw=.3)
 lim=max(q.proxy_rmse_pp_full.max(),q.proxy_rmse_pp.max())*1.05;a.plot([0,lim],[0,lim],c=v.LILAC,ls='--',lw=.7)
 a.set(xlabel='PGCAN proxy RMSE (pp)',ylabel='No-graph proxy RMSE (pp)',xlim=(0,lim),ylim=(0,lim));title(a,'i','Does removing the graph hurt?')
 a=axs[9];category_order=['industrial','mixed','low_activity','research','transport_built']
 for i,cat in enumerate(category_order):
  val=(load('Multi_relation','full')[:,cats==cat].sum(1)-load('PGCAN','full')[:,cats==cat].sum(1))*100
  colour=v.TEAL if val.mean()>=0 else v.LILAC
  a.hlines(i,val.min(),val.max(),color=colour,lw=1.3);a.scatter(val.mean(),i,c=colour,s=17,ec='white',lw=.3)
 a.axvline(0,c='#8B9885',lw=.6);a.set_yticks(range(5),[v.CAT[c][1] for c in category_order],fontsize=5.8);a.invert_yaxis();a.set_xlabel('Multi-relation − PGCAN (pp)');title(a,'j','Category mass redistribution')
 a=axs[10];keys2=['PGCAN','No_coordinate','No_graph','Multi_relation'];repeatability=[]
 for i,k in enumerate(keys2):
  selections=np.array([top(p) for p in load(k,'full')]);freq=selections.sum(0)
  counts=[int((freq==j).sum()) for j in [1,2,3]];repeatability.append(counts)
 # A raw-count matrix exposes rare disagreements hidden by dominant bars.
 a.imshow(np.asarray(repeatability),cmap=v.HEAT,norm=v.mpl.colors.PowerNorm(.3,0,max(map(max,repeatability))),aspect='auto')
 a._heatmap_units='Cells'
 for i,row in enumerate(repeatability):
  for j,val in enumerate(row):a.text(j,i,str(val),ha='center',va='center',fontsize=7)
 a.set_yticks(range(4),[labels[k] for k in keys2],fontsize=6)
 a.set_xticks(range(3),['1','2','3']);a.tick_params(length=0);a.set_xlabel('Number of seeds selecting a cell');title(a,'k','Top-198 seed repeatability')
 a=axs[11]
 for k in allkeys:
  q=fold[fold.method.eq(k)];a.scatter(q.elapsed_seconds.mean(),q.proxy_rmse_pp.mean(),c=colors[k],s=24,ec='white',lw=.4)
 a.legend(handles=[v.Line2D([],[],marker='o',ls='',ms=3,color=colors[k],label=labels[k]) for k in ['PGCAN','No_graph','No_coordinate','Multi_relation']],fontsize=5.2,loc='upper right',handletextpad=.2)
 a.set(xlabel='Mean selection + refit time (s)',ylabel='Mean proxy RMSE (pp)');title(a,'l','Complexity–error trade-off')
 # Three additional panels connect full-map diagnostics to the held-out results.
 a=axs[12]
 for k in ['PGCAN','No_coordinate','No_graph','Multi_relation']:
  curve(a,abs(mean(k)-target)*100,labels[k],colors[k])
 a.set_xscale('symlog',linthresh=1e-5);a.set(xlabel='OOF absolute share error (pp)',ylabel='Cumulative cells');title(a,'m','Error-distribution controls')
 a=axs[13];coverage(a,['PGCAN','No_graph','No_coordinate','Multi_relation'],scope='full');title(a,'n','Refitted-map site screening')
 a=axs[14]
 pos=a.get_position();a.set_position([pos.x0,.055,pos.width,.155])
 for i,k in enumerate(allkeys):
  data=maps[(maps.method.eq(k))&(maps.scope.eq('full_domain_refit'))]
  a.scatter(data.tv_prior*100,np.full(len(data),i),s=10,c=colors[k],alpha=.8)
 a.set_yticks(range(len(allkeys)),[labels[k] for k in allkeys],fontsize=5.6);a.invert_yaxis();a.set_xlabel('Mass reallocated from prior (TV, %)');title(a,'o','How strongly each model adjusts')
 footer(f,'Ablations are refitted with matched folds/seeds. Paired intervals show actual ranges, not confidence intervals.\nLearning curves: fold 1, seed 20260918; dashed = training, solid = inner validation. Added graph complexity is a hypothesis under test.')
 v.save(f,'Figure_11',['Benchmark_fold_metrics.csv','Benchmark_map_metrics.csv','Benchmark_architecture_counts.csv'],
  'Fifteen-panel controlled ablation. Proxy errors and full-domain location screening are distinct. Delta intervals use matched fold/seed or matched full-refit seeds. Map differences share a symmetric-log pp scale with linear core equal to one hundredth of the maximum absolute difference. Training and inner-validation curves show a specified fold and seed. Registered parameter counts include disabled branches; ablations test computation, not environmental causal effects.')

def cm_values(folds):return ListedColormap(['#527B36','#8DB75F','#BCD699','#DFA7C6','#AE236F'])(np.asarray(folds,dtype=int)-1)
def top(p):
 out=np.zeros(len(p),bool);out[np.lexsort((cells.grid_id.to_numpy(str),-p))[:198]]=True;return out

def graph_plate():
 f,axs=axes_grid('From geographic distance to explicit graph relations','One grid-node type · three observable relation definitions · adaptive relation gates · topology controls')
 graphs=[np.load(E/f'relation_{i}.npz') for i in range(3)]
 for a,g,l,col,label in zip(axs[:3],graphs,'abc',RELC,REL):
  a.set_position([.055+'abc'.index(l)*.335,.805,.265,.175])
  v.mapplot(a);r=g['row'];c=g['col'];rng=np.random.default_rng(20261020)
  sel=rng.choice(len(r),size=min(260,len(r)),replace=False)
  lengths=np.linalg.norm(xy[r]-xy[c],axis=1)/1000
  norm,cm,ticks,description=v.mc.classify(lengths,v.SEQ)
  collection=LineCollection(np.stack([xy[r[sel]],xy[c[sel]]],axis=1),cmap=cm,norm=norm,linewidths=.45,alpha=.8,zorder=3)
  collection.set_array(lengths[sel]);a.add_collection(collection)
  v.mc.add_bar(f,a,norm,cm,ticks,'Edge length (km)',description)
  title(a,l,label)
  f._caption_notes=getattr(f,'_caption_notes',[])+[f'Panel {l}: {len(r):,} stored directed edges; 260 displayed.']
 a=axs[3];jac=pd.read_csv(S/'Benchmark_relation_jaccard.csv',index_col=0).to_numpy()
 a.imshow(jac,cmap=v.SEQ,vmin=0,vmax=1);a._heatmap_units='Jaccard'
 for i in range(3):
  for j in range(3):a.text(j,i,f'{jac[i,j]:.3f}',ha='center',va='center',fontsize=7,color='white' if jac[i,j]>.6 else v.INK)
 a.set_xticks(range(3),['Space','Function','Built'],fontsize=6.8);a.set_yticks(range(3),['Space','Function','Built'],fontsize=6);a.tick_params(length=0);title(a,'d','Relation-edge Jaccard overlap')
 a=axs[4]
 for g,col,label in zip(graphs,RELC,REL):curve(a,np.linalg.norm(xy[g['row']]-xy[g['col']],axis=1)/1000,label,col)
 a.set(xlabel='Geographic edge length (km)',ylabel='Cumulative directed edges');a.legend(fontsize=5.5,loc='lower right',handlelength=1);title(a,'e','How far each relation reaches')
 a=axs[5];audit=pd.read_csv(S/'Benchmark_relation_audit.csv')
 a.bar(range(3),audit.same_function_fraction*100,color=RELC,width=.65)
 for i,val in enumerate(audit.same_function_fraction*100):a.text(i,val+2,f'{val:.1f}',ha='center',fontsize=6.5)
 a.set_xticks(range(3),['Space','Function','Built'],rotation=35,ha='right',fontsize=6);a.set(ylim=(0,108),ylabel='Same functional category (%)');title(a,'f','Functional homophily audit')
 for i,(a,l,label) in enumerate(zip(axs[6:9],'ghi',REL)):
  a.set_position([.055+i*.335,.415,.265,.175]);a._map_units='Learned gate'
  v.mapplot(a,G[:,i],Normalize(0,1),v.SEQ);title(a,l,label+': gate')
 cax=axs[7].inset_axes([.04,-.25,.92,.07]);cb=f.colorbar(v.mpl.cm.ScalarMappable(norm=Normalize(0,1),cmap=v.SEQ),cax=cax,orientation='horizontal',ticks=[0,.5,1]);cb.outline.set_visible(False)
 cb.ax.tick_params(labelsize=6,length=1.5);cb.set_label('Mean learned gate across three full-domain refits',fontsize=6,labelpad=1)
 a=axs[9];order=['industrial','mixed','low_activity','research','transport_built'];val=np.array([G[cats==k].mean(0) for k in order])
 a.imshow(val,cmap=v.SEQ,vmin=0,vmax=1,aspect='auto');a._heatmap_units='Mean gate'
 for i in range(5):
  for j in range(3):a.text(j,i,f'{val[i,j]:.2f}',ha='center',va='center',fontsize=6.6,color='white' if val[i,j]>.65 else v.INK)
 a.set_yticks(range(5),[v.CAT[k][1] for k in order],fontsize=5.8);a.set_xticks(range(3),['Space','Function','Built'],fontsize=6);a.tick_params(length=0);title(a,'j','Gate allocation by category')
 a=axs[10];entropy=-(G*np.log(np.maximum(G,1e-12))).sum(1)/np.log(3)
 a.set_position([.39,.225,.265,.165]);a._map_units='Gate entropy'
 v.mapplot(a,entropy,Normalize(0,1),v.SEQ);title(a,'k','Normalized gate entropy [0, 1]')
 ecbar=a.inset_axes([1.025,.12,.045,.76]);cb=f.colorbar(v.mpl.cm.ScalarMappable(norm=Normalize(0,1),cmap=v.SEQ),cax=ecbar,ticks=[0,.5,1]);cb.outline.set_visible(False);cb.ax.tick_params(labelsize=5.2,length=1.5)
 a=axs[11];a.set_position([.725,.225,.265,.165]);cm=ListedColormap(RELC);v.mapplot(a,G.argmax(1),BoundaryNorm(np.arange(-.5,3),3),cm)
 title(a,'l','Largest learned relation gate')
 a.legend(handles=[Patch(fc=c,label=l) for c,l in zip(RELC,['Space','Function','Built'])],loc='upper center',bbox_to_anchor=(.5,-.20),ncol=3,fontsize=6.8,handlelength=.6,columnspacing=.5,borderaxespad=0)
 a=axs[12];keys=['Multi_relation','MR_no_spatial','MR_no_function','MR_no_built','MR_fixed_gate','MR_rewired','PGCAN']
 forest(a,'proxy_rmse_pp',keys,log=False);a.set_xlabel('Held-out proxy RMSE (pp)');title(a,'m','Relation and topology controls')
 a=axs[13]
 for i in range(3):
  data=gates[:,:,i].reshape(-1);q=np.quantile(data,[.05,.25,.5,.75,.95])
  a.vlines(i,q[0],q[4],color=RELC[i],lw=1);a.vlines(i,q[1],q[3],color=RELC[i],lw=5);a.scatter(i,q[2],s=15,c='white',ec=RELC[i],lw=.8,zorder=3)
 a.set_xticks(range(3),['Space','Function','Built'],fontsize=6);a.set(xlim=(-.6,2.6),ylim=(0,1),ylabel='Node/seed gate distribution');title(a,'n','Gate dispersion: 5–95% range')
 a=axs[14]
 for i,k in enumerate(keys):
  q=maps[(maps.method.eq(k))&(maps.scope.eq('full_domain_refit'))]
  a.hlines(i,q.regulatory_coverage_pct.min(),q.regulatory_coverage_pct.max(),color=colors[k],lw=1)
  a.scatter(q.regulatory_coverage_pct.mean(),i,c=colors[k],s=16,ec='white',lw=.3)
 a.set_yticks(range(len(keys)),[labels[k] for k in keys],fontsize=5.8);a.invert_yaxis();a.set_xlabel('Regulatory coverage: refit maps (%)');title(a,'o','Do semantic edges aid screening?')
 footer(f,'Edges in a–c: fixed random sample of 260 per relation; all edges enter audits. Relations use covariates and geometry, never benchmark labels.\nGates describe fitted computation, not causal importance or measured energy flows. Rewiring preserves each query’s edge count and weight multiset.')
 v.save(f,'Figure_12',['Benchmark_relation_audit.csv','Benchmark_relation_jaccard.csv','Benchmark_fold_metrics.csv','Benchmark_map_metrics.csv'],
  'Fifteen-panel multi-relation graph audit. Up to eight neighbours are selected from the nearest 65 feature-space candidates within 10 km, with spatial fallback when none qualifies; the spatial relation uses distance 8NN. All training message sources are training nodes. Final full-domain graph structures and mean gates are described separately from held-out performance. This is a homogeneous grid-node multi-relation graph, not a building-street heterogeneous graph or an observed industrial-flow network.')
 pd.DataFrame({'grid_id':cells.grid_id,'spatial_gate':G[:,0],'function_gate':G[:,1],'built_gate':G[:,2],
  'normalized_gate_entropy':entropy,'largest_gate':G.argmax(1)}).to_csv(S/'Benchmark_relation_gates.csv',index=False)

def architecture_plate():
 """The architecture is an editable table; no repeated text-box sketch."""
 from architecture_table import export_table
 return export_table()

if __name__=='__main__':
 old=json.loads((P/'figure_manifest.json').read_text(encoding='utf-8'))
 for fn in [benchmark_plate,ablation_plate,graph_plate,architecture_plate]:fn()
 new={a['figure']:a for a in v.manifest}
 result=[new.pop(a['figure'],a) for a in old]+list(new.values())
 (P/'figure_manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
 print('EXPERIMENT_FIGURES_COMPLETE',len(result),flush=True)
