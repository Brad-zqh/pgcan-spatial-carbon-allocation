"""Build all report tables and prose from the executed experimental outputs."""
import os
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from PIL import Image
P=Path(__file__).resolve().parent;S=P/'source_data';E=P/'experiments';T=P/'Tables';T.mkdir(exist_ok=True)
protocol=json.loads((E/'protocol.json').read_text());assert protocol['completed']
fold=pd.concat([pd.read_csv(S/'Benchmark_fold_metrics.csv'),pd.read_csv(S/'Benchmark_direct_fold_metrics.csv')],ignore_index=True)
mapmetrics=pd.concat([pd.read_csv(S/'Benchmark_map_metrics.csv'),pd.read_csv(S/'Benchmark_direct_map_metrics.csv')],ignore_index=True)
fold.to_csv(S/'Benchmark_all_fold_metrics.csv',index=False);mapmetrics.to_csv(S/'Benchmark_all_map_metrics.csv',index=False)
fold.groupby('method').agg(rmse_pp_mean=('proxy_rmse_pp','mean'),rmse_pp_sd=('proxy_rmse_pp','std'),
 mae_pp_mean=('proxy_mae_pp','mean'),mae_pp_sd=('proxy_mae_pp','std'),r2_mean=('proxy_r2','mean'),r2_sd=('proxy_r2','std'),
 fold_records=('fold','size')).to_csv(S/'Benchmark_all_summary.csv')
labels={'Ridge_direct':'Ridge (direct shares)','Local_ridge_direct':'Local ridge (direct shares)',
 'Random_forest_direct':'Random forest (direct shares)','Extra_trees_direct':'ExtraTrees (direct shares)',
 'Hist_gradient_direct':'Histogram boosting (direct shares)','XGBoost_direct':'XGBoost (direct shares)','MLP_direct':'MLP (direct shares)',
 'PGCAN':'PGCAN spatial refit','Multi_relation':'Multi-relation candidate','No_coordinate':'Without coordinate branch',
 'No_context_branch':'Without context branch','No_function_input':'Without functional one-hot',
 'No_graph_input':'Without graph input','No_graph_penalty':'Without graph penalty','No_graph':'Without graph components',
 'No_prior_anchor':'Without prior output anchor','No_prior_penalty':'Without prior penalty',
 'MR_no_spatial':'MR: without spatial paths','MR_no_function':'MR: without function relation','MR_no_built':'MR: without built relation',
 'MR_fixed_gate':'MR: uniform relation gates','MR_rewired':'MR: random rewiring',
 'Clean_prior':'Clean prior (analytic)','Direct_mixture':'Direct mixture (target formula)'}
def label(k):return labels.get(k,k.replace('_',' '))
def fmt(x,d=4):return f'{x:.{d}g}'
def ms(frame,col,d=4):
 return fmt(frame[col].mean(),d)+' ± '+fmt(frame[col].std(ddof=1),d)
tables=[]
def table(num,title,headers,rows,note,widths):
 path=T/f'Table_{num}.csv';pd.DataFrame(rows,columns=headers).to_csv(path,index=False,encoding='utf-8-sig')
 tables.append({'id':num,'title':title,'headers':headers,'rows':rows,'note':note,'widths_mm':widths,'file':str(path)})

table(1,'What is transferred from the three reference studies',
 ['Study','Actual task / evidence','Design adapted here','Boundary of adaptation'],[
 ['Feng et al., 2026; GTCNNWR','Uneven spatiotemporal samples; fixed spatial/temporal reference frames; CNN-weighted regression.','Train/validation/test separation; spatial-support strata; explicit network widths and benchmark tables.','Current carbon data are one spatial snapshot; no GTCNNWR reproduction or temporal prediction is claimed.'],
 ['Zhang et al., 2026; SNSTFM','LST fusion; spatial non-stationary regression, residual compensation and cross-scale correction; multiple evaluation settings.','Module-by-module residual tests; matched error metrics, maps and distributions.','No coarse/fine time-paired LST or carbon observations exist here; fusion accuracy is not transferable.'],
 ['Yap et al., 2025; building operating carbon','Four urban node types, six edge types; two-layer heterogeneous GraphSAGE; building-energy supervision.','Explicit relation schemas; topology audit; feature/module controls and relation-level diagnostics.','Implemented candidate has one grid-node type and three covariate-derived relations; it is not their heterogeneous city graph.']
 ],'References inform experimental design. Their observations, performance values and causal interpretations are not copied.',[30,47,47,50])
counts=pd.read_csv(S/'Benchmark_architecture_counts.csv').set_index('method')
from layerwise_architecture import specification
layer_headers,layer_rows,layer_widths=specification()
table(2,'Layer-wise architectures and hyperparameters',layer_headers,layer_rows,
 'N denotes the current training-node count. All fully connected parameter counts include bias. Superscript 1 denotes modules used only by the multi-relation candidate; its context MLP replaces the PGCAN context MLP. The coordinate branch is shared in design, not shared across independently fitted models. There are no convolution, batch-normalization or dropout layers in this implementation. Allocation softmax runs across cells. Epoch selection uses buffered inner spatial validation every 15 epochs, patience five checks and a minimum of 120 epochs. Registered counts retain disabled branches in ablation models.',layer_widths)


audit=pd.read_csv(S/'Benchmark_relation_audit.csv')
table(3,'Graph relation definitions and full-domain topology audit',
 ['Relation','Observed definition','Edges','Median / p90 length (km)','Same-category edges'],[
 ['Spatial proximity','Eight closest training centroids; Gaussian weights.',str(int(audit.iloc[0].directed_edges)),f'{audit.iloc[0].median_distance_m/1000:.2f} / {audit.iloc[0].p90_distance_m/1000:.2f}',f'{audit.iloc[0].same_function_fraction*100:.2f}%'],
 ['Functional similarity','Standardized enterprise, industrial, patent and land-cover variables plus category one-hot.',str(int(audit.iloc[1].directed_edges)),f'{audit.iloc[1].median_distance_m/1000:.2f} / {audit.iloc[1].p90_distance_m/1000:.2f}',f'{audit.iloc[1].same_function_fraction*100:.2f}%'],
 ['Built/activity similarity','Standardized NTL, floor-area proxy and road density.',str(int(audit.iloc[2].directed_edges)),f'{audit.iloc[2].median_distance_m/1000:.2f} / {audit.iloc[2].p90_distance_m/1000:.2f}',f'{audit.iloc[2].same_function_fraction*100:.2f}%']
 ],'Similarity edges: select up to eight from 65 nearest feature-space candidates within 10 km; use spatial fallback only when none qualifies. Tied similarities follow deterministic cKDTree ordering. Edge counts describe stored candidate pairs: 63 functional-relation weights round to zero in float32 (15,475 positive-weight pairs). All receiving-node weight sums remain one. Training nodes receive only training-node messages. These are similarity relations, not observed energy or supply-chain flows.',[28,67,16,35,28])
main=['Clean_prior','Direct_mixture','Ridge_direct','Local_ridge_direct','Random_forest_direct','Extra_trees_direct','Hist_gradient_direct','XGBoost_direct','MLP_direct','PGCAN','Multi_relation']
table(4,'Main comparison: spatially held-out proxy reconstruction',
 ['Method','RMSE (pp)','MAE (pp)','R²','Fold records'],[
 [label(k),ms(fold[fold.method.eq(k)],'proxy_rmse_pp'),ms(fold[fold.method.eq(k)],'proxy_mae_pp'),ms(fold[fold.method.eq(k)],'proxy_r2'),str(int(fold.method.eq(k).sum()))] for k in main
 ],'Values are means ± sample SD across recorded fold/seed combinations. Folds and repeated seeds are dependent, so SD is descriptive. Direct mixture is the target formula, not an independently learned competitor. Near-perfect ridge reconstruction is expected because the target is itself a linear mixture of prior and activity covariates.',[43,38,38,38,17])
keys=['No_coordinate','No_context_branch','No_function_input','No_graph_input','No_graph_penalty','No_graph','No_prior_anchor','No_prior_penalty']
def differences(k,baseline='PGCAN'):
 a=fold[fold.method.eq(k)].merge(fold[fold.method.eq(baseline)][['fold','seed','proxy_rmse_pp']],on=['fold','seed'],suffixes=('','_reference'))
 return a.proxy_rmse_pp-a.proxy_rmse_pp_reference
table(5,'Controlled refit ablations of the original components',
 ['Variant','Proxy RMSE (pp)','Paired ΔRMSE mean','Paired ΔRMSE range'],[
 [label(k),ms(fold[fold.method.eq(k)],'proxy_rmse_pp'),fmt(differences(k).mean()),f'[{fmt(differences(k).min())}, {fmt(differences(k).max())}]'] for k in keys
 ],'Each difference is variant minus PGCAN on the same outer fold and training seed. Negative values favour removing the component. Removing the graph means removing both its input aggregation and its smoothness penalty. Removing a branch is different from removing one context input.',[55,40,35,44])
relkeys=['Multi_relation','MR_no_spatial','MR_no_function','MR_no_built','MR_fixed_gate','MR_rewired','PGCAN']
table(6,'Relation-level and topology controls',
 ['Variant','Proxy RMSE (pp)','Paired ΔRMSE vs MR','Meaning of intervention'],[
 [label(k),ms(fold[fold.method.eq(k)],'proxy_rmse_pp'),fmt(differences(k,'Multi_relation').mean()),meaning]
 for k,meaning in zip(relkeys,['All three relations and adaptive gate.','Remove spatial relation, spatial AX input and graph smoothness.','Remove only the functional-similarity relation.','Remove only built/activity-similarity relation.','Equal gate weights; retain separate relation encoders.','Random training-node sources; preserve each receiving row’s edge count and weight multiset.','Single spatial-relation refit without added relation encoders.'])
 ],'Rewiring does not preserve each source node’s indegree. Learned gates are computational weights, not causal relation importance. No convincing semantic-topology advantage can be inferred merely from a smaller average than the original model.',[43,36,31,64])
rows=[]
for k in main+['No_graph','MR_rewired']:
 p=np.load(E/f'full_{k}.npy').mean(0);ids=np.lexsort((cellsid:=pd.read_csv(S/'Benchmark_cell_manifest.csv').grid_id.to_numpy(str),-p))[:198]
 grid=pd.read_csv(S/'allocation_500m.csv');rc=grid.primary_count.to_numpy()
 es=pd.read_csv(S/'energy_benchmark_sites.csv');ec=np.bincount(pd.Index(grid.grid_id).get_indexer(es.grid_id),minlength=len(grid))
 rows.append([label(k),f'{int(rc[ids].sum())}/146',f'{rc[ids].sum()/146*100:.2f}',f'{int(ec[ids].sum())}/69',f'{p[ids].sum()*100:.2f}'])
original=np.load(S/'probabilities_Full.npy').mean(0);ids=np.lexsort((cellsid,-original))[:198]
rows.append(['PGCAN original saved mean',f'{int(rc[ids].sum())}/146',f'{rc[ids].sum()/146*100:.2f}',f'{int(ec[ids].sum())}/69',f'{original[ids].sum()*100:.2f}'])
table(7,'Site screening on fixed mean full-domain maps',
 ['Method','Regulatory hits','Coverage (%)','Energy hits','Top-10% mass (%)'],rows,
 'Select exactly 198 of 1,980 cells with grid-ID tie breaking. Values describe the mean allocation map, not the mean of per-seed coverage. Lists share 59 sites and POI geolocation. They validate location screening rather than measured emissions.',[54,30,25,25,40])
rows=[]
for family in ['Ridge','Local_ridge','Random_forest','Extra_trees','Hist_gradient','XGBoost','MLP']:
 a=fold[fold.method.eq(family)];b=fold[fold.method.eq(family+'_direct')]
 rows.append([label(family),ms(a,'proxy_rmse_pp'),ms(b,'proxy_rmse_pp'),ms(a,'proxy_r2'),ms(b,'proxy_r2')])
table(8,'Prediction-head sensitivity: all tested baseline results',
 ['Family','Log-ratio RMSE (pp)','Direct-share RMSE (pp)','Log-ratio R²','Direct-share R²'],rows,
 'Failed log-ratio extrapolations are retained. Log-ratio scores are clipped to ±20 before train-mass calibration; direct-share outputs are clamped nonnegative. All direct-share models explicitly receive prior share. The two heads are reported separately; test performance is not used to select a head per fold.',[30,37,37,35,35])
split=json.loads((E/'split_manifest.json').read_text())
table(9,'Spatial separation and inner-validation accounting',
 ['Fold','Outer train','Outer test','Inner fit','Inner validation','Minimum centroid gap (m)'],[
 [str(s['fold']),str(s['n_train']),str(s['n_test']),str(s['inner_train']),str(s['inner_validation']),f'{s["min_train_test_centroid_m"]:.1f}'] for s in split
 ],'Blocks are 10 km; training centroids closer than 1 km to test centroids are excluded. This is a centroid buffer, not a 1 km polygon-edge gap. Standardizers fit inner training for epoch selection, then outer training for refitting. No test node is a training message source.',[14,26,26,26,32,50])
table(10,'Fixed baseline hyperparameters and comparison budget',
 ['Model','Declared configuration'],[
 ['Ridge','alpha = 1; standardized continuous covariates.'],
 ['Local ridge','128-neighbour Gaussian geographic kernel; ridge = 0.1; intercept penalty = 1e-6; no bandwidth search.'],
 ['Random forest / ExtraTrees','220 trees; minimum leaf size 3; max features 0.8; one CPU worker.'],
 ['Histogram boosting','220 iterations; maximum 15 leaves; L2 = 1; no random early-stopping split.'],
 ['XGBoost','250 trees; depth 3; learning rate 0.04; subsample and column sample 0.85; lambda = 3.'],
 ['MLP','48, 24 hidden units; alpha = 0.01; max 700 iterations; no random early-stopping split.'],
 ['Neural allocation variants','Three seeds; five identical outer folds; one fixed inner spatial holdout per outer fold; maximum 450 selection epochs plus selected-epoch refit.'],
 ['Scope','Fixed-configuration exploratory benchmark; no exhaustive or equal-compute hyperparameter optimization and no claim of globally optimal model performance.']
 ],'The same outer splits and seed IDs are reused. Timing is measured during the run and can include contention; it is an operational observation, not a hardware-normalized speed ranking.',[42,132])

rmse=fold.groupby('method').proxy_rmse_pp.mean()
ratio=float((rmse.PGCAN-rmse.No_graph)/rmse.PGCAN*100)
keyresults={k:float(rmse[k]) for k in ['Ridge_direct','PGCAN','No_coordinate','No_graph','Multi_relation','MR_rewired']}
summary=[
 f'Executed {fold.method.nunique()} configurations and {len(fold)} outer-fold evaluations on 1,980 cells; 10 km blocks, 1 km centroid buffer and up to three seeds.',
 f'Direct-share ridge reconstructs the deterministic proxy almost exactly (mean fold RMSE {fmt(rmse.Ridge_direct)} pp). The target combines the declared prior with a linear activity score, so this does not demonstrate independently validated carbon prediction.',
 f'Removing graph input and graph smoothness lowers mean proxy RMSE from {fmt(rmse.PGCAN)} to {fmt(rmse.No_graph)} pp ({ratio:.1f}% lower). This comparison does not support a necessary benefit from the current spatial graph.',
 f'The multi-relation candidate has mean proxy RMSE {fmt(rmse.Multi_relation)} pp, compared with {fmt(rmse.MR_rewired)} pp after random rewiring. Their close averages provide no clear support for meaningful semantic-topology gain.',
 'The context-only variant outperforms the full coordinate-plus-context model on this proxy task. Functional one-hot inputs and the prior anchor can matter for proxy reconstruction, but their value for real emissions remains untested.',
 'The strongest next evidence would be independent facility energy/emission observations or genuinely held-out inventory units. Adding graph types alone cannot overcome the deterministic-supervision bottleneck.'
]
references=[
 {'title':'Feng et al. (2026). Geographically and temporally convolutional neural network weighted regression for modelling spatiotemporal non-stationarity on uneven data.','doi':'10.1080/13658816.2026.2696345'},
 {'title':'Zhang et al. (2026). SNSTFM: A novel data fusion approach based on spatial non-stationarity to generate fine-resolution land surface temperature.','doi':'10.1016/j.rse.2026.115528'},
 {'title':'Yap et al. (2025). Revealing building operating carbon dynamics for multiple cities. Nature Sustainability.','doi':'10.1038/s41893-025-01615-8'}]
manifest=json.loads((P/'figure_manifest.json').read_text(encoding='utf-8'))
corrections_path=P/'citation_audit/reviewer/correction_suggestions.json'
if corrections_path.exists():
 fixes=json.loads(corrections_path.read_text(encoding='utf-8'))
 for reference in references:
  match=next((item for item in fixes['reference_replacements'] if reference['doi'] in item['new']),None)
  if match:reference['title']=match['new'].split('] ',1)[1].split(' https://doi.org/')[0]
figures=[]
for name in ['Figure_10','Figure_11','Figure_12','Figure_13','Figure_S8']:
 entry=next((e for e in manifest if e['figure']==name),None)
 if entry:
  with Image.open(P/'Figures'/f'{name}.png') as im:size=list(im.size)
  figures.append({'name':name,'caption':entry['caption'],'path':str(P/'Figures'/f'{name}.png'),'size':size})
material={'title':'Spatial model comparison, graph ablations and architecture audit','date':'21 October 2026',
 'subtitle':'Executed experiments on constructed carbon-allocation supervision; results, complete controls and evidence limits',
 'summary':summary,'key_results':keyresults,'tables':tables,'figures':figures,'references':references,
 'protocol':protocol,'n_configurations':int(fold.method.nunique()),'n_fold_records':len(fold)}
(E/'report_material.json').write_text(json.dumps(material,indent=2,ensure_ascii=False),encoding='utf-8')
(P/'EXPERIMENT_RESULTS_CN.md').write_text('# 本轮实际实验结果\n\n'
 f'已完成 {fold.method.nunique()} 个模型/变体、{len(fold)} 组空间折次评估；五折、10 km 分块、1 km 质心缓冲，随机模型三个训练种子。\n\n'
 f'直接份额岭回归的构造目标 RMSE 为 {rmse.Ridge_direct:.7f} pp；PGCAN 空间重拟合为 {rmse.PGCAN:.7f} pp；去掉图的模型为 {rmse.No_graph:.7f} pp；多关系图为 {rmse.Multi_relation:.7f} pp；随机重连为 {rmse.MR_rewired:.7f} pp。以上均非实测排放误差。\n\n'
 '结果不支持把“增加空间图”作为已验证的核心创新。复杂图暂未优于简单对照，构造目标又能被线性模型近乎精确复现，现阶段瓶颈是独立监督和验证证据。全部失败或不利结果保留在表格中。\n\n'
 '图10、11、12各15个面板。模型架构、维度、损失与参数量通过正文可编辑表4展示，原重复框图移除。图13为15个模型及消融配置的留出预测密度组图，全部1980个网格均保留，统一坐标范围和密度色带，标注真实计算的R²、RMSE、MAE。其指标为三种子平均留出预测的合并评分，与逐折逐种子平均分口径不同。新增图S8为12面板误差诊断：四个模型的残差地图、残差密度和目标分位校准；校准带是组内预测值IQR，不是置信区间。10张表同时提供CSV、Word、PDF。指北针和比例尺不设白底，主体与字体加大；全文Times New Roman、黑色字体。\n',encoding='utf-8')
print('REPORT_MATERIAL_READY',len(tables),len(figures),flush=True)
