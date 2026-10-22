"""Held-out model atlas: one true scatter, upper-left metrics and vertical bar.

Explicit user requirements select black Times New Roman and pink/green. All
1,980 real cells are drawn per model; no jitter, sampled points or fitted bands.
"""
import os
from pathlib import Path
import json,string,os
_cache=Path(__file__).resolve().parent/'cache'
_cache.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(_cache)
os.environ['TEMP']=os.environ['TMP']=str(_cache)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator,FixedFormatter
from scipy.stats import gaussian_kde,rankdata
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import redraw_figures as v

P=v.HERE;E=P/'experiments';S=v.SD
METHODS=[('PGCAN','PGCAN refit'),('Multi_relation','Multi-relation'),('Ridge_direct','Ridge'),
 ('Local_ridge_direct','Local ridge'),('Random_forest_direct','Random forest'),('Extra_trees_direct','ExtraTrees'),
 ('Hist_gradient_direct','Hist. boosting'),('XGBoost_direct','XGBoost'),('MLP_direct','MLP'),
 ('No_coordinate','No coordinate'),('No_context_branch','No context'),('No_graph','No graph'),
 ('No_function_input','No function'),('MR_fixed_gate','Uniform gates'),('MR_rewired','Rewired edges')]


def draw():
    cells=pd.read_csv(S/'Benchmark_cell_manifest.csv');target=cells.target_share.to_numpy()*100
    assert len(target)==1980 and np.isfinite(target).all() and (target>=0).all()
    old_metrics=pd.read_csv(S/'Benchmark_prediction_density_metrics.csv').set_index('method')
    predictions={key:np.load(E/f'oof_{key}.npy')*100 for key,_ in METHODS}
    for values in predictions.values():
        assert values.ndim==2 and values.shape[1]==1980 and values.shape[0] in (1,3)
        assert np.isfinite(values).all() and (values>=0).all()
    means={key:values.mean(0) for key,values in predictions.items()}
    transform=lambda value:np.arcsinh(value/.01)
    upper=max(target.max(),max(a.max() for a in means.values()))*1.045
    maximum=float(transform(upper));xt=transform(target)
    width,height=183,259
    fig=plt.figure(figsize=(width/25.4,height/25.4));fig._font_scale=1.
    def axes(x,y,w,h):return fig.add_axes([x/width,y/height,w/width,h/height])
    ticks=np.array([0,.01,.05,.2,.8]);ticks=ticks[ticks<=upper]
    metrics=[];points=[];styling=[];ecdf_records=[];residual_records=[];kde_checks=[]
    for index,(key,label) in enumerate(METHODS):
        row,col=divmod(index,3);left=12+58*col;bottom=208-48*row
        ax=axes(left,bottom,42,42)
        prediction=means[key];yp=transform(prediction);sample=np.vstack([xt,yp])
        # KDE changes colour only. Locations remain actual target/prediction
        # pairs, including zero targets. Covariance is not inflated or jittered.
        kde=gaussian_kde(sample,bw_method='scott');density=kde(sample)
        # Numerically identical densities must receive the same colour despite
        # platform-dependent summation at machine precision.
        percentile=(rankdata(np.round(np.log(density),10),method='average')-1)/(len(density)-1)*100
        assert np.isfinite(density).all() and (density>0).all()
        order=np.argsort(density,kind='stable')
        artist=ax.scatter(xt[order],yp[order],c=percentile[order],cmap=v.DENSITY,
            norm=Normalize(0,100),s=.95,lw=0,alpha=.80,rasterized=True)
        ax.plot([0,maximum],[0,maximum],c='#535353',lw=.55,ls=(0,(3,2)),zorder=0)
        ax.set(xlim=(0,maximum),ylim=(0,maximum),aspect='equal')
        for axis in [ax.xaxis,ax.yaxis]:
            axis.set_major_locator(FixedLocator(transform(ticks)))
            axis.set_major_formatter(FixedFormatter([f'{value:g}' for value in ticks]))
        ax.tick_params(length=2,pad=1.8,labelsize=8.2,labelleft=col==0,labelbottom=row==4)
        ax.spines[['left','bottom']].set_color('#666666');ax.spines[['left','bottom']].set_linewidth(.45)
        ax.text(0,1.025,string.ascii_lowercase[index],transform=ax.transAxes,
            fontsize=10.5,fontweight='bold',va='bottom')
        ax.text(.105,1.025,label,transform=ax.transAxes,fontsize=8.8,fontweight='bold',va='bottom')
        rmse=float(np.sqrt(mean_squared_error(target,prediction)))
        mae=float(mean_absolute_error(target,prediction));r2=float(r2_score(target,prediction))
        actual=[r2,rmse,mae]
        np.testing.assert_allclose(actual,old_metrics.loc[key,['pooled_oof_r2','pooled_oof_rmse_pp','pooled_oof_mae_pp']].to_numpy(float),rtol=1e-12,atol=1e-14)
        r2_display=f'{r2:.6f}'.rstrip('0').rstrip('.') if r2>.9999 else f'{r2:.4f}'
        ax.text(.025,.975,f'R² = {r2_display}\nRMSE = {rmse:.3g} pp\nMAE = {mae:.3g} pp',
            transform=ax.transAxes,fontsize=8.2,va='top',ha='left',linespacing=1.12)
        cax=axes(left+44,bottom+3,1.25,36)
        cb=fig.colorbar(artist,cax=cax,orientation='vertical',ticks=[])
        cb.outline.set_visible(False)
        for xx,yy,txt,rotation,va in [(.5,-.025,'0',0,'top'),(.5,1.025,'100',0,'bottom'),(2.6,.5,'Density percentile (%)',90,'center')]:
            text=cb.ax.text(xx,yy,txt,transform=cb.ax.transAxes,ha='center',va=va,rotation=rotation,fontsize=6.2)
            text.set_gid('compact_diagnostic_label')
        residual=np.abs(prediction-target);sorted_error=np.sort(residual)
        fraction=np.arange(1,len(target)+1)/len(target)
        signed_error=prediction-target
        residual_records.append(pd.DataFrame({'method':key,'grid_id':cells.grid_id,
            'target_share_pct':target,'signed_residual_pp':signed_error}))
        assert len(artist.get_offsets())==len(target)
        fig._panel_descriptions=getattr(fig,'_panel_descriptions',{})
        fig._panel_descriptions[string.ascii_lowercase[index]]=label
        metrics.append({'panel':string.ascii_lowercase[index],'method':key,'display_name':label,'n_cells':1980,
            'n_training_seeds':len(predictions[key]),'pooled_oof_r2':r2,'pooled_oof_rmse_pp':rmse,
            'pooled_oof_mae_pp':mae,'zero_target_cells':int((target==0).sum()),'excluded_cells':0})
        points.append(pd.DataFrame({'method':key,'grid_id':cells.grid_id,'fold':cells.fold,
            'target_share_pct':target,'mean_oof_prediction_pct':prediction}))
        styling.append(pd.DataFrame({'method':key,'grid_id':cells.grid_id,'kde_density':density,'density_percentile':percentile}))
        ecdf_records.append(pd.DataFrame({'method':key,'order':np.arange(len(target)),
            'absolute_error_pp':sorted_error,'cumulative_fraction':fraction}))
        kde_checks.append({'method':key,'n_plotted':len(artist.get_offsets()),'kde_bandwidth_rule':'Scott',
            'bandwidth_factor':float(kde.factor),'density_min':float(density.min()),'density_max':float(density.max()),
            'jitter':False,'sampled':False})
    backdrop=axes(0,0,width,height);backdrop.axis('off');backdrop.set_zorder(-10)
    backdrop.text(.006,.526,'Held-out prediction (%)',rotation=90,ha='left',va='center',fontsize=9)
    backdrop.text(.48,.025,'Constructed target share (%)',ha='center',va='center',fontsize=9)
    fig._layout={'canvas_mm':[width,height],'panel_count':15,'primary_models_in_first_row':['PGCAN','Multi_relation','Ridge_direct'],
        'scatter_side_mm':42,'individual_vertical_colourbars':15,'signed_residual_strips':0,
        'metrics_location':'upper left inside each scatter','one_scatter_per_model':True,
        'repeated_inner_tick_labels_suppressed':True,'no_global_title':True}
    fig._density_encoding={'type':'KDE percentile','unit':'within-panel density percentile (%)','bandwidth':'Scott',
        'point_positions':'Unmodified asinh(share_pct/0.01)','all_cells_retained':1980,'colourbar_ticks':[0,100],
        'numeric_ties':'Natural-log density rounded to 10 decimals before average-rank assignment'}
    pd.DataFrame(metrics).to_csv(S/'Benchmark_prediction_density_metrics.csv',index=False)
    pd.concat(points,ignore_index=True).to_csv(S/'Benchmark_prediction_density_points.csv',index=False)
    pd.concat(styling,ignore_index=True).to_csv(S/'Benchmark_prediction_density_styling.csv',index=False)
    pd.concat(ecdf_records,ignore_index=True).to_csv(S/'Benchmark_prediction_error_ecdf.csv',index=False)
    pd.concat(residual_records,ignore_index=True).to_csv(S/'Benchmark_prediction_signed_residuals.csv',index=False)
    caption=('Spatially held-out predictions for fifteen model configurations. '
        'The first row contrasts PGCAN, the multi-relation candidate and ridge; rows two and three contain remaining classical baselines, and rows four and five contain six graph/branch controls. '
        'Every scatter shows all 1,980 actual cells, including six zero targets; no points are sampled, jittered or moved. Stochastic methods use the mean of three matched-seed out-of-fold predictions and deterministic methods one run. '
        'Scatter positions use identical axes with asinh(share in percent/0.01); tick labels show original percentage shares. The dashed diagonal denotes equality. '
        'Colour is the within-panel percentile of a bivariate Gaussian kernel-density estimate in display coordinates, with Scott bandwidth: green sparse exterior to pink dense core. Each panel has a vertical 0–100 density-percentile colourbar. These are density ranks, not cell counts, signed errors or comparable absolute density magnitudes. '
        'The upper-left corner of each scatter reports R², RMSE and MAE from untransformed percentage shares; errors are percentage points (pp). Values above 0.9999 use six decimal places to distinguish near-exact from perfect reconstruction. Each model has exactly one scatter panel; no duplicate residual plot is included. '
        'Five spatial folds use 10 km blocks and a 1 km centroid buffer. Held-out shares retain training-mass calibration and are not renormalized against test targets. '
        'Targets are deterministic constructed proxies rather than measured emissions. All metrics and unfavourable comparisons are retained unchanged; these panels quantify proxy reconstruction only.')
    v.save(fig,'Figure_13',['Benchmark_cell_manifest.csv','Benchmark_prediction_density_metrics.csv',
        'Benchmark_prediction_density_points.csv','Benchmark_prediction_density_styling.csv','Benchmark_prediction_signed_residuals.csv'],caption)
    (P/'qa/prediction_matrix_design_review.json').write_text(json.dumps({'status':'RENDERED_FOR_REVIEW',
        'layout':v.manifest[-1]['layout'],'density_encoding':v.manifest[-1]['density_encoding'],
        'kde_checks':kde_checks,'all_numeric_metrics_unchanged':True},indent=2),encoding='utf-8')
    old=json.loads((P/'figure_manifest.json').read_text(encoding='utf-8'));new={item['figure']:item for item in v.manifest}
    current=[new.pop(item['figure'],item) for item in old if item['figure']!='Figure_14']+list(new.values())
    (P/'figure_manifest.json').write_text(json.dumps(current,indent=2,ensure_ascii=False),encoding='utf-8')
    print('OOF_MODEL_ATLAS_READY: 15 single point clouds, upper-left metrics, 15 vertical density bars; metrics unchanged',flush=True)

if __name__=='__main__':draw()
