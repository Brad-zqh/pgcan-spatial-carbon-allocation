"""Matched held-out prediction density panels, computed from saved OOF arrays."""
import os
from pathlib import Path
import json,string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,LogNorm
from matplotlib.ticker import FixedLocator,FixedFormatter
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import redraw_figures as v
from visual_polish import density_norm

P=v.HERE;E=P/'experiments';S=v.SD
METHODS=[('Ridge_direct','Ridge'),('Local_ridge_direct','Local ridge'),
 ('Random_forest_direct','Random forest'),('Extra_trees_direct','ExtraTrees'),
 ('Hist_gradient_direct','Histogram boosting'),('XGBoost_direct','XGBoost'),
 ('MLP_direct','MLP'),('PGCAN','PGCAN refit'),('Multi_relation','Multi-relation'),
 ('No_coordinate','No coordinate branch'),('No_context_branch','No context branch'),
 ('No_graph','No graph components'),('No_function_input','No functional input'),
 ('MR_fixed_gate','Uniform relation gates'),('MR_rewired','Rewired relations')]

def draw():
    cells=pd.read_csv(S/'Benchmark_cell_manifest.csv')
    target=cells.target_share.to_numpy()*100
    assert len(target)==1980 and np.isfinite(target).all() and (target>=0).all()
    scale=.01
    def transform(a):return np.arcsinh(a/scale)
    predictions={k:np.load(E/f'oof_{k}.npy')*100 for k,_ in METHODS}
    for a in predictions.values():
        assert a.ndim==2 and a.shape[1]==len(target) and a.shape[0] in (1,3)
        assert np.isfinite(a).all() and (a>=0).all()
    means={k:a.mean(axis=0) for k,a in predictions.items()}
    upper=max(target.max(),max(a.max() for a in means.values()))*1.045
    extent=(0,float(transform(upper)),0,float(transform(upper)))
    # The requested green-to-pink ordering encodes positive bin counts only:
    # outer low-density bins are green; the dense core is pink.
    density=v.DENSITY
    f,axes=plt.subplots(5,3,figsize=(183/25.4,248/25.4))
    f.subplots_adjust(left=.105,right=.93,bottom=.065,top=.982,wspace=.42,hspace=.24)
    artists=[];records=[];points=[]
    xt=transform(target)
    ticks=np.array([0,.01,.05,.2,.8]);ticks=ticks[ticks<=upper]
    labels=[f'{q:g}' for q in ticks]
    for index,((key,label),ax) in enumerate(zip(METHODS,axes.flat)):
        pred=means[key];yp=transform(pred)
        artist=ax.hexbin(xt,yp,gridsize=65,extent=extent,mincnt=1,cmap=density,
            norm=LogNorm(vmin=1,vmax=200),linewidths=0,rasterized=True)
        artists.append(artist)
        assert int(artist.get_array().sum())==len(target), (key,'Hexbin lost cells')
        ax.plot([0,extent[1]],[0,extent[1]],color='#555555',lw=.65,ls=(0,(3,2)))
        ax.set(xlim=extent[:2],ylim=extent[2:],aspect='equal')
        for axis in [ax.xaxis,ax.yaxis]:
            axis.set_major_locator(FixedLocator(transform(ticks)))
            axis.set_major_formatter(FixedFormatter(labels))
        ax.tick_params(labelsize=6.0,pad=1.5,length=2)
        ax.text(-.12,1.025,string.ascii_lowercase[index],transform=ax.transAxes,
            fontsize=9,fontweight='bold',va='bottom')
        rmse=float(np.sqrt(mean_squared_error(target,pred)))
        mae=float(mean_absolute_error(target,pred));r2=float(r2_score(target,pred))
        ax.text(.035,.967,f'R² = {r2:.3f}\nRMSE = {rmse:.4g} pp\nMAE = {mae:.4g} pp',
            transform=ax.transAxes,va='top',fontsize=6.5,linespacing=1.25,
            bbox={'fc':'white','ec':'none','alpha':.85,'pad':.5})
        ax.text(.965,.035,label,transform=ax.transAxes,ha='right',va='bottom',
            fontsize=6.7,fontweight='bold',bbox={'fc':'white','ec':'none','alpha':.85,'pad':.5})
        if index%3==0:ax.set_ylabel('Held-out prediction (%)',fontsize=7,labelpad=3)
        if index//3==4:ax.set_xlabel('Constructed target share (%)',fontsize=7,labelpad=3)
        f._panel_descriptions=getattr(f,'_panel_descriptions',{})
        f._panel_descriptions[string.ascii_lowercase[index]]=label
        records.append({'panel':string.ascii_lowercase[index],'method':key,'display_name':label,
            'n_cells':len(target),'n_training_seeds':len(predictions[key]),'pooled_oof_r2':r2,
            'pooled_oof_rmse_pp':rmse,'pooled_oof_mae_pp':mae,
            'zero_target_cells':int((target==0).sum()),'excluded_cells':0})
        points.append(pd.DataFrame({'method':key,'grid_id':cells.grid_id,'fold':cells.fold,
            'target_share_pct':target,'mean_oof_prediction_pct':pred}))
    maximum=max(float(a.get_array().max()) for a in artists)
    for artist in artists:artist.set_norm(density_norm(artist.get_array()))
    for ax,artist in zip(axes.flat,artists):
        localmax=float(artist.get_array().max());ticks_cb=[1,2,10,localmax] if localmax>10 else [1,2,localmax]
        cb=f.colorbar(artist,cax=ax.inset_axes([1.035,.10,.035,.80]),ticks=ticks_cb)
        cb.set_ticklabels([f'{x:g}' for x in ticks_cb])
        cb.outline.set_visible(False);cb.ax.minorticks_off();cb.ax.tick_params(labelsize=6.2,length=1.5,pad=1)
        cb.set_label('Cells',fontsize=6.2,labelpad=1)
        for text in [*cb.ax.get_yticklabels(),cb.ax.yaxis.label]:text.set_gid('map_colourbar_label')
    pd.DataFrame(records).to_csv(S/'Benchmark_prediction_density_metrics.csv',index=False)
    pd.concat(points,ignore_index=True).to_csv(S/'Benchmark_prediction_density_points.csv',index=False)
    caption=('Spatially held-out predictions for fifteen learned model configurations. '
        'Each panel includes all 1,980 cells exactly once, including zero targets; stochastic models use the mean of three matched-seed out-of-fold predictions, whereas deterministic baselines have one seed. '
        'All panels use identical axes and hexagonal bins. Positions use the transform asinh(share in percent / 0.01); tick labels show the original share percentage. '
        'This monotone transform resolves both low allocations and the upper tail without omitting zero-valued cells. '
        'The dashed diagonal denotes equality. Green denotes singleton bins; count 2 enters pink, and higher counts darken through pink. Each panel has a separate two-segment logarithmic count colourbar with a knot at 2 (mapped to 72% of the colour ramp) and its actual maximum bin count; raw counts are labelled and colour magnitude is panel-specific. This expands the visible pink density range without changing observations or bins. Axes and bin locations are identical. The 65-column hexagonal grid makes the density marks smaller. Colour has no signed-error interpretation. '
        'R², RMSE and MAE are recomputed from untransformed percentage shares on the complete pooled out-of-fold ensemble; they differ from the mean of per-fold/per-seed scores in Table 5. '
        'Five spatial folds use 10 km blocks and a 1 km centroid exclusion buffer. Held-out shares retain training-mass calibration and are not renormalized using test targets. '
        'Targets are a deterministic constructed proxy, not observed emissions; differences therefore quantify proxy reconstruction only. '
        'Direct-share heads are shown for the classical baselines; other configurations retain their declared allocation heads. All results, including those unfavourable to graph complexity, are retained.')
    v.save(f,'Figure_13',['Benchmark_cell_manifest.csv','Benchmark_prediction_density_metrics.csv',
        'Benchmark_prediction_density_points.csv'],caption)
    old=json.loads((P/'figure_manifest.json').read_text(encoding='utf-8'))
    new={x['figure']:x for x in v.manifest}
    out=[new.pop(x['figure'],x) for x in old if x['figure']!='Figure_14']+list(new.values())
    (P/'figure_manifest.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    print('OOF_DENSITY_MATRIX_READY',len(records),'panels; common bin-count maximum',maximum,flush=True)

if __name__=='__main__':draw()
