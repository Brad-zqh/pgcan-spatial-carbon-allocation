"""Actual OOF error structure: common-scale maps, densities and calibration."""
import os
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm,LogNorm
import redraw_figures as v

METHODS=[('Ridge_direct','Ridge'),('Random_forest_direct','Random forest'),
    ('PGCAN','PGCAN refit'),('Multi_relation','Multi-relation')]

def draw():
    cells=pd.read_csv(v.SD/'Benchmark_cell_manifest.csv')
    assert cells.grid_id.tolist()==v.grid.grid_id.tolist()
    target=cells.target_share.to_numpy()*100
    predictions={key:np.load(v.HERE/'experiments'/f'oof_{key}.npy').mean(0)*100 for key,_ in METHODS}
    residuals={key:pred-target for key,pred in predictions.items()}
    limit=np.ceil(max(np.abs(a).max() for a in residuals.values())*1000)/1000
    mapnorm=SymLogNorm(linthresh=.001,linscale=1,vmin=-limit,vmax=limit,base=10)
    height=250;f=plt.figure(figsize=(183/25.4,height/25.4))
    def axes(x,y,w,h):return f.add_axes([x/183,y/height,w/183,h/height])
    def label(ax,letter,description):
        ax.text(-.035,1.05,letter,transform=ax.transAxes,fontsize=9.5,fontweight='bold',va='bottom')
        f._panel_descriptions=getattr(f,'_panel_descriptions',{})
        f._panel_descriptions[letter]=description
    transform=lambda x:np.arcsinh(x/.01)
    rtransform=lambda x:np.arcsinh(x/.001)
    upper=max(target.max(),max(a.max() for a in predictions.values()))*1.04
    tx=transform(target);rlim=float(rtransform(limit));xlim=float(transform(upper))
    bins=pd.qcut(pd.Series(target).rank(method='first'),10,labels=False).to_numpy()
    point_records=[];calibration=[];hexes=[];hex_axes=[]
    for j,(key,name) in enumerate(METHODS):
        x=15+j*42
        a=axes(15+(j%2)*84,198-(j//2)*50,75,36);a._map_units='Residual (pp)'
        v.mapplot(a,residuals[key],mapnorm,v.DIFF)
        label(a,string.ascii_lowercase[j],name+': spatially held-out residual map')
        b=axes(x,87,35,37)
        b.set_title(name,fontsize=8.5,fontweight='bold',pad=7).set_gid('comparison_method_header')
        hb=b.hexbin(tx,rtransform(residuals[key]),gridsize=50,
            extent=(0,xlim,-rlim,rlim),mincnt=1,cmap=v.DENSITY,
            norm=LogNorm(1,200),linewidths=0,rasterized=True)
        assert int(hb.get_array().sum())==len(target)
        hexes.append(hb);hex_axes.append(b)
        b.axhline(0,c='black',lw=.6,ls='--')
        ticks=np.array([0,.01,.1,.8]);ticks=ticks[ticks<=upper]
        rticks=np.array([-.1,-.01,0,.01,.1]);rticks=rticks[np.abs(rticks)<=limit]
        b.set(xlim=(0,xlim),ylim=(-rlim,rlim),xticks=transform(ticks),yticks=rtransform(rticks))
        b.set_xticklabels([f'{z:g}' for z in ticks],fontsize=7)
        b.set_yticklabels([f'{z:g}' for z in rticks],fontsize=7)
        if j>0:b.tick_params(labelleft=False)
        b.set_xlabel('Target share (%)',fontsize=7.5)
        if j==0:b.set_ylabel('Residual (pp)',fontsize=7.5)
        label(b,string.ascii_lowercase[4+j],name+': residual versus constructed target')
        c=axes(x,28,35,34)
        rows=[]
        for k in range(10):
            mask=bins==k;pred=predictions[key][mask]
            row={'method':key,'target_decile':k+1,'n_cells':int(mask.sum()),
                'mean_target_pct':float(target[mask].mean()),'mean_prediction_pct':float(pred.mean()),
                'prediction_q25_pct':float(np.quantile(pred,.25)),
                'prediction_q75_pct':float(np.quantile(pred,.75)),
                'mean_residual_pp':float(residuals[key][mask].mean())}
            rows.append(row);calibration.append(row)
        q=pd.DataFrame(rows)
        c.fill_between(q.mean_target_pct,q.prediction_q25_pct,q.prediction_q75_pct,
            color=v.TEAL,alpha=.24,lw=0)
        c.plot(q.mean_target_pct,q.mean_prediction_pct,color=v.LILAC,lw=1.1,
            marker='o',ms=3,mec='white',mew=.35)
        c.plot([0,upper],[0,upper],c='black',ls='--',lw=.65)
        c.set_xscale('symlog',linthresh=.005);c.set_yscale('symlog',linthresh=.005)
        c.set(xlim=(0,upper),ylim=(0,upper),xticks=ticks,yticks=ticks)
        c.set_xticklabels([f'{z:g}' for z in ticks],fontsize=7)
        c.set_yticklabels([f'{z:g}' for z in ticks],fontsize=7)
        c.set_xlabel('Mean target (%)',fontsize=7.5)
        if j==0:c.set_ylabel('Mean prediction (%)',fontsize=7.5)
        label(c,string.ascii_lowercase[8+j],name+': target-decile calibration and within-decile IQR')
        point_records.append(pd.DataFrame({'method':key,'grid_id':cells.grid_id,'fold':cells.fold,
            'target_share_pct':target,'mean_oof_prediction_pct':predictions[key],
            'residual_pp':residuals[key],'target_decile':bins+1}))
    cax=f.add_axes([.28,132/height,.50,1.9/height])
    cb=f.colorbar(v.mpl.cm.ScalarMappable(norm=mapnorm,cmap=v.DIFF),cax=cax,orientation='horizontal',
        ticks=[-limit,-.01,0,.01,limit])
    cb.outline.set_visible(False);cb.ax.tick_params(labelsize=7,length=2,pad=1)
    cb.set_label('Prediction − constructed target (pp; common symmetric-log scale)',fontsize=7.4,labelpad=2)
    maximum=max(float(h.get_array().max()) for h in hexes)
    for h in hexes:h.set_norm(LogNorm(1,float(h.get_array().max())))
    for a,h in zip(hex_axes,hexes):
        localmax=float(h.get_array().max())
        cb=f.colorbar(h,cax=a.inset_axes([1.03,.08,.035,.84]),ticks=[1,localmax])
        cb.set_ticklabels(['1',f'{localmax:g}'])
        cb.outline.set_visible(False);cb.ax.minorticks_off();cb.ax.tick_params(labelsize=6.2,length=1,pad=1)
        for text in cb.ax.get_yticklabels():text.set_gid('map_colourbar_label')
    pd.concat(point_records,ignore_index=True).to_csv(v.SD/'Analysis_oof_error_cells.csv',index=False)
    pd.DataFrame(calibration).to_csv(v.SD/'Analysis_oof_decile_calibration.csv',index=False)
    v.save(f,'Figure_S8',['Benchmark_cell_manifest.csv','Analysis_oof_error_cells.csv',
        'Analysis_oof_decile_calibration.csv'],
        'Spatial error structure for ridge, random forest, PGCAN refits and the multi-relation candidate, using complete matched out-of-fold predictions. '
        'Maps a–d identify ridge, random forest, PGCAN and the multi-relation candidate in that order. The residual-density and calibration columns follow the same method order. Each residual map has its own labelled empirical-quantile colourbar in percentage points, with zero preserved as a neutral sign boundary. '
        'Panels e–h show all 1,980 cells per method in common bins: horizontal positions use asinh(target percentage / 0.01), vertical positions use asinh(residual pp / 0.001), and tick labels show original units. '
        'Green denotes sparse outer bins and pink denotes dense cores; colour represents cell counts, not the error sign. Each density panel has a labelled local count scale; repeated y tick labels are suppressed after the first column because all density panels share the same axes. '
        'Panels i–l show ten equally sized target-rank groups (198 cells each), with target ties broken in the fixed cell order. Pink points and lines connect group mean predictions; green bands span the 25th to 75th percentiles of predictions within each group, not confidence intervals. '
        'Calibration axes use symmetric-log scaling with a 0.005% linear threshold. Dashed diagonals denote agreement. '
        'Stochastic models use the mean of three seed-specific OOF predictions; deterministic ridge uses its single saved prediction. The five spatial folds, training-mass calibration and constructed targets are unchanged. These are descriptive proxy reconstruction diagnostics, not independently measured emissions errors.')
    print('ERROR_DIAGNOSTICS_READY: 12 panels; 7,920 cell-method records; 40 calibration groups.',flush=True)

if __name__=='__main__':draw()
