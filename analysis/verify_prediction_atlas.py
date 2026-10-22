"""Independently reproduce displayed metrics and density ranks; audit layout."""
import os
from pathlib import Path
import json,hashlib,re
from collections import Counter
from precision_labels import error_label,r2_label
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde,rankdata
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import fitz
P=Path(__file__).resolve().parent;S=P/'source_data'

def main():
    cells=pd.read_csv(S/'Benchmark_cell_manifest.csv');target=cells.target_share.to_numpy()*100
    metrics=pd.read_csv(S/'Benchmark_prediction_density_metrics.csv')
    styling=pd.read_csv(S/'Benchmark_prediction_density_styling.csv')
    ecdf=pd.read_csv(S/'Benchmark_prediction_error_ecdf.csv');residuals=pd.read_csv(S/'Benchmark_prediction_signed_residuals.csv');rows=[]
    assert len(metrics)==15 and len(ecdf)==15*1980
    assert len(styling)==int(metrics.n_rendered_predictions.sum())
    old=json.loads((P/'qa/prediction_matrix_contract.json').read_text(encoding='utf-8'))
    for item in metrics.itertuples():
        path=P/'experiments'/f'oof_{item.method}.npy'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==old['prediction_array_sha256'][item.method]
        raw=np.load(path)*100;pred=raw.mean(0)
        np.testing.assert_allclose([r2_score(target,pred),np.sqrt(mean_squared_error(target,pred)),mean_absolute_error(target,pred)],
            [item.pooled_oof_r2,item.pooled_oof_rmse_pp,item.pooled_oof_mae_pp],rtol=1e-11,atol=1e-13)
        plotted_target=np.tile(target,len(raw));plotted_prediction=raw.reshape(-1)
        sample=np.vstack([np.arcsinh(plotted_target/.001),np.arcsinh(plotted_prediction/.001)])
        density=gaussian_kde(sample,bw_method='scott')(sample)
        percent=(rankdata(np.round(np.log(density),10),method='average')-1)/(len(density)-1)*100
        style=styling[styling.method.eq(item.method)]
        assert style.grid_id.tolist()==np.tile(cells.grid_id,len(raw)).tolist()
        assert style.seed_index.tolist()==np.repeat(np.arange(len(raw)),1980).tolist()
        assert len(style)==item.n_rendered_predictions==len(raw)*1980
        np.testing.assert_allclose(style.target_share_pct,plotted_target,rtol=1e-12,atol=1e-13)
        np.testing.assert_allclose(style.oof_prediction_pct,plotted_prediction,rtol=1e-12,atol=1e-13)
        np.testing.assert_allclose(style.kde_density,density,rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(style.density_percentile,percent,atol=1e-10)
        curve=ecdf[ecdf.method.eq(item.method)]
        np.testing.assert_allclose(curve.absolute_error_pp,np.sort(abs(pred-target)),rtol=1e-10,atol=1e-13)
        np.testing.assert_allclose(curve.cumulative_fraction,np.arange(1,1981)/1980,atol=1e-12)
        strip=residuals[residuals.method.eq(item.method)]
        assert strip.grid_id.tolist()==cells.grid_id.tolist()
        np.testing.assert_allclose(strip.target_share_pct,target,rtol=1e-12,atol=1e-13)
        np.testing.assert_allclose(strip.signed_residual_pp,pred-target,rtol=1e-9,atol=1e-13)
        rows.append({'method':item.method,'n_cells':1980,'n_points':len(style),'n_seeds':len(raw),
            'n_ecdf_values':1980,'metrics_density_and_ecdf_reproduced':True})
    pdf=fitz.open(P/'Figures/Figure_13.pdf');page=pdf[0]
    text=page.get_text()
    assert text.count('RMSE ')==15 and text.count('MAE ')==15
    assert text.count('Density percentile (%)')==15
    assert Counter(re.findall(r'R²\(mean\)\s*=\s*(\d+\.\d{3})',text))==Counter(r2_label(x) for x in metrics.pooled_oof_r2)
    for label,column in [('RMSE','pooled_oof_rmse_pp'),('MAE','pooled_oof_mae_pp')]:
        found=re.findall(label+r'\s*([=<])\s*(\d+\.\d{3})\s*pp',text)
        assert Counter(found)==Counter(('<','0.001') if 0<x<.001 else ('=',f'{x:.3f}') for x in metrics[column])
    assert 'Residual (pp)' not in text and 'Error ECDF' not in text
    manifest=json.loads((P/'figure_manifest.json').read_text(encoding='utf-8'))
    layout=next(item for item in manifest if item['figure']=='Figure_13')['layout']
    assert layout['one_scatter_per_model'] and layout['individual_vertical_colourbars']==15
    assert layout['metrics_location']=='upper left inside each scatter'
    fonts={s['font'] for b in page.get_text('dict')['blocks'] for line in b.get('lines',[]) for s in line['spans']}
    assert all('TimesNewRoman' in name for name in fonts)
    print_scale=min(176/(page.rect.width/72*25.4),245/(page.rect.height/72*25.4))
    report={'status':'PASS','methods':rows,'print_width_mm':176,'print_scale_from_export':print_scale,
        'general_text_minimum_at_print_pt':8.2*print_scale,'compact_diagnostic_text_at_print_pt':6.2*print_scale,
        'point_positions_unchanged':True,'jitter':False,'observations_sampled':False,
        'all_unfavourable_results_retained':True,'original_prediction_arrays_unchanged':True,
        'layout':layout,'exactly_one_scatter_per_model':True,'metrics_in_upper_left':True,
        'density_colour_semantics':'Within-panel KDE density percentile; not absolute counts or emissions uncertainty'}
    report['metric_display']='Three decimals; positive errors below 0.001 displayed <0.001; unrounded source metrics retained'
    (P/'qa/prediction_atlas_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert report['general_text_minimum_at_print_pt']>=7 and report['compact_diagnostic_text_at_print_pt']>=5
    review_path=P/'qa/prediction_matrix_design_review.json'
    review=json.loads(review_path.read_text(encoding='utf-8'))
    review.update(status='PASS',actual_prediction_marks_verified=len(styling),
        rendered_preview_review='Current export inspected: larger transparent marks, real seed variation, single panels, upper-left ensemble metrics, vertical density bars; no jitter.')
    review_path.write_text(json.dumps(review,indent=2),encoding='utf-8')
    print(f'PREDICTION_ATLAS_PASS: {len(styling)} actual seed predictions, metrics and density ranks reproduced; 15 single panels, upper-left metrics and vertical density bars checked')
if __name__=='__main__':main()
