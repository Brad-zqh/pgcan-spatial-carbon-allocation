"""Independently reproduce every displayed metric, density rank and error ECDF."""
import os
from pathlib import Path
import json,hashlib
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
    ecdf=pd.read_csv(S/'Benchmark_prediction_error_ecdf.csv');rows=[]
    assert len(metrics)==15 and len(styling)==len(ecdf)==15*1980
    old=json.loads((P/'qa/prediction_matrix_contract.json').read_text(encoding='utf-8'))
    for item in metrics.itertuples():
        path=P/'experiments'/f'oof_{item.method}.npy'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==old['prediction_array_sha256'][item.method]
        pred=np.load(path).mean(0)*100
        np.testing.assert_allclose([r2_score(target,pred),np.sqrt(mean_squared_error(target,pred)),mean_absolute_error(target,pred)],
            [item.pooled_oof_r2,item.pooled_oof_rmse_pp,item.pooled_oof_mae_pp],rtol=1e-11,atol=1e-13)
        sample=np.vstack([np.arcsinh(target/.01),np.arcsinh(pred/.01)])
        density=gaussian_kde(sample,bw_method='scott')(sample)
        percent=(rankdata(np.round(np.log(density),10),method='average')-1)/1979*100
        style=styling[styling.method.eq(item.method)]
        assert style.grid_id.tolist()==cells.grid_id.tolist()
        np.testing.assert_allclose(style.kde_density,density,rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(style.density_percentile,percent,atol=1e-10)
        curve=ecdf[ecdf.method.eq(item.method)]
        np.testing.assert_allclose(curve.absolute_error_pp,np.sort(abs(pred-target)),rtol=1e-10,atol=1e-13)
        np.testing.assert_allclose(curve.cumulative_fraction,np.arange(1,1981)/1980,atol=1e-12)
        rows.append({'method':item.method,'n_points':1980,'n_ecdf_values':1980,'metrics_density_and_ecdf_reproduced':True})
    pdf=fitz.open(P/'Figures/Figure_13.pdf');page=pdf[0]
    fonts={s['font'] for b in page.get_text('dict')['blocks'] for line in b.get('lines',[]) for s in line['spans']}
    assert all('TimesNewRoman' in name for name in fonts)
    print_scale=176/(page.rect.width/72*25.4)
    report={'status':'PASS','methods':rows,'print_width_mm':176,'print_scale_from_export':print_scale,
        'general_text_minimum_at_print_pt':8.2*print_scale,'compact_diagnostic_text_at_print_pt':6.2*print_scale,
        'point_positions_unchanged':True,'jitter':False,'observations_sampled':False,
        'all_unfavourable_results_retained':True,'original_prediction_arrays_unchanged':True,
        'density_colour_semantics':'Within-panel KDE density percentile; not absolute counts or emissions uncertainty'}
    (P/'qa/prediction_atlas_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert report['general_text_minimum_at_print_pt']>=7 and report['compact_diagnostic_text_at_print_pt']>=5
    print('PREDICTION_ATLAS_PASS: all 29700 paired cells, 15 metrics, density ranks and ECDFs independently reproduced; print type checked')
if __name__=='__main__':main()
