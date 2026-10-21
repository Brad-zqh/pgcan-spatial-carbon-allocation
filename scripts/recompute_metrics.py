"""Recompute Figure 13 from every released real cell, including zeros."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
ROOT=Path(__file__).resolve().parents[1]
def main():
    folder=ROOT/'data/results';target=pd.read_csv(folder/'cell_targets.csv').target_share.to_numpy()*100
    expected=pd.read_csv(folder/'Benchmark_prediction_density_metrics.csv').set_index('method');rows=[]
    for method,row in expected.iterrows():
        pred=np.load(folder/f'oof_{method}.npy').mean(0)*100
        assert len(pred)==len(target)==1980 and np.isfinite(pred).all()
        values={'pooled_oof_r2':r2_score(target,pred),'pooled_oof_rmse_pp':np.sqrt(mean_squared_error(target,pred)),'pooled_oof_mae_pp':mean_absolute_error(target,pred)}
        for field,value in values.items():np.testing.assert_allclose(value,row[field],rtol=1e-10,atol=1e-12)
        rows.append({'method':method,**values})
    print(json.dumps({'status':'PASS','all_cells_retained':1980,'models_checked':len(rows),'metrics':rows},indent=2))
if __name__=='__main__':main()
