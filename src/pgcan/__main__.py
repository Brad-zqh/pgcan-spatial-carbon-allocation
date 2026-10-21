"""Fit the original five-seed model on an authorized predictor matrix."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from . import core

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictors',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs/original'))
    parser.add_argument('--epochs',type=int,default=900)
    parser.add_argument('--seeds',type=int,nargs='+',default=core.SEEDS)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.epochs<1:parser.error('--epochs must be positive')
    core.OUT=args.output;core.EPOCHS=args.epochs
    d,c,p,u,t,xy=core.prepare(pd.read_csv(args.predictors))
    maps=[];logs=[]
    for seed in args.seeds:
        pred,log=core.train(d,c,p,t,xy,seed,save=f'model_{seed}')
        maps.append(pred);logs.append(log)
    np.save(args.output/'shares.npy',maps)
    pd.DataFrame({'grid_id':d.grid_id,'prior_share':p,'target_share':t,
                  'mean_model_share':np.mean(maps,axis=0),'adopted_carbon_t':core.carbon(np.mean(maps,axis=0))}).to_csv(args.output/'allocation.csv',index=False)
    (args.output/'run.json').write_text(json.dumps({'epochs':args.epochs,'seeds':args.seeds,'n_cells':len(d),'target':'constructed proxy, not observed emissions','logs':logs},indent=2),encoding='utf-8')
    print('Saved',len(maps),'seed fits to',args.output)
if __name__=='__main__':main()
