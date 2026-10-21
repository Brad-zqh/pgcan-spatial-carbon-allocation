"""Experiment: identify what fitted neural channels and heuristic assumptions change.

No new model is fitted. Reproduce all five saved checkpoints before interpreting
their outputs. The target remains constructed proxy supervision; benchmark lists
are used for conditional screening diagnostics only, with no emissions labels.
Run with the installed scientific environment containing torch.
"""
from pathlib import Path
import os,sys,json,hashlib,platform
sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent
os.environ['TMP']=os.environ['TEMP']=str(P/'cache')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import numpy as np
import pandas as pd
import torch
import importlib.util
from scipy.special import softmax

ROOT=Path(os.environ.get('PGCAN_PROJECT_ROOT',P.parent/'data/local/project'))
DATA=ROOT/'submission_revision_20260922'/'corrected'
OUT=P/'source_data';OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('saved_experiment',DATA/'experiment_core.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
SEEDS=list(range(20260918,20260923));TOP=198
frame=pd.read_csv(DATA/'predictors_only.csv');grid=pd.read_csv(DATA/'allocation_500m.csv')
d,lab,prior,activity,target,xy=m.prepare(frame)
assert np.array_equal(d.grid_id,grid.grid_id)
np.testing.assert_allclose(prior,grid.Clean_prior_share,rtol=1e-10,atol=1e-15)
np.testing.assert_allclose(target,grid.Direct_mixture_share,rtol=1e-10,atol=1e-15)
saved=np.load(DATA/'probabilities_Full.npy')
x=torch.tensor(m.standard(d[m.FEATURES]));coords=torch.tensor(m.standard(xy))
context=torch.tensor(np.eye(5,dtype='float32')[lab]);adj,*_=m.graph_for(xy,8)
gx=torch.sparse.mm(adj,x);priorlog=torch.tensor(np.log(np.maximum(prior,1e-12)),dtype=torch.float32)
site=pd.read_csv(DATA/'energy_benchmark_sites.csv');primary=pd.read_csv(DATA/'validation_primary_exact_sites.csv')
site['overlap']=site.id.astype(str).isin(set(primary.matched_poi_id.astype(str)))
assert site.overlap.sum()==59
ec=np.bincount(pd.Index(d.grid_id).get_indexer(site.grid_id),minlength=len(d))
unique=np.bincount(pd.Index(d.grid_id).get_indexer(site.loc[~site.overlap,'grid_id']),minlength=len(d))
rc=grid.primary_count.to_numpy(int)
def topmask(prob:np.ndarray)->np.ndarray:
    """Select a fixed number of cells with the original deterministic tie rule."""
    mask=np.zeros(len(prob),bool);mask[np.lexsort((d.grid_id.to_numpy(str),-prob))[:TOP]]=True
    return mask
def evaluate(prob:np.ndarray)->dict:
    """Fixed-map screening and composition; never grid-emissions accuracy."""
    mask=topmask(prob)
    return {'regulatory_coverage_pct':float(rc[mask].sum()/146*100),
            'energy_coverage_pct':float(ec[mask].sum()/69*100),
            'unique_energy_hits':int(unique[mask].sum()),'unique_energy_n':int(unique.sum()),
            'top10_mass_pct':float(prob[mask].sum()*100),
            'tv_prior':float(abs(prob-prior).sum()/2),'tv_target':float(abs(prob-target).sum()/2)}
def normalized_torch(logits:torch.Tensor)->np.ndarray:
    return m.normalize(torch.softmax(logits,0).numpy())

def main()->None:
    torch.set_num_threads(1)
    coord_phi=[];context_phi=[];base=[];reconstructed=[];residuals=[];rows=[];taurows=[];audit=[]
    for i,seed in enumerate(SEEDS):
        file=DATA/f'model_Full_{seed}.pt';model=m.Model();model.load_state_dict(torch.load(file,map_location='cpu',weights_only=True));model.eval()
        with torch.no_grad():
            beta=model.coord(coords)
            coordinate=(beta[:,:9]*x).sum(1)+beta[:,9]
            contextual=model.context(torch.cat([x,context,gx],1)).squeeze(1)
            r,_=model(x,coords,context,gx)
            torch.testing.assert_close(coordinate+contextual,r,rtol=0,atol=0)
            p00=normalized_torch(priorlog)
            p10=normalized_torch(priorlog+coordinate)
            p01=normalized_torch(priorlog+contextual)
            p11=normalized_torch(priorlog+r)
            maxerr=float(abs(p11-saved[i]).max())
            np.testing.assert_allclose(p11,saved[i],rtol=2e-5,atol=2e-9)
            assert np.array_equal(topmask(p11),topmask(saved[i]))
            # Exact two-channel Shapley decomposition of the fitted output.
            # This is a computational decomposition, not a causal attribution.
            phi_a=.5*((p10-p00)+(p11-p01))
            phi_b=.5*((p01-p00)+(p11-p10))
            np.testing.assert_allclose(phi_a+phi_b,p11-p00,atol=2e-16,rtol=0)
            assert abs(phi_a.sum())<1e-12 and abs(phi_b.sum())<1e-12
            gross=(abs(phi_a).sum()+abs(phi_b).sum())/2
            net=abs(p11-p00).sum()/2
            coord_phi.append(phi_a);context_phi.append(phi_b);base.append(p00);reconstructed.append(p11);residuals.append(r.numpy())
            rows.append({'seed':seed,'coordinate_tv':float(abs(phi_a).sum()/2),
                         'context_tv':float(abs(phi_b).sum()/2),'net_tv':float(net),
                         'cancellation_fraction':float(1-net/gross) if gross else 0,
                         'reproduction_max_abs_share_error':maxerr})
            for name,prob in [('Prior_clipped',p00),('Coordinate_only',p10),('Context_only',p01),('Full',p11)]:
                vals=evaluate(prob);vals.update(seed=seed,coalition=name)
                audit.append(vals)
            for tau in np.linspace(0,2,41):
                prob=normalized_torch(priorlog+float(tau)*r)
                vals=evaluate(prob);vals.update(seed=seed,residual_multiplier=float(tau))
                taurows.append(vals)
        print('CHECKPOINT_REPRODUCED',seed,'max_error',maxerr,flush=True)
    A=np.asarray(coord_phi);B=np.asarray(context_phi);BASE=np.asarray(base);FULL=np.asarray(reconstructed)
    np.savez_compressed(OUT/'Deep_channel_decomposition.npz',coordinate_phi=A,context_phi=B,baseline=BASE,reconstructed_full=FULL,residual=np.asarray(residuals))
    pd.DataFrame(rows).to_csv(OUT/'Deep_channel_summary.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'Deep_channel_coalitions.csv',index=False)
    pd.DataFrame(taurows).to_csv(OUT/'Deep_residual_response.csv',index=False)
    pd.DataFrame({'grid_id':d.grid_id,'coordinate_phi_pp':A.mean(0)*100,'context_phi_pp':B.mean(0)*100,
                  'total_change_pp':(FULL-BASE).mean(0)*100,'category':grid.function_class}).to_csv(OUT/'Deep_channel_cell_values.csv',index=False)
    cats=[]
    for seed,aa,bb in zip(SEEDS,A,B):
        for cat in m.CLASSES:
            mask=grid.function_class.eq(cat).to_numpy()
            cats.append({'seed':seed,'category':cat,'coordinate_change_pp':float(aa[mask].sum()*100),
                         'context_change_pp':float(bb[mask].sum()*100),'total_change_pp':float((aa[mask]+bb[mask]).sum()*100)})
    pd.DataFrame(cats).to_csv(OUT/'Deep_channel_category_changes.csv',index=False)
    # Reconstruct all 256 original prior-design draws from the saved parameters.
    sc=pd.read_csv(DATA/'prior_scenarios_256.csv');fields=['ntl_sum','enterprise_poi_density','industrial_poi_density','floor_area_proxy','landuse_class1_ratio','lowcarbon_patent_count']
    mm=np.stack([m.minmax(d[c]) for c in fields]);Priors=[];design_rows=[]
    for _,record in sc.iterrows():
        w=np.array([record[f'w{i}'] for i in range(6)]);fac=np.array([record[f'threshold_factor{i}'] for i in range(6)])
        label=np.select([d.industrial_poi_count>=5*fac[0],(d.road_density_km_per_km2>=8*fac[1])&(d.building_density>=.25*fac[2]),
                         (d.lowcarbon_patent_count>=5*fac[3])|(d.innovation_enterprise_count>=2*fac[4]),
                         (d.landuse_class3_ratio>=.55*fac[5])&(d.enterprise_poi_count==0),
                         (d.enterprise_poi_count>0)|(d.ntl_mean>0)],[0,1,2,4,3],default=4)
        prob=m.normalize((w[:,None]*mm).sum(0)*np.array([1.25,1.05,.9,.7,.15])[label])
        met=evaluate(prob)
        assert np.isclose(met['regulatory_coverage_pct'],record.primary_coverage*100)
        assert np.isclose(met['top10_mass_pct'],record.top10_mass_share*100)
        Priors.append(prob);design_rows.append({'scenario':int(record.scenario),**met})
    Priors=np.asarray(Priors);np.save(OUT/'Deep_prior_design_maps.npy',Priors)
    pd.DataFrame(design_rows).to_csv(OUT/'Deep_prior_design_metrics.csv',index=False)
    freq=np.mean([topmask(q) for q in Priors],axis=0)
    seed_sd=saved.std(0,ddof=1)*100;design_sd=Priors.std(0,ddof=1)*100
    design_iqr=(np.quantile(Priors,.75,axis=0)-np.quantile(Priors,.25,axis=0))*100
    adjustment=abs(grid.Full_share.to_numpy()-prior)*100
    # Predeclared 80th-percentile cutoffs: descriptive data-acquisition flags.
    # They are not optimized against site hits and are not risk probabilities.
    xcut=float(np.quantile(adjustment,.8));ycut=float(np.quantile(design_iqr,.8))
    flag=(adjustment>=xcut).astype(int)+2*(design_iqr>=ycut).astype(int)
    cell=pd.DataFrame({'grid_id':d.grid_id,'seed_sd_pp':seed_sd,'prior_design_sd_pp':design_sd,
                       'prior_design_iqr_pp':design_iqr,'neural_adjustment_abs_pp':adjustment,
                       'prior_scenario_top198_frequency':freq,'diagnostic_group':flag,
                       'category':grid.function_class})
    cell.to_csv(OUT/'Deep_structural_sensitivity_cells.csv',index=False)
    categoryrows=[]
    for cat in m.CLASSES:
        mask=grid.function_class.eq(cat).to_numpy()
        for source,values in [('Prior_design',Priors),('Neural_seeds',saved)]:
            for draw,val in enumerate(values[:,mask].sum(1)*100):categoryrows.append({'category':cat,'source':source,'draw':draw,'mass_pct':float(val)})
    pd.DataFrame(categoryrows).to_csv(OUT/'Deep_category_design_vs_seed.csv',index=False)
    shares=pd.read_csv(DATA/'comparison_shares.csv')
    for k in ['Full','Clean_prior','Direct_mixture']:np.testing.assert_allclose(shares[k],grid[k+'_share'],atol=1e-12,rtol=1e-9)
    paths=[]
    for k in ['Clean_prior','Direct_mixture','Full','No_POI_prior','No_POI_mixture','No_POI_neural']:
        vals=evaluate(shares[k].to_numpy());vals['map']=k;paths.append(vals)
    pd.DataFrame(paths).to_csv(OUT/'Deep_shared_source_screening.csv',index=False)
    nopoi=np.load(DATA/'new_probabilities_No_POI_lineage.npy');assert nopoi.shape==(5,len(d))
    np.save(OUT/'Deep_no_poi_seed_maps.npy',nopoi)
    np.save(OUT/'Deep_source_removed_prior.npy',shares.No_POI_prior.to_numpy())
    np.save(OUT/'Deep_source_removed_target.npy',shares.No_POI_mixture.to_numpy())
    summary={'purpose':'Fitted-model decomposition and design sensitivity; no new training or emissions accuracy claim',
             'checkpoint_max_abs_error':max(r['reproduction_max_abs_share_error'] for r in rows),
             'same_top198_for_all_five_checkpoints':True,
             'channel_additivity_max_error':float(abs(A+B-(FULL-BASE)).max()),
             'mean_cancellation_fraction':float(np.mean([r['cancellation_fraction'] for r in rows])),
             'mean_coordinate_attributed_tv':float(np.mean([r['coordinate_tv'] for r in rows])),
             'mean_context_attributed_tv':float(np.mean([r['context_tv'] for r in rows])),
             'all_256_prior_design_draws_reproduced':True,
             'diagnostic_cutoffs':{'absolute_adjustment_pp_q80':xcut,'design_iqr_pp_q80':ycut},
             'diagnostic_group_counts':{str(i):int((flag==i).sum()) for i in range(4)},
             'median_design_sd_to_seed_sd_ratio':float(np.median(design_sd/np.maximum(seed_sd,1e-15))),
             'numpy':np.__version__,'pandas':pd.__version__,'torch':torch.__version__,'python':platform.python_version(),
             'checkpoint_sha256':{str(seed):hashlib.sha256((DATA/f'model_Full_{seed}.pt').read_bytes()).hexdigest() for seed in SEEDS}}
    (P/'Deep_analysis_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('DEEP_ANALYSIS_COMPLETE',json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
