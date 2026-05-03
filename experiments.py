import json, os, random
import numpy as np, pandas as pd, torch
from tqdm import tqdm

from solvers import random_fourier_ic, solve_heat, solve_burgers
from models import TimeConditionedConvNet1D, FNO1D
from metrics import rel_l2, rollout_metrics, spearman_safe
from stats_utils import bootstrap_mean_ci, bootstrap_spearman_ci, paired_bootstrap_diff_ci, wilcoxon_pvalue, cohens_d
from plots import save_sg_vs_rollout, save_rollout_curves, save_seen_vs_unseen, save_lambda_ablation

QUICK = os.getenv("QUICK", "0") == "1"
N_SEEDS, QUICK_N_SEEDS = 5, 2
EPOCHS, QUICK_EPOCHS = 40, 5
BATCH_SIZE = 64
N_GRID, QUICK_N_GRID = 128, 64
N_TRAIN, N_VAL, N_TEST = 128, 32, 64
QUICK_N_TRAIN, QUICK_N_VAL, QUICK_N_TEST = 24, 8, 12
N_SAVE, T_FINAL = 21, 1.0
BOOTSTRAP_RESAMPLES, QUICK_BOOTSTRAP_RESAMPLES = 2000, 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUN_ABLATION = QUICK


def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def make_model(name):
    return TimeConditionedConvNet1D(width=32, n_blocks=3).to(DEVICE) if name=="tc_conv" else FNO1D(width=32, modes=12, n_blocks=2).to(DEVICE)

def train(model, tr, times, epochs, lambda_sg):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    n, nt, _ = tr.shape
    for _ in tqdm(range(epochs), leave=False):
        idx = np.random.permutation(n)
        for i in range(0, n, BATCH_SIZE):
            b = idx[i:i+BATCH_SIZE]
            tix = np.random.randint(1, nt, size=len(b))
            u0 = torch.tensor(tr[b, 0], device=DEVICE)
            ut = torch.tensor(tr[b, tix], device=DEVICE)
            t = torch.tensor(times[tix], device=DEVICE)
            pred = model(u0, t)
            loss = ((pred - ut) ** 2).mean()
            if lambda_sg > 0:
                s_ix = np.random.randint(1, nt // 2 + 1, size=len(b))
                t2_ix = np.random.randint(1, nt - s_ix, size=len(b))
                s = torch.tensor(times[s_ix], device=DEVICE)
                t2 = torch.tensor(times[t2_ix], device=DEVICE)
                st = torch.tensor(times[s_ix + t2_ix], device=DEVICE)
                direct = model(u0, st)
                comp = model(model(u0, s), t2)
                sg = ((direct - comp) ** 2).mean()
                loss = loss + lambda_sg * sg
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

def eval_model(model, te, times, pair_ix):
    n, nt, _ = te.shape
    with torch.no_grad():
        u0 = torch.tensor(te[:,0], device=DEVICE)
        one = model(u0, torch.full((n,), float(times[1]), device=DEVICE)).cpu().numpy()
        one_step = rel_l2(one, te[:,1]).mean()
        preds=[te[:,0]]
        u=torch.tensor(te[:,0], device=DEVICE)
        dt=torch.full((n,), float(times[1]-times[0]), device=DEVICE)
        for _ in range(1, nt):
            u = model(u, dt); preds.append(u.cpu().numpy())
        pred=np.stack(preds,axis=1)
    curves, auc, final = rollout_metrics(pred, te)
    seen, unseen = [], []
    with torch.no_grad():
        u0 = torch.tensor(te[:,0], device=DEVICE)
        for (a,b) in pair_ix['seen']:
            direct = model(u0, torch.full((n,), float(times[a+b]), device=DEVICE)).cpu().numpy()
            comp = model(model(u0, torch.full((n,), float(times[a]), device=DEVICE)), torch.full((n,), float(times[b]), device=DEVICE)).cpu().numpy()
            seen.append(rel_l2(comp, direct))
        for (a,b) in pair_ix['unseen']:
            direct = model(u0, torch.full((n,), float(times[a+b]), device=DEVICE)).cpu().numpy()
            comp = model(model(u0, torch.full((n,), float(times[a]), device=DEVICE)), torch.full((n,), float(times[b]), device=DEVICE)).cpu().numpy()
            unseen.append(rel_l2(comp, direct))
    sg_seen = np.mean(np.stack(seen,1),1); sg_unseen=np.mean(np.stack(unseen,1),1)
    rho = spearman_safe(sg_unseen, auc)
    return dict(one_step=one_step, rollout_auc=auc.mean(), rollout_final=final.mean(), sg_seen=sg_seen.mean(), sg_unseen=sg_unseen.mean(), spearman=rho,
                per_traj=dict(auc=auc, sg_seen=sg_seen, sg_unseen=sg_unseen, curve=curves))


def main():
    n_seeds = QUICK_N_SEEDS if QUICK else N_SEEDS
    epochs = QUICK_EPOCHS if QUICK else EPOCHS
    n_grid = QUICK_N_GRID if QUICK else N_GRID
    n_train = QUICK_N_TRAIN if QUICK else N_TRAIN
    n_val = QUICK_N_VAL if QUICK else N_VAL
    n_test = QUICK_N_TEST if QUICK else N_TEST
    n_boot = QUICK_BOOTSTRAP_RESAMPLES if QUICK else BOOTSTRAP_RESAMPLES
    os.makedirs('outputs', exist_ok=True)
    rows=[]; raw=[]; curves={}
    for seed in range(n_seeds):
        set_seed(seed)
        pairs={'seen':[(1,1),(2,1),(1,2)], 'unseen':[(4,3),(5,3),(6,2)]}
        for system, nu, solver in [("heat",0.05,solve_heat),("burgers",0.02,solve_burgers)]:
            for regime, sp, nu_eval in [("id",2.0,nu), ("ood_ic",1.0,nu), ("ood_nu",2.0,0.03 if system=='burgers' else nu)]:
                u0 = random_fourier_ic(n_train+n_val+n_test, n_grid, spectrum_power=sp, rng=seed)
                traj, times = solver(u0, nu_eval, T_FINAL, N_SAVE)
                tr, te = traj[:n_train], traj[n_train+n_val:]
                for model_name in ["tc_conv","fno1d"]:
                    for variant, lam in [("baseline",0.0),("sg",0.01)]:
                        m = make_model(model_name); train(m, tr, times, epochs, lam)
                        out = eval_model(m, te, times, pairs)
                        row=dict(seed=seed,system=system,regime=regime,model=model_name,variant=variant,lambda_sg=lam,
                                 one_step=out['one_step'],rollout_auc=out['rollout_auc'],rollout_final=out['rollout_final'],sg_seen=out['sg_seen'],sg_unseen=out['sg_unseen'],spearman=out['spearman'])
                        rows.append(row); curves[f"{system}-{model_name}-{variant}"]=out['per_traj']['curve']
                        for i in range(len(out['per_traj']['auc'])):
                            raw_row = {k: v for k, v in row.items() if k not in ('rollout_auc', 'sg_seen', 'sg_unseen')}
                            raw.append(dict(**raw_row, traj=i, rollout_auc=out['per_traj']['auc'][i], sg_seen=out['per_traj']['sg_seen'][i], sg_unseen=out['per_traj']['sg_unseen'][i]))
    if RUN_ABLATION:
        for lam in [0.0,0.001,0.01,0.1]:
            set_seed(999); u0=random_fourier_ic(n_train+n_val+n_test,n_grid,rng=999); trj,times=solve_heat(u0,0.05,T_FINAL,N_SAVE)
            m=make_model('tc_conv'); train(m,trj[:n_train],times,max(3,epochs//2),lam); out=eval_model(m,trj[n_train+n_val:],times,{'seen':[(1,1),(2,1),(1,2)],'unseen':[(4,3),(5,3),(6,2)]})
            rows.append(dict(seed=999,system='heat',regime='id',model='tc_conv',variant='ablation',lambda_sg=lam,one_step=out['one_step'],rollout_auc=out['rollout_auc'],rollout_final=out['rollout_final'],sg_seen=out['sg_seen'],sg_unseen=out['sg_unseen'],spearman=out['spearman']))
    df=pd.DataFrame(rows); rdf=pd.DataFrame(raw)
    df.to_csv('outputs/summary.csv', index=False); rdf.to_csv('outputs/raw_results.csv', index=False)
    save_sg_vs_rollout(rdf,'outputs/fig_sg_vs_rollout.pdf','outputs/fig_sg_vs_rollout.png')
    save_rollout_curves(curves,'outputs/fig_rollout_curves.pdf','outputs/fig_rollout_curves.png')
    save_seen_vs_unseen(rdf,'outputs/fig_seen_vs_unseen.pdf','outputs/fig_seen_vs_unseen.png')
    adf=df[df['variant']=='ablation'] if 'ablation' in df['variant'].values else df.iloc[:1].assign(lambda_sg=[0],sg_unseen=[np.nan])
    save_lambda_ablation(adf,'outputs/fig_lambda_ablation.pdf','outputs/fig_lambda_ablation.png')
    # stats + tables
    main=[]
    for k,g in df[df['variant']!='ablation'].groupby(['system','model','variant','regime']):
        rec=dict(zip(['system','model','variant','regime'],k))
        for m in ['one_step','rollout_auc','rollout_final','sg_seen','sg_unseen','spearman']:
            mean,ci=bootstrap_mean_ci(g[m].values,n_resamples=n_boot,rng=0); rec[m]=mean; rec[m+'_ci_low']=ci[0]; rec[m+'_ci_high']=ci[1]
        main.append(rec)
    pd.DataFrame(main).to_csv('outputs/summary.csv', index=False)
    stats={"hypotheses":{}}
    if len(rdf)>5:
        r,ci=bootstrap_spearman_ci(rdf['sg_unseen'],rdf['rollout_auc'],n_boot,rng=0); stats['hypotheses']['H1']={"rho":r,"ci":ci}
    stats['hypotheses']['H2']={"mean_diff_unseen_minus_seen":float((rdf['sg_unseen']-rdf['sg_seen']).mean())}
    b=df[(df.variant=='baseline')&(df.regime=='id')]; s=df[(df.variant=='sg')&(df.regime=='id')]
    if len(b)==len(s) and len(b)>0:
        md,ci=paired_bootstrap_diff_ci(b['sg_unseen'],s['sg_unseen'],n_boot,rng=0)
        stats['hypotheses']['H3']={"baseline_minus_sg_sg_unseen":md,"ci":ci,"wilcoxon_p":wilcoxon_pvalue(b['sg_unseen'],s['sg_unseen']),"cohens_d":cohens_d(b['sg_unseen'],s['sg_unseen'])}
        md2,ci2=paired_bootstrap_diff_ci(b['rollout_auc'],s['rollout_auc'],n_boot,rng=0)
        stats['hypotheses']['H4']={"baseline_minus_sg_rollout_auc":md2,"ci":ci2,"wilcoxon_p":wilcoxon_pvalue(b['rollout_auc'],s['rollout_auc']),"cohens_d":cohens_d(b['rollout_auc'],s['rollout_auc'])}
    stats['hypotheses']['H5']={"note":"Compare one_step vs rollout/sg differences in summary.csv; may be mixed."}
    with open('outputs/stats.json','w') as f: json.dump(stats,f,indent=2)
    with open('outputs/paper_tables.tex','w') as f: f.write(df.head(20).to_latex(index=False,float_format='%.4f'))
    with open('outputs/run_config.json','w') as f: json.dump(dict(QUICK=QUICK,n_seeds=n_seeds,epochs=epochs,n_grid=n_grid,n_train=n_train,n_val=n_val,n_test=n_test,n_bootstrap=n_boot,run_ablation=RUN_ABLATION),f,indent=2)
    with open('outputs/README_results.txt','w') as f: f.write('Lightweight workshop results. See summary.csv, raw_results.csv, stats.json, and figures.\n')

if __name__=='__main__': main()
