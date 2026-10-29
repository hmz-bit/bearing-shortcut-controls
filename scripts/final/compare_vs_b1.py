"""Paired bearing-level bootstrap: LF-PNCN family vs retrained B1 (WDCNN) for Tier 3 (H3) and Tier 2."""
import glob, json, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'evaltools'))
import bootstrap as bs

R23, RB = 'results/tier2_3/', 'results/b1/'
B = 2000
out = {}

def cat(pat): return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(pat))], ignore_index=True)

# ---- Tier 3
b1 = pd.read_parquet(RB + 't3_predictions_long.parquet')
res3 = {}
for meth in ('lfpncn', 'b4', 'f'):
    a = cat(R23 + f't3_{meth}_*_predictions_long.parquet')
    d = {}
    for cond, tag in (('N15_M07_F10', 'same'), ('N09_M07_F10', 'r900')):
        fa, fb = a[a.test_cond == cond], b1[b1.test_cond == cond]
        cols = ['split', 'bearing', 'y_inner', 'y_outer', 's_inner', 's_outer']
        d[tag] = bs.paired_bootstrap_multi_split(fa[cols], fb[cols], B=B, seed=0, return_draws=True)
    same, r900 = d['same']['draws']['diff'], d['r900']['draws']['diff']
    loss_diff = same - r900            # (A_same - B1_same) - (A_900 - B1_900) = loss_A - loss_B1
    pt = d['same']['diff']['point'] - d['r900']['diff']['point']
    lo, hi = np.percentile(loss_diff, [2.5, 97.5])
    res3[meth] = {'same_speed_minus_B1': {k: d['same']['diff'][k] for k in ('point', 'ci_low', 'ci_high')},
                  'speed900_minus_B1': {k: d['r900']['diff'][k] for k in ('point', 'ci_low', 'ci_high')},
                  'loss_minus_B1_loss': {'point': float(pt), 'lo': float(lo), 'hi': float(hi)}}
out['tier3'] = res3

# ---- Tier 2 (B1 has split = pair index 1..15; LF-PNCN family has pair 0..14, seeds p and p+1000)
b1t2 = pd.read_parquet(RB + 't2_predictions_long.parquet'); b1t2 = b1t2.assign(split=b1t2.split)
cols = ['split', 'bearing', 'y_inner', 'y_outer', 's_inner', 's_outer']
res2 = {}
for meth in ('lfpncn', 'b4', 'f'):
    a = cat(R23 + f't2_{meth}_*_predictions_long.parquet')
    for seed_rule in ('first', 'second'):
        sub = []
        for p, g in a.groupby('pair'):
            seeds = sorted(g.seed.unique())
            s = seeds[0] if seed_rule == 'first' else seeds[-1]
            sub.append(g[g.seed == s].assign(split=p + 1))
        fa = pd.concat(sub)[cols]
        r = bs.paired_bootstrap_multi_split(fa, b1t2[cols], B=B, seed=0)
        res2[f'{meth}_{seed_rule}_seed'] = {'A': r['model_A']['point'], 'B1': r['model_B']['point'], 'diff': r['diff']}
out['tier2'] = res2
json.dump(out, open('results/b1/compare_vs_b1.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
