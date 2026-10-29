import sys, os, json, numpy as np, pandas as pd
sys.argv=[sys.argv[0], sys.argv[1], "/dev/null"]
src=open("scripts/noisy_test_background/sim_extend.py").read().split("MODES = {")[0]
exec(src)
for x in (8.0,10.0):
    cfg=NoisyTestConfig(enabled=True,cutoff=CUT,span=SPAN,noise_db=x,max_extra_sec=40.0)
    for lab in ("F","T"):
        ts=[t for t in tracks if t["label"]==lab]
        n=np.array([replay(cfg,t["floors"],t["scores"]).n_hops for t in ts])*0.5
        print(f"rel{x} extend40 {lab}: extended {100*(n>20).mean():.0f}% of tests; mean length {n.mean():.1f}s; "
              f"among extended median {np.median(n[n>20]) if (n>20).any() else 0:.1f}s; >40s {100*(n>40).mean():.0f}%")
