import os, sys, json, pandas as pd
sys.path.insert(0, os.getcwd())
import offline_score as o
from concurrent.futures import ProcessPoolExecutor
S=sys.argv[1]
d=pd.read_csv("../training/one_shot_1/dataset.csv"); d=d[(d.year==2026)&(d.duration_s.round(1)==20.0)]
t=pd.read_csv(f"{S}/bg/table.csv",usecols=["abs_path"])
paths=d[~d.abs_path.isin(t.abs_path)].abs_path.tolist(); print(len(paths),flush=True)
out={}
init=("models/9_1_2/model.tflite","models/9_1_2/scaler.json",dict(o.DEFAULT_PREP_PARAMS))
with ProcessPoolExecutor(16,initializer=o._init_worker,initargs=init) as ex:
    for k,(p,s) in enumerate(ex.map(o._score_one,paths,chunksize=8),1):
        out[p]=[float(v) for v in s]
        if k%500==0: print(k,flush=True)
json.dump(out,open(f"{S}/bg/missing_scores.json","w"))
