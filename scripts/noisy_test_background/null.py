import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from app.decision.threshold import EwmaPeakDecision, ThresholdConfig
S=sys.argv[1]; CUT=json.load(open("models/9_1_2/decision_threshold.json"))["cutoff"]
d=pd.read_csv(f"{S}/bg/table.csv"); d["sc"]=d.scores.map(json.loads); d["fl"]=d.floors.map(json.loads)
def peak(s):
    a=EwmaPeakDecision(ThresholdConfig(cutoff=CUT,span=5.0))
    for v in s: a.update(v)
    return a.peak
rng=np.random.default_rng(0)
for X in [6,10]:
    out={("F","noise"):[],("T","noise"):[],("F","random"):[],("T","random"):[]}
    for r in d.itertuples():
        fl=np.array(r.fl); ref=np.percentile(fl,20); sc=np.array(r.sc)
        keep=np.array([fl[w:w+5].max()<=ref+X for w in range(len(sc))])
        if keep.sum()<12 or keep.all(): continue   # only recordings where something was dropped
        out[(r.label,"noise")].append(peak(sc[keep])>CUT)
        hits=[]
        for _ in range(20):
            idx=np.sort(rng.choice(len(sc),keep.sum(),replace=False)); hits.append(peak(sc[idx])>CUT)
        out[(r.label,"random")].append(np.mean(hits))
    for lab in ["F","T"]:
        print(f"X={X} {lab} (n={len(out[(lab,'noise')])} recordings with some windows dropped): positive when dropping NOISY windows "
              f"{100*np.mean(out[(lab,'noise')]):.1f}%  vs dropping the same number at RANDOM {100*np.mean(out[(lab,'random')]):.1f}%")
