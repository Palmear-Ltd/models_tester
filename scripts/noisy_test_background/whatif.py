import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from app.decision.threshold import EwmaPeakDecision, ThresholdConfig
S=sys.argv[1]; CUT=json.load(open("models/9_1_2/decision_threshold.json"))["cutoff"]
d=pd.read_csv(f"{S}/bg/table.csv"); d["sc"]=d.scores.map(json.loads); d["fl"]=d.floors.map(json.loads)
def peak(s):
    a=EwmaPeakDecision(ThresholdConfig(cutoff=CUT,span=5.0))
    for v in s: a.update(v)
    return a.peak
d["pos"]=d.sc.map(peak)>CUT
print("positive rate by role:"); print((100*d.groupby(["role","label"]).pos.mean()).round(1).unstack())
F=d[d.label=="F"]; T=d[d.label=="T"]
print("\nabsolute level as 'too noisy' (median background above L):")
for L in [-50,-45,-40,-35]:
    fF=F.fl_med>L; fT=T.fl_med>L
    print(f"  L={L}: flags {100*fF.mean():.0f}% of F, {100*fT.mean():.0f}% of T; F false alarms: flagged {100*F[fF].pos.mean():.0f}% vs unflagged {100*F[~fF].pos.mean():.0f}%; "
          f"share of all F false alarms that are flagged {100*F[fF].pos.sum()/F.pos.sum():.0f}%")
print("\nverdict from quiet windows only (window loudest hop <= recording's 20th pct + X dB); fewer than 12 quiet windows = too noisy:")
for X in [6,10,15]:
    res=[]
    for r in d.itertuples():
        fl=np.array(r.fl); ref=np.percentile(fl,20)
        q=[s for w,s in enumerate(r.sc) if fl[w:w+5].max()<=ref+X]
        res.append(("NZ" if len(q)<12 else peak(q)>CUT))
    d["v"]=res
    for lab in ["F","T"]:
        g=d[d.label==lab]; nz=(g.v=="NZ").mean(); pos=(g.v==True).mean()
        print(f"  X={X} {lab}: positive {100*pos:.1f}%  too-noisy {100*nz:.1f}%  (today {100*g.pos.mean():.1f}%)")
