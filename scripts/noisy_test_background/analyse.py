import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from app.decision.threshold import EwmaPeakDecision, ThresholdConfig
S=sys.argv[1]
CUT=json.load(open("models/9_1_2/decision_threshold.json"))["cutoff"]
d=pd.read_csv(f"{S}/bg/table.csv")
d["sc"]=d.scores.map(json.loads); d["fl"]=d.floors.map(json.loads)
def peak(s):
    a=EwmaPeakDecision(ThresholdConfig(cutoff=CUT,span=5.0))
    for v in s: a.update(v)
    return a.peak
d["peak"]=d.sc.map(peak); d["pos"]=d.peak>CUT
pd.set_option("display.width",220)
def table(col,bins):
    d["bin"]=pd.cut(d[col],bins)
    g=d.groupby(["bin","label"],observed=True).agg(n=("pos","size"),pos=("pos","mean"),sess=("session_id","nunique")).unstack("label")
    g[("pos","T")]=(100*g[("pos","T")]).round(1); g[("pos","F")]=(100*g[("pos","F")]).round(1)
    print(f"\n== by {col} (dBFS) ==  pos T = caught, pos F = false alarm"); print(g)
table("fl_med",[-100,-60,-55,-50,-45,-40,-35,-30,0])
table("fl_p90",[-100,-55,-50,-45,-40,-35,-30,-25,0])
table("swing",[0,4,8,12,16,20,30,100])
print("\n== by hardware/cable: median background ==")
print(d.groupby(["cable","label"]).agg(n=("pos","size"),fl_med=("fl_med","median"),pos=("pos","mean")).round(2))
# window level, T only: score vs how loud this window's hops are relative to the recording's own median
rows=[]
for r in d.itertuples():
    fl=np.array(r.fl); med=np.median(fl)
    for w,s in enumerate(r.sc):
        rows.append((r.label,r.session_id,s,fl[w:w+5].max()-med,fl[w:w+5].max()))
w=pd.DataFrame(rows,columns=["label","sess","score","rel","absmax"])
w["relbin"]=pd.cut(w.rel,[-50,0,3,6,10,15,50])
print("\n== window score by window loudness vs own recording median (dB) ==")
print(w.groupby(["relbin","label"],observed=True).score.agg(["size","mean"]).unstack("label").round(3))
w["absbin"]=pd.cut(w.absmax,[-100,-55,-50,-45,-40,-35,-30,0])
print("\n== window score by window absolute loudness (dBFS) ==")
print(w.groupby(["absbin","label"],observed=True).score.agg(["size","mean"]).unstack("label").round(3))
d.drop(columns=["sc","fl","scores","floors"]).to_csv(f"{S}/bg/per_rec.csv",index=False)
# within session: rank recordings by background inside their own session (sessions with >=8 of that label)
print("\n== within session: quietest third vs loudest third of each session's recordings ==")
out=[]
for (sess,lab),g in d.groupby(["session_id","label"]):
    if len(g)<8: continue
    q=g.fl_med.rank(pct=True)
    out.append((lab,sess,len(g),g[q<=1/3].pos.mean(),g[q>2/3].pos.mean(),g[q>2/3].fl_med.median()-g[q<=1/3].fl_med.median()))
o=pd.DataFrame(out,columns=["label","sess","n","quiet","loud","gap_db"])
for lab,g in o.groupby("label"):
    print(f"{lab}: {len(g)} sessions; positive rate quiet third {100*g.quiet.mean():.1f}%  loud third {100*g.loud.mean():.1f}%  "
          f"loud>quiet in {(g.loud>g.quiet).sum()}, quiet>loud in {(g.quiet>g.loud).sum()}; median gap {g.gap_db.median():.1f} dB")
# within recording, window level, by session-balanced mean
w["louder6"]=w.rel>6
m=w.groupby(["label","sess","louder6"]).score.mean().unstack()
m=m.dropna()
for lab in ["F","T"]:
    x=m.loc[lab]; print(f"{lab}: windows >6 dB above own median vs rest, per session: mean diff {(x[True]-x[False]).mean():+.3f}, "
                       f"higher in {(x[True]>x[False]).sum()}/{len(x)} sessions")
