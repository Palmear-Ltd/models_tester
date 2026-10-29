import os, sys, glob, json, numpy as np, pandas as pd, soundfile as sf
from concurrent.futures import ProcessPoolExecutor
OUT=sys.argv[1]
d=pd.read_csv("training/one_shot_1/dataset.csv")
d=d[(d.year==2026)&(d.duration_s.round(1)==20.0)]
sc={}
for f in glob.glob("training/9_1_5/.score_cache/9_1_2/*.json"):
    j=json.load(open(f)); sc.setdefault(os.path.basename(j["path"]),j["scores"])
for p,s in json.load(open(os.path.join(os.path.dirname(OUT),"missing_scores.json"))).items(): sc.setdefault(os.path.basename(p),s)
d=d[d.basename.isin(sc)].copy()
HOP=22050
def floor(x,p=90):
    lim=np.percentile(np.abs(x),p); c=np.clip(x,-lim,lim)
    return 20*np.log10(np.sqrt(np.mean(c*c))+1e-12)
def job(p):
    x,sr=sf.read(p,frames=20*44100,always_2d=True); x=x[:,0].astype(np.float64)
    fl=np.array([floor(x[h*HOP:(h+1)*HOP]) for h in range(len(x)//HOP)])
    return dict(abs_path=p,fl_med=np.median(fl),fl_p20=np.percentile(fl,20),fl_p90=np.percentile(fl,90),fl_max=fl.max(),
                swing=np.percentile(fl,90)-np.percentile(fl,10),floors=json.dumps(np.round(fl,2).tolist()))
os.environ["OMP_NUM_THREADS"]="1"
with ProcessPoolExecutor(16) as ex: r=pd.DataFrame(list(ex.map(job,d.abs_path,chunksize=16)))
d=d.merge(r,on="abs_path")
d["scores"]=d.basename.map(lambda b: json.dumps(sc[b]))
d.to_csv(OUT,index=False); print(len(d)); print(d.groupby(["role","label"]).size())
