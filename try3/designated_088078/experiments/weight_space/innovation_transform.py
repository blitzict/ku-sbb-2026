"""Recover approximately exchangeable innovations from the real process families."""
import numpy as np
import pandas as pd
from scipy.special import expit,logsumexp
from threadpoolctl import threadpool_limits

from partition_evidence import ROOT,OUT


# Integrate over an exactly memoryless law, bounded stationary AR(1), and an
# integrated process. Point masses matter for the random walk and white noise.
PHI = np.r_[0.,1.,np.linspace(-.94,.94,77)]
PRIOR = np.log(np.r_[.55,.09,np.full(77,.36/77)])


def law(x):
    a,b = x[:-1]-x[:-1].mean(),x[1:]-x[1:].mean()
    rss = (b@b-2*PHI*(a@b)+PHI**2*(a@a)).clip(1e-8)
    return -.5*len(a)*np.log(rss/len(a)),rss


def posterior(ll):
    z = logsumexp(ll+PRIOR)
    w = np.exp(ll+PRIOR-z)
    return w@PHI,w,z


def moments(a,b):
    n,m = len(a),len(b)
    length = np.sqrt(n*m/(n+m))
    mu,mv = a.mean(),b.mean()
    s,t = max(a.std(),1e-7),max(b.std(),1e-7)
    std = np.sqrt((n*s*s+m*t*t)/(n+m))
    u,v = (a-mu)/s,(b-mv)/t
    q = np.array([.01,.05,.1,.25,.5,.75,.9,.95,.99])
    uq,vq = np.quantile(u,q),np.quantile(v,q)
    gap = (vq-uq)*length
    level = (mv-mu)*length/std
    scale = np.log(t/s)*length
    peak = abs(b[0]-mu)/s
    f = dict(level=abs(level),scale=abs(scale),median=length*abs(np.median(b)-np.median(a))/std,
             mad=length*abs(np.log(max(np.median(abs(a-np.median(a))),1e-7)/max(np.median(abs(b-np.median(b))),1e-7))),
             iqr_shape=abs(gap[5]-gap[3]),tail_shape=abs(gap[7]-gap[1]),shape=np.linalg.norm(gap),
             skew_shape=abs(gap[6]+gap[2]-2*gap[4]),abs_gap=abs(np.mean(abs(u))-np.mean(abs(v)))*length,
             pre_sd=s,post_sd=t,mean_pre=abs(mu),mean_post=abs(mv),start_jump=peak)
    return f


def transform(raw,b,s_common,s_separate,g):
    a,c = raw[:b],raw[b:]
    # A low-frequency random walk should never be demodulated at a short
    # accidental spectral peak. Mix the two representations with the gate.
    combined = raw+g*(s_common-raw)
    source = raw+g*(s_separate-raw)
    a,c = source[:b],source[b:]
    l,r = law(a)[0],law(c)[0]
    both = l+r
    common,w,marg = posterior(both)
    pre,wp,p = posterior(l)
    post,wq,q = posterior(r)
    ctr = float((a.mean()+c.mean())/2)
    previous = np.r_[combined[0],combined[:-1]]
    # Retain actual sample order and the observed increment at the boundary.
    # The initial observation has no predecessor and is kept in centered units.
    shared = combined-ctr-common*(previous-ctr)
    separate = source-ctr-np.r_[np.full(b,pre),np.full(len(c),post)]*(np.r_[source[0],source[:-1]]-ctr)
    shared[0],separate[0] = combined[0]-ctr,source[0]-ctr
    f = dict(white=w[0],walk=w[1],phi=common,pre_phi=pre,post_phi=post,pre_walk=wp[1],post_walk=wq[1],
             pre_white=wp[0],post_white=wq[0],dynamics_bf=p+q-marg,
             dynamics_delta=abs(pre-post)*np.sqrt(b*len(c)/(b+len(c))),
             crossover_walk=abs(wp[1]-wq[1]),crossover_white=abs(wp[0]-wq[0]),
             raw_jump=abs(raw[b]-raw[b-1]),walk_reset=abs(raw[b])-abs(raw[b-1]),
             pre_endpoint=abs(raw[b-1]),post_initial=abs(raw[b]),start=abs(raw[0]))
    for prefix,stream in [("shared",shared),("separate",separate),("increments",np.r_[0,np.diff(raw)])]:
        values = moments(stream[1:b],stream[b:])
        for k,v in values.items():
            f[prefix+"_"+k] = v
    return shared,separate,f


def run(part):
    with np.load(ROOT/f"tune_out/seq_cache_{part}.npz",allow_pickle=True) as z:
        ids,raw,n,b = (z[k] for k in ("ids","x","n","b"))
    tab = pd.read_parquet(OUT/f"periodic_{part}.parquet")
    assert np.array_equal(ids,tab.index)
    with np.load(OUT/f"periodic_streams_{part}.npz") as z:
        assert np.array_equal(ids,z["ids"])
        c,s = z["common"],z["separate"]
    g = tab.gate.to_numpy()*expit((tab.period.to_numpy()-9)/1.5)*expit((38-tab.period.to_numpy())/2)
    common,separate = np.zeros_like(raw),np.zeros_like(raw)
    rows = []
    for i in range(len(raw)):
        end = n[i]
        u,v,f = transform(raw[i,:end].astype(float),b[i],c[i,:end],s[i,:end],g[i])
        common[i,:end],separate[i,:end] = u,v
        rows.append(f)
    features = pd.DataFrame(rows,index=tab.index)
    assert np.isfinite(features).all().all()
    features.to_parquet(OUT/f"innov_v2_{part}.parquet")
    np.savez(OUT/f"innov_v2_streams_{part}.npz",ids=tab.index.to_numpy(dtype=str),common=common,separate=separate,n=n,b=b)
    print(part,"walk posterior > 0.5",(features.walk>.5).mean(),"white",(features.white>.5).mean(),flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        run("train")
        run("test")
