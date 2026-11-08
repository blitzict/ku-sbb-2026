"""Integrate a short nuisance pulse while retaining genuine persistent steps."""
import numpy as np
import pandas as pd
from scipy.special import expit,logsumexp
from threadpoolctl import threadpool_limits

from partition_evidence import ROOT,OUT


TIME=np.arange(-42,43)
CENTER,WIDTH=np.meshgrid(np.arange(-8,9),[.8,1.25,1.8,2.6,3.8,5.5,8.])
CENTER,WIDTH=CENTER.ravel(),WIDTH.ravel()
PULSE=np.exp(-.5*((TIME[:,None]-CENTER)/WIDTH)**2)
STEP=(TIME>=0).astype(float)
BASIS=np.column_stack([np.ones(len(TIME)),TIME/42,STEP,PULSE])
PRIOR=np.exp(-.5*(CENTER/4.5)**2)
PRIOR/=PRIOR.sum()


def process(v,br):
    sides=[v[:br-16],v[br+16:]]
    mats=[];targets=[]
    for s in sides:
        q=s-np.median(s)
        sd=max(np.median(abs(q))*1.4826,.12*q.std(),1e-5)
        q=q.clip(-5*sd,5*sd)
        mats.append(np.column_stack([q[1:-1],q[:-2]]))
        targets.append(q[2:])
    X=np.vstack(mats);y=np.concatenate(targets)
    theta=np.linalg.solve(X.T@X+np.eye(2)*.8*y.var(),X.T@y)
    # Keep the integrated-process point mass and softly shrink uncertain
    # higher-order coefficients, preserving well-supported negative memory.
    if theta.sum()>.955:
        theta=np.array([1.,0.])
    else:
        theta[1]*=expit((abs(theta[1])*np.sqrt(len(y))-2.8)*2)
    res=[]
    for s in sides:
        r=s[2:]-theta[0]*s[1:-1]-theta[1]*s[:-2]
        res.append(r-np.median(r))
    scale=max(np.median(abs(np.concatenate(res)))*1.4826,1e-5)
    return theta,scale


def one(raw,v,b):
    ar,sig=process(v,b)
    obs=v[b+TIME]
    X=BASIS[2:]-ar[0]*BASIS[1:-1]-ar[1]*BASIS[:-2]
    y=(obs[2:]-ar[0]*obs[1:-1]-ar[1]*obs[:-2])/sig
    Q=X[:,:2]
    # For an exact walk the intercept disappears. Use the pseudoinverse so
    # that the remaining slope in the local increments is still removed.
    inv=np.linalg.pinv(Q)
    y-=Q@(inv@y)
    rhs=X[:,2:]-Q@(inv@X[:,2:])
    step,bump=rhs[:,0],rhs[:,1:]
    ss=step@step
    xy=bump.T@y
    xx=(bump*bump).sum(axis=0)
    sx=step@bump
    sy=step@y
    pulse_var=np.clip((2*raw.std()/sig)**2,1.5,5000.)
    step_var=np.clip((.9*raw.std()/sig)**2,.5,1500.)
    step_bf=.5*(sy**2/(ss+1/step_var)-np.log1p(ss*step_var))
    base=np.array([0,step_bf])+np.log([.70,.30])
    yy=np.stack([xy,xy-sx*sy/(ss+1/step_var)])
    gg=np.stack([xx,xx-sx**2/(ss+1/step_var)]).clip(1e-9)
    coef=yy/(gg+1/pulse_var)
    ratio=.5*(yy*coef-np.log1p(gg*pulse_var))
    a=base[:,None]+ratio+np.log(PRIOR)
    log_yes=logsumexp(a)
    log_no=logsumexp(base)
    bf=log_yes-log_no
    gate=expit(bf+np.log(.10/.90))
    post=np.exp(a-log_yes)
    marginal=post.sum(axis=0)
    pulse=PULSE@((post*coef).sum(axis=0))*sig
    adjusted=raw.copy()
    adjusted[b+TIME]-=gate*pulse
    corrected_bf=logsumexp(ratio[1]+np.log(PRIOR))-logsumexp(ratio[0]+np.log(PRIOR))+step_bf
    clean_delta=logsumexp([np.log(.9)+step_bf,np.log(.1)+logsumexp(ratio[1]+np.log(PRIOR))+step_bf])-logsumexp([np.log(.9),np.log(.1)+logsumexp(ratio[0]+np.log(PRIOR))])
    feat=dict(gate=gate,bf=bf,center=marginal@CENTER,width=marginal@WIDTH,amplitude=np.max(abs(pulse))/sig,
              raw_step=step_bf,step_given_pulse=corrected_bf,step=clean_delta,
              dispersion=marginal@(CENTER**2)-(marginal@CENTER)**2,phi=ar[0],phi2=ar[1],scale=sig)
    physical=v.copy()
    physical[b+TIME]-=gate*pulse
    return adjusted,physical,feat


def run(part):
    with np.load(ROOT/f"tune_out/seq_cache_{part}.npz",allow_pickle=True) as q:
        ids,x,n,b=(q[k] for k in ("ids","x","n","b"))
    with np.load(OUT/f"periodic_streams_{part}.npz") as q:
        assert np.array_equal(q["ids"],ids)
        periodic=q["separate"]
    with np.load(OUT/f"curve_streams_{part}.npz") as q:
        assert np.array_equal(q["ids"],ids)
        curve=q["separate"]
    s=pd.read_parquet(OUT/f"periodic_{part}.parquet")
    assert np.array_equal(s.index,ids)
    g=(s.gate*expit((s.period-9)/1.5)*expit((38-s.period)/2)).to_numpy()
    common,separate=np.zeros_like(x),np.zeros_like(x)
    rows=[]
    for i in range(len(x)):
        sl=slice(0,n[i])
        raw=x[i,sl].astype(float)
        v=curve[i,sl].astype(float)+g[i]*(periodic[i,sl]-curve[i,sl])
        u,v,h=one(raw,v,b[i])
        common[i,sl],separate[i,sl]=u,v
        rows.append(h)
    frame=pd.DataFrame(rows,index=s.index)
    assert np.isfinite(frame).all().all()
    frame.to_parquet(OUT/f"pulse_{part}.parquet")
    np.savez(OUT/f"pulse_streams_{part}.npz",ids=s.index.to_numpy(dtype=str),common=common,separate=separate,n=n,b=b)
    print(part,"pulse posterior >0.5",int((frame.gate>.5).sum()),flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        run("train")
        run("test")
