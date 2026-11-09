"""Marginalize shared slow trends before testing abrupt, boundary-locked changes."""
import numpy as np
import pandas as pd
from scipy.special import expit,logsumexp
from threadpoolctl import threadpool_limits

from partition_evidence import ROOT,OUT


def one(raw,b,phi,df):
    n=len(raw)
    u=np.arange(n)/(n-1)
    v=u-u[b]
    side=(np.arange(n)>=b).astype(float)
    # Each column is one distinct prediction about the time profile of an
    # intervention. The absolute clock and a clock restarted at the break
    # give different predictions even if both permit the same slope.
    X0=np.column_stack([np.ones(n),u,u*u])
    H0=np.column_stack([side,side*v,side*u,side*u*u,side*v*v,side*(u+u*u)/2])
    phi=np.clip(phi,-.9,.94)
    X=X0[1:]-phi*X0[:-1]
    H=H0[1:]-phi*H0[:-1]
    y=raw[1:]-phi*raw[:-1]
    rich=np.column_stack([X,H[:,:2],H[:,4]])
    beta=np.linalg.lstsq(rich,y,rcond=None)[0]
    for _ in range(3):
        e=y-rich@beta
        scale=max(np.median(abs(e-np.median(e)))*1.4826,.18*e.std(),1e-6)
        w=(df+1)/(df+(e/scale)**2)
        beta=np.linalg.solve(rich.T@(rich*w[:,None])+np.eye(rich.shape[1])*1e-7,rich.T@(w*y))
    noise=y-rich@beta
    # Normalize the estimating equation by its observed score dispersion.
    precision=w/max(np.sum(w*w*noise**2)/max(w.sum(),1e-6),1e-9)
    allx=np.column_stack([X,H])
    gram=allx.T@(allx*precision[:,None])
    cross=allx.T@(y*precision)
    # Remove the common intercept with its flat prior exactly.
    gg=gram[1:,1:]-np.outer(gram[1:,0],gram[0,1:])/gram[0,0]
    cc=cross[1:]-gram[1:,0]*cross[0]/gram[0,0]
    paths=[np.zeros((2,0)),np.array([[1.],[0.]]),np.array([[0.],[1.]]),
           np.array([[.5],[.5]]),np.array([[1.,0.],[0.,.65]])]
    lp=np.log([.76,.075,.055,.05,.06])
    scores,changes,coefs,adjusted=[],[],[],[]
    for path in paths:
        G=path.T@gg[:2,:2]@path
        c=path.T@cc[:2]
        prior=(1.8*scale)**2
        mat=G+np.eye(len(c))/prior
        inv=np.linalg.inv(mat) if len(c) else mat
        coef=path@(inv@c)
        like=.5*(c@inv@c-np.linalg.slogdet(np.eye(len(c))+prior*G)[1])
        scores.append(like)
        coefs.append(coef)
        join=gg[2:,:2]@path
        ef=cc[2:]-join@inv@c
        info=(np.diag(gg[2:,2:])-np.diag(join@inv@join.T)).clip(1e-8)
        powers=(scale*np.array([1.,2.,1.5,1.5,2.,1.5]))**2
        change=.5*(ef**2/(info+1/powers)-np.log1p(powers*info))
        changes.append(change)
        # For nuisance removal allow a discontinuous level change. Never
        # subtract its estimated amplitude from the observed series.
        jump=ef[0]/(info[0]+1/powers[0])
        adjusted.append(path@(inv@(c-path.T@gg[:2,2]*jump)))
    scores=np.array(scores)+lp
    changes=np.array(changes)
    marginal=logsumexp(scores)
    z=logsumexp(scores[:,None]+changes,axis=0)-marginal
    alternative=logsumexp(scores+changes[:,0])
    coeff0=np.exp(scores-marginal)@np.array(coefs)
    coeff1=np.exp(scores+changes[:,0]-alternative)@np.array(adjusted)
    # A physical shift can occur in either the observation or the slope.
    mix=logsumexp(z+np.log([.50,.15,.15,.08,.05,.07]))
    dwell=1-np.exp(scores[0]-marginal)
    safe=1-np.exp(scores[0]+changes[0,0]-alternative)
    xx=np.column_stack([u,u*u])
    common=raw-xx@coeff0
    separate=raw-xx@coeff1
    e=separate
    a,c=e[:b],e[b:]
    ma,mb=a.mean(),c.mean()
    sa,sb=max(a.std(),1e-7),max(c.std(),1e-7)
    ne=b*(n-b)/n
    f=dict(null_trend=dwell,jump_trend=safe,trend_bf=marginal,linear=scores[1]-scores[0],
           quadratic=scores[2]-scores[0],mixed=scores[3]-scores[0],
           curvature=coeff1[1]/scale,slope=coeff1[0]/scale,obs_step=z[0],slope_step=z[1],
           global_linear=z[2],global_quad=z[3],local_quad=z[4],combo=z[5],mixture=mix,
           selected_curve=max(z[1:]),adjusted_level=abs(ma-mb)*np.sqrt(ne)/scale,
           adjusted_scale=abs(np.log(sa/sb))*np.sqrt(ne),scale=scale)
    return common,separate,f


def run(part):
    with np.load(ROOT/f"tune_out/seq_cache_{part}.npz",allow_pickle=True) as a:
        ids,x,n,b=(a[k] for k in ("ids","x","n","b"))
    t=pd.read_parquet(OUT/f"trend_{part}.parquet")
    assert np.array_equal(t.index,ids)
    phi,df=t.phi.to_numpy(),t.df.to_numpy()
    s=pd.read_parquet(OUT/f"periodic_{part}.parquet").loc[t.index]
    season=s.gate.to_numpy()*expit((s.period.to_numpy()-9)/1.5)*expit((38-s.period.to_numpy())/2)
    walk=pd.read_parquet(OUT/f"innov_v2_{part}.parquet").loc[t.index,"pre_walk"].to_numpy()
    rows=[]
    common,separate=np.zeros_like(x),np.zeros_like(x)
    for i in range(len(x)):
        raw=x[i,:n[i]].astype(float)
        u,v,f=one(raw,b[i],phi[i],df[i])
        keep=(1-walk[i])*(1-season[i])
        common[i,:n[i]],separate[i,:n[i]]=raw+keep*(u-raw),raw+keep*(v-raw)
        f["gate"]=keep*max(f["jump_trend"],.5*f["null_trend"])
        rows.append(f)
    frame=pd.DataFrame(rows,index=t.index)
    assert np.isfinite(frame).all().all()
    frame.to_parquet(OUT/f"curve_{part}.parquet")
    np.savez(OUT/f"curve_streams_{part}.npz",ids=t.index.to_numpy(dtype=str),common=common,separate=separate,n=n,b=b)
    print(part,"active",int((frame.gate>.5).sum()),flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        run("train")
        run("test")
