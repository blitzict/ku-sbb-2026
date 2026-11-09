"""Compare explicit clock resets, stopped trends and continuation at the boundary."""
import numpy as np
import pandas as pd
from scipy.special import expit,logsumexp
from threadpoolctl import threadpool_limits

from partition_evidence import OUT


SHAPES=np.array([[1.,0.],[0.,1.],[.5,.5]])
LOG_SHAPE=np.log([.50,.35,.15])


def integrate(G,c,maps,amp):
    # All one-coefficient scientific hypotheses share the same magnitude prior.
    h=np.einsum("ij,jk,ik->i",maps,G,maps).clip(0)
    v=maps@c
    info=h+1/amp**2
    return .5*(v*v/info-np.log1p(h*amp**2)),v/info


def reduce_gram(X,y,w):
    G=X.T@(w[:,None]*X)
    c=X.T@(w*y)
    return G[1:,1:]-np.outer(G[1:,0],G[0,1:])/max(G[0,0],1e-9),c[1:]-G[1:,0]*c[0]/max(G[0,0],1e-9)


def one(raw,b,phi,df,keep):
    n=len(raw);u=np.arange(n)/(n-1);cut=u[b]
    s=(np.arange(n)>=b).astype(float)
    X0=np.column_stack([np.ones(n),u,u*u,s,s*u,s*u*u])
    X=X0[1:]-phi*X0[:-1];y=raw[1:]-phi*raw[:-1]
    beta=np.linalg.lstsq(X,y,rcond=None)[0]
    for _ in range(3):
        e=y-X@beta
        scale=max(np.median(abs(e-np.median(e)))*1.4826,.12*e.std(),1e-5)
        w=(df+1)/(df+(e/scale)**2)
        beta=np.linalg.solve(X.T@(w[:,None]*X)+np.eye(6)*1e-7,X.T@(w*y))
    e=y-X@beta
    precision=w/max((w*w*e**2).sum()/max(w.sum(),1),1e-8)
    G,c=reduce_gram(X,y,precision)
    basic=np.column_stack([SHAPES,np.zeros((3,3))])
    amp=2.15*scale
    zero,_=integrate(G,c,basic,amp)
    trend_ev=logsumexp(zero+LOG_SHAPE)
    null=np.logaddexp(np.log(.78),np.log(.22)+trend_ev)
    # On restarting the same clock, (u-cut)^2 - u^2 = cut^2-2*cut*u.
    boundary=SHAPES@np.array([cut,cut**2])
    diff=np.column_stack([-cut*SHAPES[:,0]+cut**2*SHAPES[:,1],-2*cut*SHAPES[:,1],np.zeros(3)])
    post=np.column_stack([np.zeros(3),SHAPES])
    changes={"restart":diff,"restart_smooth":diff+np.column_stack([boundary,np.zeros((3,2))]),
             "freeze":-post+np.column_stack([boundary,np.zeros((3,2))]),"off":-post,
             "reverse":-2*post,"reverse_smooth":-2*post+np.column_stack([2*boundary,np.zeros((3,2))])}
    vals={}
    for name,d in changes.items():
        maps=basic.copy();maps[:,2:]+=d
        ev,_=integrate(G,c,maps,amp)
        vals[name]=logsumexp(ev+LOG_SHAPE)-null
    turn_on=basic.copy();turn_on[:,:2]=0;turn_on[:,2:]=post
    start=turn_on.copy();start[:,2:]+=diff
    for name,maps in (("on",turn_on),("start",start)):
        ev,_=integrate(G,c,maps,amp)
        vals[name]=logsumexp(ev+LOG_SHAPE)-null
    vals["change"]=logsumexp(list(vals.values()))-np.log(len(vals))
    vals["trend"]=trend_ev
    series=[];gates=[]
    for mask in (np.arange(n-1)<b-1,np.arange(n-1)>=b):
        g,k=reduce_gram(X[mask,:3],y[mask],precision[mask])
        ev,co=integrate(g,k,SHAPES,amp)
        q=ev+LOG_SHAPE+np.log(.22)
        p=np.exp(q-np.logaddexp(np.log(.78),logsumexp(q)))
        trend=(p*co)@SHAPES
        series.append(raw-keep*(X0[:,1:3]@trend))
        gates.append(p.sum())
    vals.update(before=gates[0],after=gates[1],gate=keep*max(*gates,.5*expit(trend_ev-1.3)))
    return *series,vals


def run(part):
    with np.load(OUT/f"pulse_streams_{part}.npz") as q:
        ids,x,n,b=(q[k] for k in ("ids","common","n","b"))
    s=pd.read_parquet(OUT/f"periodic_{part}.parquet")
    assert np.array_equal(s.index,ids)
    t=pd.read_parquet(OUT/f"trend_{part}.parquet").loc[s.index]
    i=pd.read_parquet(OUT/f"innov_v2_{part}.parquet").loc[s.index]
    keep=((1-i.pre_walk)*(1-s.gate*expit((s.period-9)/1.5)*expit((38-s.period)/2))).to_numpy()
    common,separate=np.zeros_like(x),np.zeros_like(x); rows=[]
    for j in range(len(x)):
        raw=x[j,:n[j]].astype(float)
        a,c,f=one(raw,b[j],np.clip(t.phi.iloc[j],-.92,.94),t.df.iloc[j],keep[j])
        common[j,:n[j]],separate[j,:n[j]]=a,c
        rows.append(f)
    frame=pd.DataFrame(rows,index=s.index)
    assert np.isfinite(frame).all().all()
    frame.to_parquet(OUT/f"clock_{part}.parquet")
    np.savez(OUT/f"clock_streams_{part}.npz",ids=ids,n=n,b=b,common=common,separate=separate)
    print(part,"active clocks",int((frame.gate>.5).sum()),flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        run("train")
        run("test")
