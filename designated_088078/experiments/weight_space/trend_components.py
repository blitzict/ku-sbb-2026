"""Identify persistent trends using within-segment slopes; retain true level jumps."""
import numpy as np
import pandas as pd
from scipy.special import expit,logsumexp
from threadpoolctl import threadpool_limits

from phase_likelihood import DFS,ll_grid,student_beta
from partition_evidence import ROOT,OUT


def bayes(z,variance,prior):
    variance=max(variance,1e-10)
    ratio=prior/variance
    return .5*(z*z/variance*ratio/(1+ratio)-np.log1p(ratio))


def one(raw,b,walk,season):
    length=len(raw)
    tt=(np.arange(length)-b)/length
    B=np.column_stack([np.ones(length),tt])
    a,c=raw[:b],raw[b:]
    pa,pc=(np.linalg.lstsq(q,x,rcond=None)[0] for q,x in ((B[:b],a),(B[b:],c)))
    ea,ec=a-B[:b]@pa,c-B[b:]@pc
    lag=np.r_[ea[:-1],ec[:-1]]
    future=np.r_[ea[1:],ec[1:]]
    ph=np.clip(lag@future/max(lag@lag,1e-9),-.93,.97)
    phi=ph*expit((abs(ph)*np.sqrt(length)-2.5)*2)
    # Keep all paired lags inside their segment so a level jump cannot enter
    # the shared-slope estimate as a single large transition.
    U=B[1:b]-phi*B[:b-1]
    V=B[b+1:]-phi*B[b:-1]
    l,r=a[1:]-phi*a[:-1],c[1:]-phi*c[:-1]
    la,lc=ll_grid(l-U@pa),ll_grid(r-V@pc)
    df=DFS[np.argmax(la+lc)]
    pl,sl,vl,el=student_beta(U,l,pa,df)
    pr,sr,vr,er=student_beta(V,r,pc,df)
    var_l=max(vl[1,1],1e-8)
    var_r=max(vr[1,1],1e-8)
    sig=max((sl+sr)/2,1e-6)
    prior=(2*sig)**2
    v=1/(1/var_l+1/var_r)
    shared=v*(pl[1]/var_l+pr[1]/var_r)
    zshared=shared/np.sqrt(v)
    lbf=bayes(shared,v,prior)
    sepbf=bayes(pl[1],var_l,prior)+bayes(pr[1],var_r,prior)
    g=expit(np.logaddexp(lbf+np.log(.14),sepbf+np.log(.025))-np.log(.835))
    gate=g*(1-walk)*(1-season)
    shared_slope=shared*prior/(v+prior)
    u=raw-gate*shared_slope*tt
    al=pl[1]*prior/(var_l+prior)
    ar=pr[1]*prior/(var_r+prior)
    vstream=raw-gate*np.r_[al*tt[:b],ar*tt[b:]]
    # The common-slope estimate contributes to both level uncertainties.
    k1=vl[0,1]/var_l
    k2=vr[0,1]/var_r
    adjusted_delta=(pr[0]-k2*(pr[1]-shared))-(pl[0]-k1*(pl[1]-shared))
    adjusted_v=max(vl[0,0]-k1**2*var_l,1e-8)+max(vr[0,0]-k2**2*var_r,1e-8)+(k2-k1)**2*v
    level=bayes(adjusted_delta,adjusted_v,(.9*sig)**2)
    hinge=bayes(pr[1]-pl[1],var_r+var_l,prior)
    free_level=bayes(pr[0]-pl[0],vl[0,0]+vr[0,0],(.9*sig)**2)
    ne=b*(length-b)/length
    scale=bayes(np.log(sr/sl),1/(ne*2*df/(df+3)),.3)
    shape=logsumexp(la)+logsumexp(lc)-logsumexp(la+lc)-np.log(len(DFS))
    f=dict(gate=gate,confidence=g,common_bf=lbf,unshared_bf=sepbf,common_slope=abs(shared_slope)/sig,
           prior_slope=abs(al)/sig,post_slope=abs(ar)/sig,slope_z=abs(zshared),phi=phi,df=df,
           level_bf=level,hinge_bf=hinge,free_level_bf=free_level,scale_bf=scale,shape_bf=shape,
           pre_scale=sl,post_scale=sr,magnitude=abs(adjusted_delta)/sig,
           slope_ratio=abs(pr[1]-pl[1])/sig,coherence=np.sign(pl[1]*pr[1])*min(abs(pl[1])/np.sqrt(var_l),abs(pr[1])/np.sqrt(var_r)))
    f["intervention"]=logsumexp(np.array([level,hinge,free_level,scale,shape,0])+np.log([.27,.18,.10,.24,.11,.10]))
    return u,vstream,f


def run(part):
    with np.load(ROOT/f"tune_out/seq_cache_{part}.npz",allow_pickle=True) as z:
        ids,raw,n,b=(z[k] for k in ("ids","x","n","b"))
    p=pd.read_parquet(OUT/f"periodic_{part}.parquet")
    innov=pd.read_parquet(OUT/f"innov_v2_{part}.parquet").loc[p.index]
    assert np.array_equal(ids,p.index)
    season=p.gate.to_numpy()*expit((p.period.to_numpy()-9)/1.5)*expit((38-p.period.to_numpy())/2)
    walk=innov.pre_walk.to_numpy()
    common,separate=np.zeros_like(raw),np.zeros_like(raw)
    rows=[]
    for i in range(len(raw)):
        a=raw[i,:n[i]].astype(float)
        u,v,f=one(a,b[i],walk[i],season[i])
        common[i,:n[i]],separate[i,:n[i]]=u,v
        rows.append(f)
    tab=pd.DataFrame(rows,index=p.index)
    assert np.isfinite(tab).all().all() and np.isfinite(common).all() and np.isfinite(separate).all()
    tab.to_parquet(OUT/f"trend_{part}.parquet")
    np.savez(OUT/f"trend_streams_{part}.npz",ids=p.index.to_numpy(dtype=str),common=common,separate=separate,n=n,b=b)
    print(part,"gate above 0.5",(tab.gate>.5).sum(),flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        run("train")
        run("test")
