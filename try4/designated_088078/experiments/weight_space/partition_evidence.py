"""Label-free, multiresolution evidence for a change in the innovation law.

The beta-binomial splits follow the Pólya-tree two-sample construction of
Holmes et al. Here the partitions come from pooled ranks. Sparse pooling and
tail scans are additional features; temporal dependence prevents interpreting
them directly as posterior probabilities.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betaln, xlogy
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tune_out/weight_space"
VIEWS = ["level","innov","shape","difference","amplitude","lag_product"]
STATS = ["coarse","mid","fine","sparse","max_node","left_tail","right_tail","both_tails"]
P = 6


def fit_ar(a,b):
    sides = [np.lib.stride_tricks.sliding_window_view(v,P+1) for v in (a,b)]
    centered = np.vstack([m-m.mean(axis=0) for m in sides])
    lags = centered[:,:P][:,::-1]
    target = centered[:,-1]
    gram = lags.T@lags
    cov = lags.T@target
    rss = target@target
    best = len(target)*np.log(max(rss/len(target),1e-12))
    best_coef = np.zeros(P)
    for k in range(1,P+1):
        beta = np.linalg.solve(gram[:k,:k]+np.eye(k)*max(rss*1e-8,1e-10),cov[:k])
        error = max(rss - 2*(cov[:k]@beta) + beta@gram[:k,:k]@beta,1e-10)
        bic = len(target)*np.log(error/len(target))+k*np.log(len(target))
        if bic < best:
            best = bic
            best_coef[:] = 0
            best_coef[:k] = beta
    return tuple(m[:,-1]-m[:,:P][:,::-1]@best_coef for m in sides)


def shape(x):
    med = np.median(x)
    mad = np.median(np.abs(x-med))
    return (x-med)/max(mad,.05*x.std(),1e-8)


def histogram(a,b):
    both = np.concatenate([a,b])
    order = np.argsort(both,kind="stable")
    bins = np.arange(len(order))*64//len(order)
    values = both[order]
    ties = np.flatnonzero(values[1:]==values[:-1])
    if ties.size:
        starts,counts = np.unique(values,return_counts=True,return_index=True)[1:]
        bins = np.repeat(((2*starts+counts-1)*32//len(order)),counts)
    left = np.bincount(bins[order<len(a)],minlength=64)
    right = np.bincount(bins[order>=len(a)],minlength=64)
    prefix = np.r_[0,np.cumsum(order>=len(a))]
    # Split at unique values only. A tied observation has no preferred side.
    valid = np.r_[False,values[1:]!=values[:-1],False]
    return left,right,prefix,valid


def tail_scan(total_after,after_counts,ks,n,valid):
    side = total_after/n
    freq = after_counts/ks
    stat = xlogy(after_counts,freq/side)+xlogy(ks-after_counts,(1-freq)/(1-side))
    # Finite-population correction matters for the larger tested prefixes.
    stat /= (1-ks/n)
    return float(np.max(np.where(valid,stat,0)))


def profiles(a,b):
    left,right,prefix,valid = histogram(a,b)
    evidence = []
    sparse,max_node = 0.,0.
    for depth in range(6):
        count = 2**depth
        x = left.reshape(count,2,-1).sum(axis=-1)
        y = right.reshape(count,2,-1).sum(axis=-1)
        prior = .5*(depth+1)**2
        def ll(k):
            return betaln(k[:,0]+prior,k[:,1]+prior)-betaln(prior,prior)
        delta = ll(x)+ll(y)-ll(x+y)
        evidence.append(delta.sum())
        sparse += np.logaddexp(np.log(.9),np.log(.1)+delta).sum()
        max_node = max(max_node,float(delta.max()))
    n = len(a)+len(b)
    ks = np.unique(np.minimum(np.array([1,2,3,5,8,12,16,24,32,48,64,96,128]),n//5))
    lt = tail_scan(len(b),prefix[ks],ks,n,valid[ks])
    rt = tail_scan(len(b),len(b)-prefix[n-ks],ks,n,valid[n-ks])
    return [sum(evidence[:2]),sum(evidence[2:4]),sum(evidence[4:]),sparse,max_node,lt,rt,lt+rt]


def row_features(x,b):
    a,c = x[:b],x[b:]
    u,v = fit_ar(a,c)
    su,sv = shape(u),shape(v)
    views = [(a,c),(u,v),(su,sv),(np.diff(a),np.diff(c)),
             (abs(su),abs(sv)),(su[:-1]*su[1:],sv[:-1]*sv[1:])]
    return np.concatenate([profiles(left,right) for left,right in views])


def run(part):
    source = ROOT/f"tune_out/seq_cache_{part}.npz"
    with np.load(source,allow_pickle=True) as m:
        ids,x,n,b = (m[k] for k in ("ids","x","n","b"))
    index = pd.read_parquet(ROOT/f"clean/features_{part}.parquet",columns=[]).index
    assert index.is_unique and np.array_equal(index,ids)
    labels = [f"{view}_{stat}" for view in VIEWS for stat in STATS]
    matrix = np.empty((len(ids),len(labels)))
    for i in range(len(ids)):
        matrix[i] = row_features(x[i,:n[i]].astype(float),b[i])
        if (i+1)%4000==0:
            print(part,i+1,flush=True)
    assert np.isfinite(matrix).all()
    dest = OUT/f"partition_{part}.parquet"
    pd.DataFrame(matrix,index=index,columns=labels).to_parquet(dest)
    print(dest,flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--part",choices=["train","test","both"],default="both")
    args = p.parse_args()
    OUT.mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        for part in (["train","test"] if args.part=="both" else [args.part]):
            run(part)
