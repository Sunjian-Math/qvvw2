from __future__ import annotations
import argparse,json,time,sys
from pathlib import Path
import numpy as np,pandas as pd
import scipy.sparse as sp, scipy.sparse.linalg as spla
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parent.parent
RES=ROOT/'results'; RES.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'code'))
from qvvw2.progress import progress_hit, progress_print
from qvvw2.core import (
    frozen_setup,frozen_objective_grad,normalized_l2_objective_grad,l2_objective_grad,
    fixed_scale_setup,fixed_scale_objective_grad,
)

METHODS=['L2','Normalized-L2','vvW2','Q-vvW2']
n=20; N=n*n; h=1/(n+1); q0=28.; nsrc=3; beta=2e-4
T=sp.diags([-np.ones(n-1),2*np.ones(n),-np.ones(n-1)],[-1,0,1],format='csr')/h**2
I=sp.eye(n,format='csr'); L=sp.kron(I,T)+sp.kron(T,I)
Tr=sp.diags([-np.ones(n-1),2*np.ones(n),-np.ones(n-1)],[-1,0,1],format='csr')
Lreg=sp.kron(I,Tr)+sp.kron(Tr,I)
x=np.linspace(h,1-h,n); X,Y=np.meshgrid(x,x,indexing='ij')
mtrue=(.9*np.exp(-((X-.34)**2+(Y-.38)**2)/.018)-.72*np.exp(-((X-.70)**2+(Y-.64)**2)/.025)).ravel()
srcs=[]
for cx,cy,sgn in [(.22,.25,1),(.75,.22,-1),(.48,.82,1)]:
    srcs.append((sgn*(np.exp(-((X-cx)**2+(Y-cy)**2)/.006)-.72*np.exp(-((X-(1-cx))**2+(Y-(1-cy))**2)/.010))).ravel())
srcs=np.array(srcs)

def fw(m):
    q=q0*np.exp(m); lu=spla.splu((L+sp.diags(q)).tocsc())
    U=np.stack([lu.solve(s) for s in srcs])
    return U,q,lu

Utrue,_,_=fw(mtrue); DC=Utrue.ravel()

def reg(m):
    return .5*beta*float(m@(Lreg@m))/N,beta*(Lreg@m)/N

class Obj:
    def __init__(self,d,kind):
        self.d=np.asarray(d,float); self.kind=kind
        if kind=='vvW2':
            self.st=fixed_scale_setup(self.d,eps=.12,data_shape=(nsrc,n,n),cross_weight=.2,between_block_weight=.12)
        elif kind=='Q-vvW2':
            self.st=frozen_setup(self.d,eps=.12,data_shape=(nsrc,n,n),cross_weight=.2,between_block_weight=.12)
        U,_,_=fw(np.zeros(N)); y=U.ravel()
        self.scale=max(abs(self.data(y)[0]),1e-30)
    def data(self,y):
        if self.kind=='L2': return l2_objective_grad(y,self.d)
        if self.kind=='Normalized-L2': return normalized_l2_objective_grad(y,self.d)
        if self.kind=='vvW2': return fixed_scale_objective_grad(y,self.st)
        if self.kind=='Q-vvW2': return frozen_objective_grad(y,self.st)
        raise KeyError(self.kind)
    def fg(self,m):
        U,q,lu=fw(m); v,xi=self.data(U.ravel())
        v/=self.scale; xi=xi/self.scale
        Xi=xi.reshape(nsrc,N); gm=np.zeros(N)
        for k in range(nsrc): gm+=-(q*U[k])*lu.solve(Xi[k])
        vr,gr=reg(m)
        return v+vr,gm+gr

def metrics(m):
    return float(np.linalg.norm(m-mtrue)/np.linalg.norm(mtrue)),float(np.corrcoef(m,mtrue)[0,1])

def solve(
    d,
    kind,
    maxiter=40,
    save=None,
    progress="compact",
    log_segments=4,
):
    obj = Obj(d, kind)
    hist = []
    iter_counter = {"k": 0}

    if progress != "silent":
        print(
            f"[E04 {kind}] start | maxiter={maxiter}"
            + (f" | case={save}" if save else ""),
            flush=True,
        )

    def f(m):
        v, g = obj.fg(m)
        hist.append(v)
        return v, g

    def callback(xk):
        iter_counter["k"] += 1
        k = iter_counter["k"]

        show = (
            progress == "full"
            or (
                progress == "compact"
                and progress_hit(
                    k,
                    maxiter,
                    segments=log_segments,
                )
            )
        )

        if show:
            re_i, corr_i = metrics(
                np.asarray(xk)
            )

            obj_i = (
                hist[-1]
                if hist
                else float("nan")
            )

            progress_print(
                f"E04 {kind}",
                k,
                maxiter,
                (
                    f"obj={obj_i:.3e} | "
                    f"RE={re_i:.4f} | "
                    f"corr={corr_i:.4f}"
                ),
            )

    st = time.perf_counter()

    r = minimize(
        f,
        np.zeros(N),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-1.2, 1.2)] * N,
        callback=callback,
        options={
            "maxiter": maxiter,
            "ftol": 1e-11,
            "gtol": 1e-7,
            "maxls": 25,
        },
    )

    sec = time.perf_counter() - st
    re, corr = metrics(r.x)

    out = {
        "method": kind,
        "relative_model_error": re,
        "correlation": corr,
        "iterations": int(r.nit),
        "function_evals": int(r.nfev),
        "runtime_s": sec,
        "success": bool(r.success),
        "initial_data_objective": obj.scale,
    }

    if save:
        np.save(
            RES / f"{save}_{kind}.npy",
            r.x.reshape(n, n),
        )

        pd.DataFrame(
            {
                "evaluation": np.arange(
                    1,
                    len(hist) + 1,
                ),
                "objective": hist,
            }
        ).to_csv(
            RES / f"{save}_{kind}_history.csv",
            index=False,
        )

    if progress != "silent":
        print(
            f"[E04 {kind}] done"
            f" | nit={r.nit}"
            f" | RE={re:.4f}"
            f" | corr={corr:.4f}"
            f" | time={sec:.1f}s",
            flush=True,
        )

    return out


def gradient_gates():
    rng=np.random.default_rng(13); m=.08*rng.normal(size=N); p=rng.normal(size=N); p/=np.linalg.norm(p)
    rows=[]
    for kind in METHODS:
        o=Obj(DC,kind); _,g=o.fg(m); an=float(g@p)
        for e in [1e-3,3e-4,1e-4,3e-5,1e-5]:
            vp=o.fg(m+e*p)[0]; vm=o.fg(m-e*p)[0]; fd=(vp-vm)/(2*e)
            rows.append([kind,e,fd,an,abs(fd-an)/max(abs(fd),abs(an),1e-14)])
    pd.DataFrame(rows,columns=['method','step','finite_difference','adjoint','relative_error']).to_csv(
        RES/'gradient_check.csv',index=False)

def core():
    print(
        "\n[E04] Core experiments",
        flush=True,
    )

    gradient_gates()

    rng = np.random.default_rng(3)

    z = rng.normal(size=DC.size)
    z *= (
        .05
        * np.linalg.norm(DC)
        / np.linalg.norm(z)
    )

    cases = {
        "clean_g1": DC,
        "clean_g05": .5 * DC,
        "clean_g2": 2 * DC,
        "noise5_g1": DC + z,
    }

    rows = []

    total = len(cases) * len(METHODS)
    count = 0

    for cn, d in cases.items():

        print(
            f"\n[E04] case={cn}",
            flush=True,
        )

        for method in METHODS:

            count += 1

            print(
                f"[E04 core] inversion "
                f"{count:02d}/{total:02d}"
                f" | {method}",
                flush=True,
            )

            o = solve(
                d,
                method,
                40,
                cn,
                progress="compact",
                log_segments=4,
            )

            o["case"] = cn
            rows.append(o)

    pd.DataFrame(rows).to_csv(
        RES / "core_summary.csv",
        index=False,
    )

    print(
        f"[E04] Core complete"
        f" | {total}/{total} inversions",
        flush=True,
    )


def mc(
    seed_start=0,
    seed_end=12,
):
    rows = []

    noise_levels = [
        .02,
        .05,
        .10,
    ]

    total = (
        len(noise_levels)
        * (seed_end - seed_start)
    )

    completed = 0

    print(
        f"\n[E04] Monte Carlo"
        f" | noise={len(noise_levels)}"
        f" | seeds={seed_end-seed_start}"
        f" | methods={len(METHODS)}",
        flush=True,
    )

    t_all = time.perf_counter()

    for nl in noise_levels:

        print(
            f"\n[E04] noise={nl:.0%}",
            flush=True,
        )

        for seed in range(
            seed_start,
            seed_end,
        ):

            t_seed = time.perf_counter()

            rg = np.random.default_rng(
                1000 + seed
            )

            z = rg.normal(
                size=DC.size
            )

            z *= (
                nl
                * np.linalg.norm(DC)
                / np.linalg.norm(z)
            )

            d = DC + z

            for method in METHODS:

                o = solve(
                    d,
                    method,
                    40,
                    progress="silent",
                )

                rows.append(
                    {
                        "noise_relative": nl,
                        "seed": seed,
                        **o,
                    }
                )

            completed += 1

            progress_print(
                "E04 MC",
                completed,
                total,
                (
                    f"noise={nl:.0%}"
                    f" | seed={seed:02d}"
                    f" | {len(METHODS)} methods"
                    f" | "
                    f"{time.perf_counter()-t_seed:.1f}s"
                ),
            )

    pd.DataFrame(rows).to_csv(
        RES
        / f"mc_{seed_start:02d}_{seed_end:02d}.csv",
        index=False,
    )

    print(
        f"[E04] Monte Carlo complete"
        f" | inversions={len(rows)}"
        f" | time="
        f"{(time.perf_counter()-t_all)/60:.1f} min",
        flush=True,
    )


def finalize():
    files=sorted(RES.glob('mc_*.csv'))
    df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True).drop_duplicates(
        ['noise_relative','seed','method']).sort_values(['noise_relative','seed','method'])
    df.to_csv(RES/'noise_monte_carlo.csv',index=False)
    s=df.groupby(['noise_relative','method']).agg(
        mean_error=('relative_model_error','mean'),std_error=('relative_model_error','std'),
        median_error=('relative_model_error','median'),mean_corr=('correlation','mean')).reset_index()
    s.to_csv(RES/'noise_monte_carlo_summary.csv',index=False)
    gc=pd.read_csv(RES/'gradient_check.csv'); issues=[]
    for method in METHODS:
        if gc[gc.method==method].relative_error.min()>1e-6: issues.append('gradient '+method)
    if len(df)!=144: issues.append(f'MC count {len(df)}')
    core_df=pd.read_csv(RES/'core_summary.csv')
    if len(core_df)!=16: issues.append(f'core count {len(core_df)}')
    out={'status':'PASS' if not issues else 'FAIL','issues':issues,'mc_rows':len(df),
         'methods':METHODS,'core_rows':len(core_df)}
    (RES/'VALIDATION.json').write_text(json.dumps(out,indent=2))
    print(s.to_string(index=False)); print(out)
    if issues: raise SystemExit(2)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--core',action='store_true')
    p.add_argument('--mc',nargs=2,type=int); p.add_argument('--finalize',action='store_true'); a=p.parse_args()
    if a.core: core()
    if a.mc: mc(*a.mc)
    if a.finalize: finalize()
