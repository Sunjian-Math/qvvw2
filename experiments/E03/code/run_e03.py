from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import sys,time,json,argparse
from pathlib import Path
import numpy as np,pandas as pd, torch
HERE=Path(__file__).resolve().parent
from experiments.E03.code import distributed_wave_model as M
from experiments.E03.code.hv_batch_reimpl import HVBatch
from qvvw2.baselines.uot import li_uot_value_grad_batch_fast
from qvvw2.baselines.softplus_w2sq import beta_from_observed, batch_value_grad_exact
from qvvw2.registry import METHODS, DISPLAY_NAMES as DISPLAY
from qvvw2.progress import progress_hit, progress_print
# Sambridge et al. (2022) is loaded from the verified external waveform-ot snapshot.
from qvvw2.baselines.sambridge2022_marginal_w2sq import ricker_util as RU

SCENARIOS=['clean','gain_only','source_mismatch','gain_noise']

CONSOLE_DISPLAY = {
    "L2": "L^2",
    "Normalized-L2": "Normalized L^2",
    "Softplus-W2sq": "Softplus-W_2^2",
    "Marginal-W2sq": "Marginal-W_2^2",
    "UOT": "UOT",
    "HV": "HV metric",
    "Q-vvW2": "Q-vvW_2",
}

def _load_summary(outroot, P, scenario, method):
    path=Path(outroot)/f"P{P}"/scenario/method/"summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())

def _relative_runtime(outroot, P, scenario, method_time):
    l2=_load_summary(outroot,P,scenario,"L2")
    if not l2:
        return None
    base=float(l2.get("algorithm_time_s",0.0))
    if base<=0:
        return None
    return float(method_time)/base

def print_run_metrics(summary, outroot):
    reltime=_relative_runtime(outroot,summary["P"],summary["scenario"],summary["algorithm_time_s"] )
    print("\n"+"="*74)
    print("E03 TERMINAL METRICS")
    print(f"Method                    : {CONSOLE_DISPLAY[summary['method']]}")
    print(f"Scenario                  : {summary['scenario']}")
    print(f"Unknowns                  : {summary['n_unknowns']} (P={summary['P']})")
    print(f"Outer updates             : {summary['steps']}")
    print(f"Relative model error      : {summary['relative_model_error']:.9f}")
    print(f"Model correlation         : {summary['correlation']:.9f}")
    print(f"Common standardized resid.: {summary['common_std_residual']:.9f}")
    print(f"Algorithm time            : {summary['algorithm_time_s']:.6f} s")
    if reltime is None:
        print("Relative runtime vs L^2   : n/a (L^2 summary not available yet)")
    else:
        print(f"Relative runtime vs L^2   : {reltime:.4f}x")
    print("="*74,flush=True)

def build_quantitative_table(outroot, P=8, scenario="clean", write_csv=True):
    rows=[]
    l2=_load_summary(outroot,P,scenario,"L2")
    if l2 is None:
        raise FileNotFoundError(f"Missing L2 summary for P{P}/{scenario}")
    tbase=float(l2["algorithm_time_s"])
    for method in METHODS:
        x=_load_summary(outroot,P,scenario,method)
        if x is None:
            raise FileNotFoundError(f"Missing summary for P{P}/{scenario}/{method}")
        rows.append({
            "method":method,
            "display_name":CONSOLE_DISPLAY[method],
            "relative_model_error":float(x["relative_model_error"]),
            "correlation":float(x["correlation"]),
            "common_std_residual":float(x["common_std_residual"]),
            "algorithm_time_s":float(x["algorithm_time_s"]),
            "relative_runtime_vs_L2":float(x["algorithm_time_s"])/tbase,
        })
    df=pd.DataFrame(rows)
    if write_csv:
        out=Path(outroot)/f"P{P}_{scenario}_quantitative_metrics.csv"
        df.to_csv(out,index=False)
    return df

def print_quantitative_table(outroot, P=8, scenario="clean"):
    df=build_quantitative_table(outroot,P,scenario,write_csv=True)
    print("\n"+"="*112)
    print(f"E03 QUANTITATIVE COMPARISON | P={P} ({P*P} unknowns) | scenario={scenario}")
    print("Timing field: algorithm_time_s (accumulated measured algorithm time); relative runtime is normalized by L^2.")
    print("-"*112)
    print(f"{'Method':<24}{'Rel. model error':>18}{'Correlation':>15}{'Std. residual':>16}{'Time (s)':>14}{'vs L^2':>12}")
    print("-"*112)
    for _,r in df.iterrows():
        print(f"{r['display_name']:<24}{r['relative_model_error']:>18.6f}{r['correlation']:>15.6f}{r['common_std_residual']:>16.6f}{r['algorithm_time_s']:>14.3f}{r['relative_runtime_vs_L2']:>11.3f}x")
    print("="*112,flush=True)
    return df


def scenario_data(name):
    if name=='clean': return M.d.copy(),7.0
    if name=='gain_only': return 1.8*M.d.copy(),7.0
    if name=='source_mismatch': return M.d.copy(),6.6
    if name=='gain_noise':
        rg=np.random.default_rng(11);base=1.8*M.d.copy();eta=rg.normal(size=base.shape);eta*=.04*np.linalg.norm(base)/np.linalg.norm(eta);return base+eta,7.0
    raise ValueError(name)

class Obj:
    def __init__(self,method,dobs):
        self.method=method; self.d=np.asarray(dobs,float); self.t=np.asarray(M.Tnp,float)
        if method=='Softplus-W2sq': self.beta=beta_from_observed(self.d, beta_unit=2.0)
        if method=='Q-vvW2': self.qst=M.qsetup(self.d,eps=.15,cross=.08,rec_w=.08,source_w=.03)
        if method=='UOT': self.x=np.linspace(0,1,self.d.shape[-1]); self.k=1.5/max(float(np.max(self.d)),1e-12)
        if method=='Marginal-W2sq':
            amax=1.5*max(abs(float(self.d.min())),abs(float(self.d.max())),1e-6)
            self.grid=(float(self.t[0]),float(self.t[-1]),-amax,amax,36,len(self.t)); self.targets=[]
            for tr in self.d.reshape(-1,self.d.shape[-1]):
                _,o=RU.BuildOTobjfromWaveform(self.t,tr,self.grid,lambdav=.03,theta=45.);self.targets.append(o)
    def vg(self,y):
        y=np.asarray(y,float)
        if self.method=='L2':
            v,g=M.l2_objective_grad(y.ravel(),self.d.ravel());return float(v),g.reshape(y.shape),{}
        if self.method=='Normalized-L2':
            v,g=M.normalized_l2_objective_grad(y.ravel(),self.d.ravel());return float(v),g.reshape(y.shape),{}
        if self.method=='Softplus-W2sq':
            v,g,meta=batch_value_grad_exact(y,self.d,self.t,self.beta,reduction='mean');return float(v),g.reshape(y.shape),meta
        if self.method=='Q-vvW2':
            v,g=M.qog(y,self.qst);return float(v),g.reshape(y.shape),{}
        if self.method=='UOT':
            vals,gr,meta=li_uot_value_grad_batch_fast(y.reshape(-1,y.shape[-1]),self.d.reshape(-1,self.d.shape[-1]),self.x,k=self.k,shape_iters=2000)
            return float(np.mean(vals)),(gr/len(vals)).reshape(y.shape),meta
        if self.method=='Marginal-W2sq':
            vals=[];gr=[]
            for i,tr in enumerate(y.reshape(-1,y.shape[-1])):
                wf,o=RU.BuildOTobjfromWaveform(self.t,tr,self.grid,lambdav=.03,deriv=True,theta=45.)
                try:
                    v,g,_=RU.CalcWasserWaveform(o,self.targets[i],wf,distfunc='W2',deriv=True)
                    gg=np.nan_to_num(np.asarray(g,float),nan=0.,posinf=0.,neginf=0.)
                except Exception:
                    # objective remains source-faithful; zero is the stable subgradient at a nondifferentiable fingerprint-grid contact.
                    v=RU.CalcWasserWaveform(o,self.targets[i],wf,distfunc='W2',deriv=False);gg=np.zeros_like(tr)
                vals.append(float(v));gr.append(gg)
            return float(np.mean(vals)),(np.stack(gr)/len(vals)).reshape(y.shape),{'n_traces':len(vals)}
        raise ValueError(self.method)

def regularizer(m,P):
    lam=.03*((P-1)/7.)**2
    return lam*(((m[1:]-m[:-1])**2).mean()+((m[:,1:]-m[:,:-1])**2).mean())+.0005*(m*m).mean()

def run_one(method,scenario='clean',P=8,steps=25,lr=.04,outroot=None,start_step=0):
    M.P=P;dobs,f_inv=scenario_data(scenario); outroot=Path(outroot or (HERE.parent/'results'))
    od=outroot/f'P{P}'/scenario/method;od.mkdir(parents=True,exist_ok=True);(od/'checkpoints').mkdir(exist_ok=True)
    if method=='HV': obj=HVBatch(dobs,workers=4,params={'kappa':1e-4,'lam':1e-4,'eps':1e-12,'nt':12,'maxiter':1800,'gtol':1e-10},warm_start=True)
    else: obj=Obj(method,dobs)
    m=torch.zeros((P,P),dtype=M.DTYPE,requires_grad=True); opt=torch.optim.Adam([m],lr=lr,betas=(.9,.999),eps=1e-8)
    hist=[]
    if start_step>0:
        m0=np.load(od/'checkpoints'/f'latent_iter_{start_step:04d}.npy');m.data.copy_(torch.tensor(m0,dtype=M.DTYPE));opt.load_state_dict(torch.load(od/'checkpoints'/f'adam_state_{start_step:04d}.pt',weights_only=False))
        hp=od/'checkpoints'/f'hv_cache_{start_step:04d}.npz'
        if method=='HV' and hp.exists():
            z=np.load(hp,allow_pickle=True);obj.cache=[z[str(i)] for i in range(len(obj.D))]
        hp0=od/'history.csv'
        if hp0.exists():hist=pd.read_csv(hp0).to_dict('records')
    # freeze normalization at common initial model, independent of restart point
    with torch.no_grad(): z=torch.zeros((P,P),dtype=M.DTYPE);c0=M.cfrom(z);dt0,yf0=M.sim(c0,f=f_inv);y0=M.rs(yf0,dt0).numpy()
    if method=='HV':
        # do not overwrite a resumed path cache while evaluating m0
        tmp_cache=obj.cache;obj.cache=[None]*len(obj.D);data0,g0,meta0=obj.vg(y0,update_cache=False);obj.cache=tmp_cache
    else:data0,g0,meta0=obj.vg(y0)
    scale=max(abs(float(data0)),1e-30);t0=time.perf_counter()
    for it in range(start_step+1,steps+1):
        opt.zero_grad(set_to_none=True);c=M.cfrom(m);dt,yf=M.sim(c,f=f_inv);y=M.rs(yf,dt);yn=y.detach().numpy()
        raw,gy,meta=obj.vg(yn)
        current_res=float(np.linalg.norm(yn-dobs)/max(np.linalg.norm(dobs),1e-30))
        y.backward(gradient=torch.tensor(gy/scale,dtype=M.DTYPE),retain_graph=True); gd=m.grad.detach().clone();m.grad=None
        rr=regularizer(m,P);rr.backward();m.grad += gd;gn=float(torch.linalg.norm(m.grad));total=float(raw/scale+rr.detach());opt.step()
        with torch.no_grad():m.clamp_(-1,1);cc=M.cfrom(m);rel,corr=M.metrics(cc.numpy())
        row={'iteration':it,'raw_data':float(raw),'normalized_data':float(raw/scale),'reg':float(rr.detach()),'total_preupdate':total,'gradnorm':gn,'rel_error_postupdate':float(rel),'corr_postupdate':float(corr),'common_std_residual_preupdate':current_res,**meta};hist.append(row)
        np.save(od/'checkpoints'/f'model_iter_{it:04d}.npy',cc.numpy());np.save(od/'checkpoints'/f'latent_iter_{it:04d}.npy',m.detach().numpy());torch.save(opt.state_dict(),od/'checkpoints'/f'adam_state_{it:04d}.pt')
        if method=='HV':np.savez_compressed(od/'checkpoints'/f'hv_cache_{it:04d}.npz',**{str(i):v for i,v in enumerate(obj.cache)})
        pd.DataFrame(hist).to_csv(od/'history.csv',index=False)
        (od/'run_state.json').write_text(json.dumps({'completed_steps':it,'target_steps':steps,'method':method,'scenario':scenario,'P':P},indent=2))
        if progress_hit(it, steps, segments=5):
            progress_print(
                f"E03 {method} {scenario} P={P}",
                it,
                steps,
                (
                    f"RE={rel:.4e} | "
                    f"corr={corr:.4f} | "
                    f"obj={total:.4e} | "
                    f"|g|={gn:.2e}"
                ),
            )
    elapsed=time.perf_counter()-t0
    if method=='HV':obj.close()
    with torch.no_grad():cfin=M.cfrom(m).numpy();dt,yf=M.sim(torch.tensor(cfin,dtype=M.DTYPE),f=f_inv);pred=M.rs(yf,dt).numpy();rel,corr=M.metrics(cfin)
    np.save(od/'final_model.npy',cfin);np.save(od/'final_pred.npy',pred);np.save(od/'observed.npy',dobs)
    oldelapsed=0.0
    if (od/'summary_partial_time.json').exists():oldelapsed=float(json.loads((od/'summary_partial_time.json').read_text()).get('algorithm_time_s',0.))
    elapsed_total=oldelapsed+elapsed
    if steps<25:(od/'summary_partial_time.json').write_text(json.dumps({'algorithm_time_s':elapsed_total},indent=2))
    summ={'method':method,'scenario':scenario,'P':P,'n_unknowns':P*P,'steps':steps,'optimizer':'Adam','learning_rate':lr,'initial_data_objective':float(data0),'normalization':'J_k(m)/J_k(m0); data gradient divided by J_k(m0)','regularizer':'continuum-consistent H1 finite-difference regularizer, lambda=.03*((P-1)/7)^2, plus .0005 mean(m^2)','relative_model_error':float(rel),'correlation':float(corr),'common_std_residual':float(np.linalg.norm(pred-dobs)/max(np.linalg.norm(dobs),1e-30)),'algorithm_time_s':float(elapsed_total),'initial_meta':meta0}
    (od/'summary.json').write_text(json.dumps(summ,indent=2))
    print_run_metrics(summ,outroot)
    return summ

def fd_gate(method,P=8,seed=123):
    M.P=P;dobs,f_inv=scenario_data('clean');
    if method=='HV': obj=HVBatch(dobs,workers=4,params={'kappa':1e-4,'lam':1e-4,'eps':1e-12,'nt':12,'maxiter':500,'gtol':5e-8},warm_start=False)
    else: obj=Obj(method,dobs)
    rng=np.random.default_rng(seed);m0=np.zeros((P,P));h=rng.normal(size=(P,P));h/=np.linalg.norm(h)
    def fg(arr,wantg=True):
        mt=torch.tensor(arr,dtype=M.DTYPE,requires_grad=wantg);c=M.cfrom(mt);dt,yf=M.sim(c,f=f_inv);y=M.rs(yf,dt);yn=y.detach().numpy();raw,gy,meta=obj.vg(yn)
        if not wantg:return float(raw)
        y.backward(gradient=torch.tensor(gy,dtype=M.DTYPE));return float(raw),mt.grad.detach().numpy(),meta
    val,g,meta=fg(m0,True);an=float(np.sum(g*h));rows=[]
    for e in [1e-3,3e-4,1e-4]:
        fd=(fg(m0+e*h,False)-fg(m0-e*h,False))/(2*e);rows.append({'eps':e,'fd':fd,'analytic':an,'relative_error':abs(fd-an)/max(abs(fd),abs(an),1e-14)})
    if method=='HV':obj.close()
    return {'method':method,'initial_objective':val,'meta':meta,'rows':rows,'best_relative_error':min(r['relative_error'] for r in rows)}

def run_all(outroot):
    outroot=Path(outroot);outroot.mkdir(parents=True,exist_ok=True)
    gates=[]
    for m in METHODS:
        g=fd_gate(m);gates.append(g);print('GATE',m,g['best_relative_error'],flush=True)
    (outroot/'gradient_gates.json').write_text(json.dumps(gates,indent=2))
    summaries=[]
    for sc in SCENARIOS:
        for m in METHODS:summaries.append(run_one(m,sc,8,outroot=outroot))
    for P in [12,16]:
        for m in METHODS:summaries.append(run_one(m,'clean',P,outroot=outroot))
    df=pd.DataFrame(summaries)
    # Add the relative runtime against L^2 for every (P, scenario) group.
    reltimes=[]
    for _,row in df.iterrows():
        base=df[(df.P==row.P)&(df.scenario==row.scenario)&(df.method=='L2')].algorithm_time_s
        reltimes.append(float(row.algorithm_time_s)/float(base.iloc[0]) if len(base) else np.nan)
    df['relative_runtime_vs_L2']=reltimes
    df.to_csv(outroot/'summary_all_runs.csv',index=False)
    print('COMPLETE',len(summaries),'runs',flush=True)
    print_quantitative_table(outroot,8,'clean')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--all',action='store_true',help='Run all 42 formal E03 inversions and print the final P8-clean quantitative table.')
    ap.add_argument('--print-summary',action='store_true',help='Print saved terminal metrics without rerunning the inversions.')
    ap.add_argument('--method',choices=METHODS)
    ap.add_argument('--scenario',default='clean',choices=SCENARIOS)
    ap.add_argument('--P',type=int,default=8)
    ap.add_argument('--start-step',type=int,default=0)
    ap.add_argument('--steps',type=int,default=25)
    ap.add_argument('--outroot',default=str(HERE.parent/'results'))
    args=ap.parse_args()
    if args.print_summary:
        print_quantitative_table(args.outroot,args.P,args.scenario)
    elif args.all:
        run_all(args.outroot)
    elif args.method:
        run_one(args.method,args.scenario,args.P,steps=args.steps,outroot=args.outroot,start_step=args.start_step)
    else:
        ap.error('choose --all, --print-summary, or --method METHOD')
