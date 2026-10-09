"""Generate paper figures from freshly computed experimental outputs (not frozen data)."""
from pathlib import Path
import sys,json,argparse
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import PchipInterpolator
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qvvw2.core import lift,jacobian_dense
from qvvw2.registry import METHODS
R=ROOT/'reference'
COL={'L2':'#2563EB','Normalized-L2':'#E97716','Softplus-W2sq':'#159447','Marginal-W2sq':'#D83A3A','UOT':'#7853B8','Q-vvW2':'#111111','vvW2':'#159447','Fixed-vvW2':'#159447','Lift-Euclidean':'#D83A3A'}
LAB={'L2':r'$L^2$','Normalized-L2':r'Normalized $L^2$','Softplus-W2sq':r'Softplus-$W_2^2$','Marginal-W2sq':r'Marginal-$W_2^2$','UOT':'UOT','Q-vvW2':r'Q-vv$W_2$','vvW2':'Fixed-scale','Fixed-vvW2':'Fixed-scale','Lift-Euclidean':'Lift–Euclidean'}
SC=['clean','gain_only','source_mismatch','gain_noise'];SCL=['Clean','Gain','Source mismatch','Gain + noise']
CONTROL=['Normalized-L2','Lift-Euclidean','Fixed-vvW2','Q-vvW2'];FIELD=['L2','Normalized-L2','vvW2','Q-vvW2']
from scripts import plot_style_stix as locked_style
locked_style.init_locked_style()
plt.rcParams['lines.linewidth'] = 1.8
plt.rcParams.update({'font.size':12, 'axes.labelsize':12, 'axes.titlesize':12.5, 'xtick.labelsize':12, 'ytick.labelsize':12, 'legend.fontsize':12})

PROV=[];OUT=None
def read(p):
 p=R/p;PROV.append(p);return pd.read_csv(p)
def line(ax,x,y,m,**kw):return ax.plot(x,y,color=COL[m],lw=2.1 if m=='Q-vvW2' else 1.7,label=LAB[m],**kw)
def title(ax,i,s):ax.set_title(f'({chr(97+i)}) {s}',pad=9)
def save(fig,num):
 colorbars=[ax for ax in fig.axes if hasattr(ax,'_colorbar')]
 main=[ax for ax in fig.axes if ax not in colorbars]
 # Match actual plot rectangles, including fixed-aspect image panels.
 fig.canvas.draw()
 widths=[ax.get_position().width for ax in main]; heights=[ax.get_position().height for ax in main]
 w,h=min(widths),min(heights)
 for ax in main:
  pos=ax.get_position()
  ax.set_position([pos.x0+(pos.width-w)/2,pos.y0+(pos.height-h)/2,w,h])
  locked_style.style_axis(ax,minor=False)
  # Matplotlib creates off-range tick artists that are not painted. Exclude
  # their labels explicitly so the template audits displayed text only.
  for axis,limits in ((ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())):
   lo,hi=sorted(limits)
   for tick in axis.get_major_ticks()+axis.get_minor_ticks():
    if tick.get_loc()<lo or tick.get_loc()>hi:
     tick.label1.set_visible(False);tick.label2.set_visible(False)
 for ax in colorbars:
  ax.yaxis.label.set_fontsize(9)
  for axis,limits in ((ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())):
   lo,hi=sorted(limits)
   for tick in axis.get_major_ticks()+axis.get_minor_ticks():
    if tick.get_loc()<lo or tick.get_loc()>hi:
     tick.label1.set_visible(False);tick.label2.set_visible(False)
 for text in fig.findobj(match=matplotlib.text.Text):
  if text.get_visible():
   original_size=text.get_fontsize()
   text.set_fontproperties(locked_style.EN_BOLD)
   text.set_fontsize(original_size)
   text.set_math_fontfamily("custom")
   text.set_fontsize(max(15.0 if num==10 else 12.0,min(text.get_fontsize(),16.0 if num==10 else 13.0)))
 locked_style.save_final_figure(fig,OUT/f'figure{num:02d}',
                                main_axes=main,colorbar_axes=colorbars,
                                dpi=220,run_pdf_render=False)
 plt.close(fig)

def legend(fig,axes,ncol=3):
 handles,labels=axes.get_legend_handles_labels();fig.legend(handles,labels,ncol=ncol,loc='lower center',frameon=False,bbox_to_anchor=(.5,.005))
def fig1():
 t=np.linspace(0,1,128);d=np.exp(-.5*((t-.29)/.06)**2)-.65*np.exp(-.5*((t-.71)/.08)**2);rho=lift(d)[0];J=jacobian_dense(d)
 h=np.gradient(d,t);h-=d*(d@h)/(d@d);h/=np.linalg.norm(h);rad=d/np.linalg.norm(d)
 f,a=plt.subplots(2,2,figsize=(8,5.5));a=a.ravel()
 title(a[0],0,'Common positive gain');
 for gain,c in zip([.5,1,2],['#A9A9A9','#2563EB','#111111']):a[0].plot(t,gain*d,color=c,label=fr'$a={gain:g}$')
 a[0].set(xlabel='Time',ylabel='Amplitude');a[0].legend(frameon=False)
 title(a[1],1,'Scale-invariant species');a[1].plot(t,rho[:128],color='#2563EB',label=r'$\rho^+$');a[1].plot(t,rho[128:],color='#E97716',label=r'$\rho^-$');a[1].set(xlabel='Time',ylabel='Mass');a[1].legend(frameon=False)
 title(a[2],2,'Polarity is retained');neg=lift(-d)[0];a[2].plot(t,rho[:128],color='#2563EB',label=r'$\rho^+(d)$');a[2].plot(t,neg[:128],color='#E97716',label=r'$\rho^+(-d)$');a[2].set(xlabel='Time',ylabel='Mass');a[2].legend(frameon=False)
 title(a[3],3,'Radial and transverse directions');a[3].plot(t,h,color='#2563EB',label='Transverse input');a[3].plot(t,rad,color='#A0A0A0',label='Radial input');a[3].set(xlabel='Time',ylabel='Unit direction');a[3].legend(frameon=False)
 f.tight_layout(h_pad=2,w_pad=2);save(f,1)
def fig2():
 scale=read('E01/scale_invariance.csv');eig=read('E01/pullback_eigenvalues.csv');resp=read('E01/directional_response.csv');exp=read('E01/local_expansion.csv')
 summary=json.loads((R/'E01/summary.json').read_text());PROV.append(R/'E01/summary.json')
 f,a=plt.subplots(2,2,figsize=(8,5.6));a=a.ravel()
 title(a[0],0,'Lift invariance');a[0].loglog(scale.scale,np.maximum(scale.lift_gap,1e-19),color='#2563EB');a[0].set(xlabel='Positive gain',ylabel='Lift difference\n(Euclidean norm)')
 title(a[1],1,'Quotient tensor spectrum');vals=np.abs(eig.eigenvalue.to_numpy());a[1].semilogy(np.arange(1,len(vals)+1),np.maximum(vals,1e-18),'o',ms=3,color='#2563EB');a[1].plot(1,max(vals[0],1e-18),'o',ms=5,color='black');a[1].set(xlabel='Eigenvalue index',ylabel='Absolute eigenvalue')
 title(a[2],2,'Directional response')
 # Column names are fixed by the structural experiment.
 for name,g in resp.groupby('direction',sort=False):a[2].loglog(g.relative_perturbation,np.maximum(g.Q_objective,1e-32),label=name)
 a[2].set(xlabel='Perturbation magnitude',ylabel='Frozen objective');a[2].legend(frameon=False,fontsize=8)
 title(a[3],3,'Frozen Taylor remainder');a[3].loglog(exp.step,np.maximum(exp.remainder,1e-30),'o-',ms=3,color='black',label=f"Slope {summary['local_remainder_loglog_slope']:.3f}");a[3].set(xlabel='Step size',ylabel='Absolute remainder');a[3].legend(frameon=False)
 f.tight_layout(h_pad=2,w_pad=2);save(f,2)
def fig3():
 f,a=plt.subplots(2,2,figsize=(8,5.5));a=a.ravel()
 for i,(sweep,xlab,ref) in enumerate([('gain','Positive gain',1),('ratio','Relative negative amplitude',.65),('shift','Time shift',0),('signed_gain','Signed gain',1)]):
  df=read('E02/'+sweep+'.csv');title(a[i],i,['Positive gain','Relative amplitude','Time shift','Polarity'][i]);x=df.parameter.to_numpy()
  for m in METHODS:
   y=df[m].to_numpy()
   if sweep=='gain':y=np.abs(y-df[m].iloc[np.argmin(abs(x-1))]);a[i].plot(x,np.maximum(y,1e-32),color=COL[m],label=LAB[m],lw=2.1 if m=='Q-vvW2' else 1.7);a[i].set_yscale('log');a[i].set_xscale('log')
   else:
    norm=max(np.ptp(y),1e-30);yp=(y-y.min())/norm
    if sweep=='signed_gain':
     # Zero input is outside the quotient domain; do not bridge across it.
     split=int(np.flatnonzero(x>0)[0])
     line(a[i],np.insert(x,split,np.nan),np.insert(yp,split,np.nan),m)
    else:line(a[i],x,yp,m)
  a[i].axvline(ref,color='.5',ls='--',lw=1);a[i].set(xlabel=xlab,ylabel='Absolute objective change' if sweep=='gain' else 'Within-method normalized\nobjective')
 f.tight_layout(h_pad=2,w_pad=2,rect=(0,.14,1,1));legend(f,a[0]);save(f,3)
def wave_truth(n=49):
 x=np.linspace(0,1,n);X,Y=np.meshgrid(x,x,indexing='ij');return 1+.105*np.exp(-((X-.36)**2+(Y-.48)**2)/(2*.095**2))-.085*np.exp(-((X-.66)**2+(Y-.61)**2)/(2*.105**2))
def fig4():
 fields=[wave_truth(),np.ones((49,49))];labels=['True speed','Initial speed']
 for m in METHODS:
  p=R/'E03/original/P8/clean'/m/'final_model.npy';PROV.append(p);fields.append(np.load(p));labels.append(LAB[m])
 span=max(np.max(abs(v-1)) for v in fields);norm=TwoSlopeNorm(vcenter=1,vmin=1-span,vmax=1+span)
 f,a=plt.subplots(2,4,figsize=(8.6,4.5));a=a.ravel()
 for i,(ax,v,label) in enumerate(zip(a,fields,labels)):
  im=ax.imshow(v.T,origin='lower',extent=[0,1,0,1],cmap='RdBu_r',norm=norm);title(ax,i,label);ax.set(xticks=[0,.5,1],yticks=[0,.5,1]);ax.set_xlabel(r'$x_1$')
  if i%4==0:ax.set_ylabel(r'$x_2$')
 f.subplots_adjust(left=.065,right=.86,bottom=.12,top=.93,wspace=.28,hspace=.40);cax=f.add_axes([.89,.17,.017,.65]);f.colorbar(im,cax=cax,label='Wave speed');save(f,4)
def fig5():
 f,a=plt.subplots(2,2,figsize=(8,5.5));a=a.ravel()
 for i,sc in enumerate(SC):
  title(a[i],i,SCL[i])
  for m in METHODS:
   df=read(f'E03/budget/P8/{sc}/{m}/history.csv');assert len(df)==100
   line(a[i],df.iteration,df.rel_error_postupdate,m)
  for step in [25,50,100]:a[i].axvline(step,color='.8',lw=.7,ls=':')
  a[i].set(xlabel='Outer updates',ylabel='Relative model error',xlim=(0,102))
 f.tight_layout(h_pad=2,w_pad=2,rect=(0,.14,1,1));legend(f,a[0]);save(f,5)
def fig6():
 df=read('mechanism/mechanism_comparison.csv');ens=read('mechanism/all_metrics.csv');ens=ens[ens.group=='ensemble']
 f,a=plt.subplots(1,2,figsize=(8.2,3.7));title(a[0],0,'Component controls, 100 updates')
 for m in CONTROL:
  g=df[df.method==m].set_index('scenario');line(a[0],np.arange(4),g.loc[SC].relative_model_error.to_numpy(),m,marker='o',ms=4)
 a[0].set(xticks=np.arange(4),xticklabels=['Clean','Gain','Source\nmismatch','Gain +\nnoise'],ylabel='Relative model error')
 title(a[1],1,'Paired 4% noise ensembles')
 for group,sc in enumerate(['gain_noise','correlated_noise']):
  for j,m in enumerate(CONTROL):
   vals=ens[(ens.scenario==sc)&(ens.method==m)].relative_model_error.to_numpy();assert len(vals)==10
   xpos=group*5+j;a[1].scatter(xpos+np.linspace(-.10,.10,10),vals,s=10,color=COL[m],alpha=.4)
   a[1].errorbar(xpos,vals.mean(),yerr=vals.std(ddof=1),fmt='o',ms=4,color=COL[m],capsize=3,lw=1.4)
 a[1].set(xticks=[1.5,6.5],xticklabels=['Independent','Correlated'],ylabel='Relative model error')
 f.tight_layout(w_pad=2,rect=(0,.14,1,1));legend(f,a[0],ncol=4);save(f,6)
def fig7():
 df=read('mechanism/optimizer_robustness.csv');f,a=plt.subplots(1,2,figsize=(8,3.7))
 for i,sc in enumerate(['clean','gain_noise']):
  title(a[i],i,SCL[SC.index(sc)])
  for m in CONTROL:
   g=df[(df.scenario==sc)&(df.method==m)].sort_values('initial');line(a[i],g.initial,g.relative_model_error,m,marker='o',ms=4)
  a[i].set(xticks=[0,1,2],xticklabels=['Zero','Positive mode','Negative mode'],ylabel='Relative model error')
 f.tight_layout(w_pad=2,rect=(0,.14,1,1));legend(f,a[0],ncol=4);save(f,7)
def elliptic_truth():
 x=np.linspace(1/21,20/21,20);X,Y=np.meshgrid(x,x,indexing='ij');return .9*np.exp(-((X-.34)**2+(Y-.38)**2)/.018)-.72*np.exp(-((X-.70)**2+(Y-.64)**2)/.025)
def fig8():
 f,a=plt.subplots(2,5,figsize=(8.6,4.0));allfields=[]
 for row,case in enumerate(['clean_g1','noise5_g1']):
  fields=[elliptic_truth()]
  for m in FIELD:
   p=R/'E04'/f'{case}_{m}.npy';PROV.append(p);fields.append(np.load(p).reshape(20,20))
  allfields.append(fields)
 span=max(np.max(abs(v)) for fields in allfields for v in fields)
 for row,fields in enumerate(allfields):
  for col,v in enumerate(fields):
   ax=a[row,col];i=row*5+col;im=ax.imshow(v.T,origin='lower',extent=[1/42,1-1/42,1/42,1-1/42],vmin=-span,vmax=span,cmap='RdBu_r');title(ax,i,'Truth' if col==0 else LAB[FIELD[col-1]]);ax.set(xticks=[0,1],yticks=[0,1],xlim=(0,1),ylim=(0,1))
   if row==1:ax.set_xlabel(r'$x_1$')
   if col==0:ax.set_ylabel('Clean\n'+r'$x_2$' if row==0 else '5% noise\n'+r'$x_2$')
 f.subplots_adjust(left=.065,right=.86,bottom=.12,top=.93,wspace=.35,hspace=.38);cax=f.add_axes([.89,.18,.017,.64]);f.colorbar(im,cax=cax,label='Log-coefficient');save(f,8)
def fig9():
 df=read('E04/core_summary.csv');mc=read('E04/noise_monte_carlo.csv');f,a=plt.subplots(1,2,figsize=(8,3.7))
 title(a[0],0,'Common-gain response');title(a[1],1,'Paired additive noise')
 for m in FIELD:
  y=[df[(df.method==m)&(df.case==case)].relative_model_error.iloc[0] for case in ['clean_g05','clean_g1','clean_g2']];line(a[0],[.5,1,2],y,m,marker='o',ms=4)
  means=[];sds=[]
  for j,noise in enumerate([.02,.05,.10]):
   vals=mc[(mc.method==m)&(mc.noise_relative==noise)].relative_model_error.to_numpy();assert len(vals)==12;means.append(vals.mean());sds.append(vals.std(ddof=1))
   a[1].scatter(100*noise+np.linspace(-.18,.18,12),vals,color=COL[m],s=8,alpha=.25)
  a[1].errorbar([2,5,10],means,yerr=sds,fmt='o-',color=COL[m],ms=4,capsize=3,lw=1.7,label=LAB[m])
 a[0].set(xlabel='Observation gain',ylabel='Relative model error',yscale='log',xticks=[.5,1,2]);a[1].set(xlabel='Relative noise (%)',ylabel='Relative model error',xticks=[2,5,10]);f.tight_layout(w_pad=2,rect=(0,.14,1,1));legend(f,a[1],ncol=4);save(f,9)
def fig10():
 obj=read('E05/objectives.csv');ms=read('E05/multistart.csv');summary=read('E05/summary.csv').set_index('method');p=R/'E05/tdem_traces.npz';PROV.append(p);z=np.load(p);t=z['T'];obs=z['observed'];depths=z['depths'];Y=z['candidates']
 f,a=plt.subplots(6,3,figsize=(8.2,9.0));amp=max(abs(obs).max(),abs(Y).max())
 for i,m in enumerate(METHODS):
  row=summary.loc[m];idx=np.argmin(abs(depths-row.global_min_depth));a[i,0].plot(t,obs,color='.45',lw=1.2);a[i,0].plot(t,Y[idx],color=COL[m],lw=1.7);a[i,0].set(ylim=(-1.1*amp,1.1*amp),yticks=[0]);a[i,0].text(-.44 if i==0 else -.26,.5,LAB[m],transform=a[i,0].transAxes,ha='right',va='center',fontsize=9)
  g=obj[obj.method==m].sort_values('depth');v=g.objective.to_numpy();v=(v-v.min())/max(np.ptp(v),1e-30);plot_depth=np.linspace(float(g.depth.min()),float(g.depth.max()),1081);a[i,1].plot(plot_depth,PchipInterpolator(g.depth.to_numpy(),v)(plot_depth),color=COL[m],lw=1.7);a[i,1].axvline(.52,color='.4',ls='--',lw=.8);a[i,1].set(yticks=[0,1],ylim=(-.08,1.08),xlim=(.34,.70))
  h=ms[ms.method==m];a[i,2].plot(h.initial_depth,h.recovered_depth,'o-',color=COL[m],ms=2.5,lw=.8);a[i,2].axhspan(.51,.53,color='.9');a[i,2].axhline(.52,color='.4',ls='--',lw=.8);a[i,2].set(xlim=(.34,.70),ylim=(.34,.70),yticks=[.35,.52,.69]);a[i,2].text(.04,.86,f"{int(h.success.sum())}/23",transform=a[i,2].transAxes,ha='left',va='top',fontsize=9.5)
  for j in range(3):
   if i<5:a[i,j].set_xticklabels([])
 a[0,0].set_yticks([-amp,0,amp]);a[0,0].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
 for j,s in enumerate(['Scattered traces','Sampled objectives','Recovery basins']):title(a[0,j],j,s)
 a[5,0].set_xlabel('Time');a[5,1].set_xlabel('Candidate depth');a[5,2].set_xlabel('Initial depth');a[2,1].set_ylabel('Normalized objective');a[2,2].set_ylabel('Recovered depth')
 f.subplots_adjust(left=.27,right=.99,bottom=.09,top=.93,wspace=.35,hspace=.26);save(f,10)
def render_selected(numbers, data_root=None, output_root=None):
    """Render selected publication figures from a fresh-run staging directory.

    This method intentionally does not load the frozen publication reference tree.
    """
    global R, OUT
    R=Path(data_root or ROOT/'.qvvw2_generated'/'plot_inputs').resolve()
    OUT=Path(output_root or ROOT/'.qvvw2_generated'/'paper_figures').resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    fig_functions=[fig1,fig2,fig3,fig4,fig5,fig6,fig7,fig8,fig9,fig10]
    numbers=[int(n) for n in numbers]
    if not numbers or any(n<1 or n>10 for n in numbers):raise ValueError('numbers must be subset 1..10')
    global PROV
    PROV=[]
    for n in numbers:
        fig_functions[n-1]()
        print(f'Figure {n} generated from fresh-run inputs', flush=True)
    return [OUT/f'figure{n:02d}.png' for n in numbers]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'.qvvw2_generated'/'paper_figures')
    p.add_argument('--data-root',type=Path,default=ROOT/'.qvvw2_generated'/'plot_inputs')
    p.add_argument('--figures',type=int,nargs='+',default=list(range(1,11)))
    args=p.parse_args()
    render_selected(args.figures,args.data_root,args.output)

if __name__=='__main__':main()
