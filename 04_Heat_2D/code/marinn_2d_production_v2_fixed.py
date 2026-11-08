
import math, time, copy, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.fft import dstn, idstn

torch.set_num_threads(int(os.environ.get("MARINN_THREADS", "1")))
DEVICE = "cpu"
DTYPE = torch.float32
SEED = int(os.environ.get("MARINN_SEED", "2026"))
WIDTH = 48
EPOCHS_HEAT = 500
EPOCHS_AC_DIRECT = 500
EPOCHS_AC_INNER = 250
PICARD_MAX = 10
LR = 2e-3

OUT = Path(os.environ.get("MARINN_OUT", f"/mnt/data/MARINN_2D_PRODUCTION_FIXEDK10/seed{SEED}"))
OUT.mkdir(parents=True, exist_ok=True)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

class MLP(torch.nn.Module):
    def __init__(self, width=48):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3,width), torch.nn.Tanh(),
            torch.nn.Linear(width,width), torch.nn.Tanh(),
            torch.nn.Linear(width,width), torch.nn.Tanh(),
            torch.nn.Linear(width,1)
        )
        # zero output at initialization while retaining random hidden features
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self,z):
        return self.net(z)

def clone(m):
    n=MLP(WIDTH)
    n.load_state_dict(copy.deepcopy(m.state_dict()))
    return n

def tensor_grid(nx,ny,nt,T=1.0,dtype=DTYPE,seed=None):
    if seed is None:
        x=(torch.arange(nx,dtype=dtype)+0.5)/nx
        y=(torch.arange(ny,dtype=dtype)+0.5)/ny
        t=T*(torch.arange(nt,dtype=dtype)+0.5)/nt
    else:
        g=torch.Generator(device="cpu"); g.manual_seed(int(seed))
        x=(torch.arange(nx,dtype=dtype)+torch.rand(nx,generator=g,dtype=dtype))/nx
        y=(torch.arange(ny,dtype=dtype)+torch.rand(ny,generator=g,dtype=dtype))/ny
        t=T*(torch.arange(nt,dtype=dtype)+torch.rand(nt,generator=g,dtype=dtype))/nt
    TT,YY,XX=torch.meshgrid(t,y,x,indexing="ij")
    return XX.reshape(-1,1),YY.reshape(-1,1),TT.reshape(-1,1),(nx,ny,nt,T)

def u0_heat(x,y):
    return torch.sin(math.pi*x)*torch.sin(math.pi*y)

def u0_ac(x,y,A=0.9):
    return A*torch.sin(math.pi*x)*torch.sin(math.pi*y)

def hard_field(m,x,y,t,u0):
    return u0(x,y)+t*x*(1-x)*y*(1-y)*m(torch.cat([x,y,t],1))

def derivs(m,pack,u0):
    x,y,t,shape=pack
    x=x.detach().clone().requires_grad_(True)
    y=y.detach().clone().requires_grad_(True)
    t=t.detach().clone().requires_grad_(True)
    u=hard_field(m,x,y,t,u0)
    one=torch.ones_like(u)
    ut=torch.autograd.grad(u,t,one,create_graph=True,retain_graph=True)[0]
    ux=torch.autograd.grad(u,x,one,create_graph=True,retain_graph=True)[0]
    uy=torch.autograd.grad(u,y,one,create_graph=True,retain_graph=True)[0]
    uxx=torch.autograd.grad(ux,x,one,create_graph=True,retain_graph=True)[0]
    uyy=torch.autograd.grad(uy,y,one,create_graph=True,retain_graph=True)[0]
    return u,ut,ux,uy,uxx,uyy,x,y,t

def mixed_loss_normalized(r,pack):
    """Time-normalized squared L2_t(L4_xy) loss on the unit square."""
    nx,ny,nt,T=pack[3]
    R=r.reshape(nt,ny,nx)
    return torch.mean(torch.mean(torch.abs(R)**4,dim=(1,2))**0.5)

def mixed_loss_continuum_squared(r,pack):
    """Quadrature-consistent squared L2(0,T;L4(Omega)) norm."""
    T=pack[3][3]
    return T*mixed_loss_normalized(r,pack)

def mixed_loss(r,pack):
    # Backward-compatible alias used by the optimizer: normalized form.
    return mixed_loss_normalized(r,pack)

def flat_l4_sqnorm(r):
    return torch.mean(torch.abs(r)**4)**0.5

def objective(r,pack,kind):
    if kind=="mse": return torch.mean(r*r)
    if kind=="flat_l4": return flat_l4_sqnorm(r)
    if kind=="marinn": return mixed_loss_normalized(r,pack)
    raise ValueError(kind)

def residual_heat(m,pack,nu=0.1):
    u,ut,ux,uy,uxx,uyy,x,y,t=derivs(m,pack,u0_heat)
    f=(-1+2*nu*math.pi**2)*torch.exp(-t)*torch.sin(math.pi*x)*torch.sin(math.pi*y)
    return ut-nu*(uxx+uyy)-f

def residual_ac(m,pack,nu=0.01):
    u,ut,ux,uy,uxx,uyy,x,y,t=derivs(m,pack,lambda a,b:u0_ac(a,b,0.9))
    return ut-nu*(uxx+uyy)+u**3-u

def frozen_ac_residual(m,prev,pack,nu=0.01):
    u,ut,ux,uy,uxx,uyy,x,y,t=derivs(m,pack,lambda a,b:u0_ac(a,b,0.9))
    with torch.no_grad():
        v=hard_field(prev,x.detach(),y.detach(),t.detach(),lambda a,b:u0_ac(a,b,0.9))
    return ut-nu*(uxx+uyy)+(v*v-1.0)*u

def train(m,train_pack,val_pack,res_fn,kind,epochs,tag,prev=None):
    opt=torch.optim.Adam(m.parameters(),lr=LR)
    best=copy.deepcopy(m.state_dict())
    bestv=float("inf")
    hist=[]
    t0=time.perf_counter()
    for ep in range(1,epochs+1):
        opt.zero_grad(set_to_none=True)
        r=res_fn(m,train_pack) if prev is None else res_fn(m,prev,train_pack)
        loss=objective(r,train_pack,kind)
        loss.backward()
        opt.step()
        if ep==1 or ep%50==0:
            rv=res_fn(m,val_pack) if prev is None else res_fn(m,prev,val_pack)
            vl=objective(rv,val_pack,kind).detach().item()
            hist.append([ep,float(loss.detach()),vl])
            if vl<bestv:
                bestv=vl
                best=copy.deepcopy(m.state_dict())
    m.load_state_dict(best)
    return m,pd.DataFrame(hist,columns=["epoch","train_loss","val_loss"]),time.perf_counter()-t0,bestv

def heat_eval(m,n=36,nt=21,nu=0.1):
    xv=torch.linspace(1/(n+1),n/(n+1),n,dtype=torch.float64)
    yv=xv.clone()
    tv=torch.linspace(0,1,nt,dtype=torch.float64)
    TT,YY,XX=torch.meshgrid(tv,yv,xv,indexing="ij")
    x=XX.reshape(-1,1); y=YY.reshape(-1,1); t=TT.reshape(-1,1)
    md=copy.deepcopy(m).double()
    u,ut,ux,uy,uxx,uyy,xg,yg,tg=derivs(md,(x,y,t,(n,n,nt,1.0)),lambda a,b:torch.sin(math.pi*a)*torch.sin(math.pi*b))
    ue=torch.exp(-tg)*torch.sin(math.pi*xg)*torch.sin(math.pi*yg)
    uet=-ue
    lap_e=-2*math.pi**2*ue
    e=u-ue
    et=ut-uet
    laperr=(uxx+uyy)-lap_e
    r=ut-nu*(uxx+uyy)-(-1+2*nu*math.pi**2)*ue
    rel=(torch.linalg.vector_norm(e.detach())/torch.linalg.vector_norm(ue.detach())).item()
    eta=torch.sqrt(mixed_loss(r,(x,y,t,(n,n,nt,1.0)))).detach().item()
    etn=torch.sqrt(mixed_loss(et,(x,y,t,(n,n,nt,1.0)))).detach().item()
    lapn=torch.sqrt(mixed_loss(laperr,(x,y,t,(n,n,nt,1.0)))).detach().item()
    return dict(rel_L2=rel,eta_MR=eta,E_MR=etn+nu*lapn,max_residual=r.detach().abs().max().item()), \
           ue.detach().reshape(nt,n,n).numpy(),u.detach().reshape(nt,n,n).numpy()

def reaction_step(u,h):
    return u/np.sqrt(u*u+(1-u*u)*np.exp(-2*h))

def allen_cahn_reference(N=40,nu=0.01,T=0.4,dt=0.001,A=0.9,nt_save=21):
    x=np.arange(1,N+1)/(N+1)
    X,Y=np.meshgrid(x,x,indexing="xy")
    u=A*np.sin(np.pi*X)*np.sin(np.pi*Y)
    k=np.arange(1,N+1)
    lam=np.pi**2*(k[:,None]**2+k[None,:]**2)
    diff=np.exp(-nu*lam*dt)
    steps=int(round(T/dt))
    save_steps=np.linspace(0,steps,nt_save,dtype=int)
    states=[]; laps=[]; uts=[]; times=[]
    def diagnostics(v,tn):
        vh=dstn(v,type=1,norm="ortho")
        lap=idstn(-lam*vh,type=1,norm="ortho")
        ut=nu*lap+v-v**3
        states.append(v.copy()); laps.append(lap.copy()); uts.append(ut.copy()); times.append(tn)
    si=0
    diagnostics(u,0.0); si=1
    for n in range(1,steps+1):
        u=reaction_step(u,dt/2)
        uh=dstn(u,type=1,norm="ortho")
        u=idstn(uh*diff,type=1,norm="ortho")
        u=reaction_step(u,dt/2)
        if si<len(save_steps) and n==save_steps[si]:
            diagnostics(u,n*dt); si+=1
    return x,np.array(times),np.array(states),np.array(uts),np.array(laps)

def ac_eval(m,ref,nu=0.01):
    x1,times,Uref,Utref,Lapref=ref
    n=len(x1); nt=len(times)
    xv=torch.tensor(x1,dtype=torch.float64)
    yv=xv.clone(); tv=torch.tensor(times,dtype=torch.float64)
    TT,YY,XX=torch.meshgrid(tv,yv,xv,indexing="ij")
    x=XX.reshape(-1,1); y=YY.reshape(-1,1); t=TT.reshape(-1,1)
    md=copy.deepcopy(m).double()
    u,ut,ux,uy,uxx,uyy,*_=derivs(md,(x,y,t,(n,n,nt,float(times[-1]))),lambda a,b:0.9*torch.sin(math.pi*a)*torch.sin(math.pi*b))
    Up=u.detach().reshape(nt,n,n).numpy()
    Utp=ut.detach().reshape(nt,n,n).numpy()
    Lapp=(uxx+uyy).detach().reshape(nt,n,n).numpy()
    E=Up-Uref
    rel=np.linalg.norm(E)/np.linalg.norm(Uref)
    def mixed_np(A):
        # Continuum-scale L2(0,T;L4) norm on the unit square.
        T=float(times[-1])
        return np.sqrt(T*np.mean(np.sqrt(np.mean(np.abs(A)**4,axis=(1,2)))))
    eta_res=Utp-nu*Lapp+Up**3-Up
    eta=mixed_np(eta_res)
    et=mixed_np(Utp-Utref)
    lap=mixed_np(Lapp-Lapref)
    return dict(rel_L2=rel,eta_MR=eta,E_MR=et+nu*lap,max_residual=np.max(np.abs(eta_res))),Uref,Up

def picard_delta(a,b,pack,T=0.4):
    x,y,t,_=pack
    with torch.no_grad():
        ua=hard_field(a,x,y,t,lambda aa,bb:u0_ac(aa,bb,0.9))
        ub=hard_field(b,x,y,t,lambda aa,bb:u0_ac(aa,bb,0.9))
    return (torch.linalg.vector_norm(ua-ub)/torch.linalg.vector_norm(ua)).item()

def plot_composite(ref,preds,names,times,path,title):
    # selected middle time
    k=len(times)//2
    cols=1+len(preds)
    fig,axes=plt.subplots(2,cols,figsize=(4*cols,7),constrained_layout=True)
    vmin,vmax=ref[k].min(),ref[k].max()
    im=axes[0,0].imshow(ref[k],origin="lower",extent=[0,1,0,1],vmin=vmin,vmax=vmax)
    axes[0,0].set_title(f"Reference t={times[k]:.2f}"); fig.colorbar(im,ax=axes[0,0],fraction=.046)
    axes[1,0].axis("off")
    emax=0
    for P in preds: emax=max(emax,np.max(np.abs(P[k]-ref[k])))
    for j,(P,name) in enumerate(zip(preds,names),start=1):
        im=axes[0,j].imshow(P[k],origin="lower",extent=[0,1,0,1],vmin=vmin,vmax=vmax)
        axes[0,j].set_title(name); fig.colorbar(im,ax=axes[0,j],fraction=.046)
        im=axes[1,j].imshow(np.abs(P[k]-ref[k]),origin="lower",extent=[0,1,0,1],vmin=0,vmax=emax)
        axes[1,j].set_title(f"|error| {name}"); fig.colorbar(im,ax=axes[1,j],fraction=.046)
    fig.suptitle(title)
    plt.savefig(path,dpi=200,bbox_inches="tight"); plt.close(fig)

def main():
    set_seed(SEED)
    summary=[]

    # ---------- 2D HEAT ----------
    tr=tensor_grid(10,10,12,T=1.0,seed=100*SEED+1)
    va=tensor_grid(8,8,10,T=1.0,seed=100*SEED+2)
    base=MLP(WIDTH)
    heat_fields={}
    heat_ref=None
    for name,kind in [("PINN-MSE","mse"),("Flat-L4-squared","flat_l4"),("MARINN","marinn")]:
        m=clone(base)
        m,h,wall,bv=train(m,tr,va,lambda mm,pp:residual_heat(mm,pp,0.1),kind,EPOCHS_HEAT,name)
        met,ref,pred=heat_eval(m)
        summary.append(dict(problem="2D heat",method=name,wall_seconds=wall,**met))
        h.to_csv(OUT/f"heat_{name}_history.csv",index=False)
        heat_fields[name]=pred; heat_ref=ref
        torch.save(m.state_dict(),OUT/f"heat_{name}.pt")
    times=np.linspace(0,1,21)
    plot_composite(heat_ref,[heat_fields["PINN-MSE"],heat_fields["Flat-L4-squared"],heat_fields["MARINN"]],
                   ["PINN-MSE","Flat-$L_4^2$","MARINN"],times,OUT/"heat_2d_composite.png","2D heat production-v1")

    # ---------- 2D ALLEN-CAHN ----------
    T=0.4
    tr=tensor_grid(10,10,12,T=T,seed=100*SEED+11)
    va=tensor_grid(8,8,10,T=T,seed=100*SEED+12)
    mon=tensor_grid(8,8,10,T=T,seed=100*SEED+13)
    ref=allen_cahn_reference()
    base=MLP(WIDTH)
    ac_models={}
    ac_preds={}
    # direct
    for name,kind in [("Direct-MSE","mse"),("Direct-MARINN","marinn")]:
        m=clone(base)
        m,h,wall,bv=train(m,tr,va,lambda mm,pp:residual_ac(mm,pp,0.01),kind,EPOCHS_AC_DIRECT,name)
        met,Uref,Up=ac_eval(m,ref)
        summary.append(dict(problem="2D Allen-Cahn",method=name,wall_seconds=wall,picard_steps=0,**met))
        h.to_csv(OUT/f"ac_{name}_history.csv",index=False)
        ac_models[name]=m; ac_preds[name]=Up
        torch.save(m.state_dict(),OUT/f"ac_{name}.pt")
    # Picard
    for name,kind in [("PINN-Picard","mse"),("MARINN-Picard","marinn")]:
        current=clone(base)
        trace=[]; walltot=0
        for k in range(1,PICARD_MAX+1):
            prev=clone(current)
            cand=clone(prev)
            cand,h,wall,bv=train(cand,tr,va,lambda mm,pv,pp:frozen_ac_residual(mm,pv,pp,0.01),kind,EPOCHS_AC_INNER,name,prev=prev)
            current=cand; walltot+=wall
            d=picard_delta(current,prev,mon,T)
            rr=residual_ac(current,mon,0.01)
            rho=float(torch.sqrt(mixed_loss_continuum_squared(rr,mon)).detach())
            trace.append([k,d,rho,bv,wall])
        pd.DataFrame(trace,columns=["picard_step","delta","rho_full","best_val","stage_wall"]).to_csv(OUT/f"ac_{name}_picard_trace.csv",index=False)
        met,Uref,Up=ac_eval(current,ref)
        summary.append(dict(problem="2D Allen-Cahn",method=name,wall_seconds=walltot,picard_steps=len(trace),**met))
        ac_models[name]=current; ac_preds[name]=Up
        torch.save(current.state_dict(),OUT/f"ac_{name}.pt")
    plot_composite(ref[2],[ac_preds["Direct-MSE"],ac_preds["Direct-MARINN"],ac_preds["PINN-Picard"],ac_preds["MARINN-Picard"]],
                   ["Direct MSE","Direct MARINN","PINN-Picard","MARINN-Picard"],ref[1],OUT/"allen_cahn_2d_composite.png","2D Allen-Cahn non-manufactured production-v1")

    df=pd.DataFrame(summary)
    df.to_csv(OUT/"production_summary.csv",index=False)
    print(df.to_string(index=False))
    (OUT/"config.json").write_text(json.dumps({
        "seed":SEED,"width":WIDTH,"architecture":"3-48-48-48-1","lr":LR,
        "heat_epochs":EPOCHS_HEAT,"ac_direct_epochs":EPOCHS_AC_DIRECT,"ac_inner_epochs":EPOCHS_AC_INNER,
        "picard_max":PICARD_MAX,"picard_stopping":"fixed K=10; delta and rho diagnostic only","mixed_loss_training":"time-normalized L2(L4)^2; reported diagnostics continuum-scaled","train_grid":"10x10x12","val_grid":"8x8x10",
        "heat_nu":0.1,"allen_cahn_nu":0.01,"allen_cahn_T":0.4,"allen_cahn_u0_amplitude":0.9,
        "allen_cahn_reference":"40x40 sine pseudospectral Strang splitting, dt=0.001"
    },indent=2))

if __name__=="__main__":
    main()
