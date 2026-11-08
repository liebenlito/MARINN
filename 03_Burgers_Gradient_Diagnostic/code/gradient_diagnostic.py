import sys,copy,math,time
from pathlib import Path
import numpy as np,pandas as pd,torch
sys.path.insert(0,str(Path(__file__).parent/'MARINN_REPRODUCIBILITY_BUNDLE'/'code'))
from marinn_core import *
torch.set_num_threads(2)
out=Path(__file__).parent
rows=[]
for seed in [2024,2025,2026,2027,2028]:
 c=Config();c.nu=.02;d=data_sets(seed,c)
 set_seed(seed); initial=MLP(c.hidden,c.depth)
 for method in ['mse','marinn']:
  model=clone_model(initial,c); opt=torch.optim.Adam(model.parameters(),lr=c.lr,betas=(.9,.999),eps=1e-8)
  best=None; bestval=float('inf'); best_epoch=0
  for ep in range(1,1201):
   opt.zero_grad(set_to_none=True)
   r=residual_sharp(model,d['train'],c);L=objective_from_residual(r,d['train'],method)
   L.backward();opt.step()
   if ep%50==0:
    rr=residual_sharp(model,d['val'],c);vv=objective_from_residual(rr,d['val'],method).item()
    if vv<bestval: bestval=vv;best=copy.deepcopy(model.state_dict());best_epoch=ep
   if ep in [1,300,600,1200]:
    r=residual_sharp(model,d['train'],c)
    # gradients of EACH loss at THE SAME weights, data
    params=list(model.parameters()); gs=[]
    for obj in ['mse','marinn']:
     loss=objective_from_residual(r,d['train'],obj)
     grad=torch.autograd.grad(loss,params,retain_graph=True)
     gs.append(torch.cat([v.detach().flatten() for v in grad]))
    a,b=gs; cos=(a@b/(a.norm()*b.norm()+1e-30)).item()
    rr=r.detach().reshape(c.nt_train,c.nx_train)
    if method=='mse': w=2*rr/(c.nx_train*c.nt_train)
    else:
     S=rr.abs().pow(4).mean(dim=1).clamp_min(1e-30)
     w=(2/(c.nt_train*c.nx_train))*rr.pow(3)/S.pow(.5).unsqueeze(1)
    ww=w.abs().flatten().numpy(); k=max(1,round(ww.size*.01))
    top=float(np.sort(ww)[-k:].sum()/ww.sum())
    x=d['train'][0].detach().flatten().numpy()
    layer=float(ww[np.abs(x-(.35+.3*d['train'][1].detach().flatten().numpy()))<.1].sum()/ww.sum())
    rows.append(dict(seed=seed,method=method,epoch=ep,training_loss=float(L.item()),grad_norm_mse=float(a.norm()),grad_norm_marinn=float(b.norm()),cosine=cos,gradient_weight_top1pct=top,gradient_weight_near_layer=layer))
   if ep%300==0: print(seed,method,ep,flush=True)
  model.load_state_dict(best)
  torch.save(model.state_dict(),out/f'sharp_{method}_seed{seed}.pt')
  print('BEST',seed,method,best_epoch,bestval,flush=True)
 pd.DataFrame(rows).to_csv(out/'gradient_diagnostic.csv',index=False)
print('FINISHED',len(rows),flush=True)
