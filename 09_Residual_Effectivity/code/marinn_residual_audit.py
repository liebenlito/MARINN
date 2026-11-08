#!/usr/bin/env python3
"""Compute residual indicators from independently evaluated 1D space-time arrays.

Input NPZ: x (nx,), t (nt,), residual (nt,nx), optional error_t and error_A
(nt,nx). Boundary/initial matching is assumed for E_MR's theoretical interpretation;
it is not checked by this utility.

Usage:
 python marinn_residual_audit.py indicators --directory evaluated_npz --output model_residuals.csv
 python marinn_residual_audit.py select --input model_residuals.csv --output selection_summary.csv

Filenames: <family>__<model_id>.npz. Required evaluation arrays are independent
of the training and checkpoint-selection points. All models in a family must
use the identical domain/grid; the script checks this.
"""
import argparse, csv, math
from pathlib import Path
import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import spearmanr

def norm(R,x,t,q,p):
    v=trapezoid(np.abs(R)**q,x,axis=1)
    return float(trapezoid(v**(p/q),t)**(1/p))

def extract(directory,output):
    paths=sorted(Path(directory).glob('*.npz'))
    if not paths: raise ValueError('No .npz files found in '+str(directory))
    rows=[]; ref={}
    for path in paths:
        stem=path.stem
        if '__' not in stem: raise ValueError(f'Filename must be family__model_id.npz: {path.name}')
        family,mid=stem.split('__',1)
        with np.load(path,allow_pickle=False) as a:
            x=np.asarray(a['x'],dtype=float); t=np.asarray(a['t'],dtype=float)
            r=np.asarray(a['residual'],dtype=float)
            if x.ndim!=1 or t.ndim!=1 or len(x)<2 or len(t)<2 or r.shape!=(len(t),len(x)):
                raise ValueError(f'Bad shapes: {path}')
            if not(np.all(np.diff(x)>0) and np.all(np.diff(t)>0) and np.all(np.isfinite(r))):
                raise ValueError(f'Non-increasing grid or nonfinite residual in {path}')
            if family in ref:
                xx,tt=ref[family]
                if not(np.array_equal(x,xx) and np.array_equal(t,tt)):
                    raise ValueError(f'Inconsistent evaluation grid within {family}')
            else: ref[family]=(x.copy(),t.copy())
            emr=float('nan')
            if 'error_t' in a and 'error_A' in a:
                et=np.asarray(a['error_t'],dtype=float); ea=np.asarray(a['error_A'],dtype=float)
                if et.shape!=r.shape or ea.shape!=r.shape or not(np.all(np.isfinite(et)) and np.all(np.isfinite(ea))):
                    raise ValueError(f'Bad error arrays in {path}')
                emr=norm(et,x,t,4,2)+norm(ea,x,t,4,2)
            rows.append(dict(family=family,model_id=mid,E_MR=emr,
                eta_MR=norm(r,x,t,4,2),eta_2=norm(r,x,t,2,2),eta_4=norm(r,x,t,4,4)))
    with open(output,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['family','model_id','E_MR','eta_MR','eta_2','eta_4']);w.writeheader();w.writerows(rows)
    print(f'Extracted {len(rows)} models in {len(ref)} PDE families to {output}.')
    print('Reminder: trapezoidal quadrature values are NOT certified continuum upper bounds.')

def select(inp,out):
    with open(inp,newline='') as f: rows=list(csv.DictReader(f))
    required={'family','model_id','E_MR','eta_MR','eta_2','eta_4'}
    if not rows or not required.issubset(rows[0]):raise ValueError('Missing columns: '+str(required))
    groups={}
    for r in rows:
        v={k:float(r[k]) for k in ('E_MR','eta_MR','eta_2','eta_4')}
        if any(not math.isfinite(z) or z<0 for z in v.values()):
            raise ValueError('E_MR and all indicator columns must be available and finite for model selection')
        groups.setdefault(r['family'],[]).append((r['model_id'],v))
    summary=[]
    for fam,models in sorted(groups.items()):
        oracle=min(models,key=lambda a:(a[1]['E_MR'],a[0]))
        for name in ('eta_MR','eta_2','eta_4'):
            selected=min(models,key=lambda a:(a[1][name],a[0]))
            regret=selected[1]['E_MR']-oracle[1]['E_MR']
            rho=float(spearmanr([m[1][name] for m in models],[m[1]['E_MR'] for m in models]).statistic) if len(models)>=3 else float('nan')
            summary.append(dict(family=fam,n_models=len(models),indicator=name,
                selected_id=selected[0],oracle_id=oracle[0],success=int(regret<=1e-12*max(1,oracle[1]['E_MR'])),
                absolute_regret=regret,relative_regret=regret/oracle[1]['E_MR'] if oracle[1]['E_MR'] else float('nan'),spearman=rho))
    with open(out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    print('Evaluated',len(groups),'families;',len(summary),'rows. Written:',out)
    for r in summary:print(r)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); commands=parser.add_subparsers(dest='cmd',required=True)
    a=commands.add_parser('indicators');a.add_argument('--directory',required=True);a.add_argument('--output',default='model_residuals.csv')
    b=commands.add_parser('select');b.add_argument('--input',required=True);b.add_argument('--output',default='selection_summary.csv')
    args=parser.parse_args()
    if args.cmd=='indicators':extract(args.directory,args.output)
    else:select(args.input,args.output)
