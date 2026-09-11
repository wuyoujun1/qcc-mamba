# -*- coding: utf-8 -*-
"""训练后评估：MSE + 拆掉全部跨变量耦合（no_offdiag）后的 ΔMSE%。"""
import os,sys,re,torch,numpy as np,argparse,glob,importlib,json
os.chdir("/home/youjun/dataops_ws"); sys.path.insert(0,"/home/youjun/dataops_ws")
import qcc.quantum_mix as qm
from torch.utils.data import DataLoader
SRC=open("run.py").read(); ST=SRC.index("parser = argparse.ArgumentParser"); EN=SRC.index("args = parser.parse_args()")
def cfg_from_log(mid):
    txt=open(f"logs/{mid}.log",encoding="utf-8",errors="ignore").read(); ns=re.search(r"Namespace\((.+?)\)\n",txt,re.S).group(1)
    nd={"argparse":argparse}; exec("\n".join(l[4:] if l.startswith("    ") else l for l in SRC[ST:EN].split("\n")),nd); p=nd["parser"]
    a=p.parse_args(["--is_training","1","--model_id","d","--model","Q_S_Mamba","--data","custom"])
    for m in re.finditer(r"(\w+)=('([^']*)'|True|False|None|[\d.e\-]+)",ns):
        k,v=m.group(1),m.group(2)
        if v=="True":a.__dict__[k]=True
        elif v=="False":a.__dict__[k]=False
        else:
            try:a.__dict__[k]=int(v)
            except:
                try:a.__dict__[k]=float(v)
                except:a.__dict__[k]=v.strip("'")
    return a
from data_provider.data_factory import data_provider
def load(mid,dev):
    cfg=cfg_from_log(mid); cfg.model="Q_S_Mamba"
    model=importlib.import_module("model.Q_S_Mamba").Model(cfg).to(dev)
    ck=glob.glob(f"checkpoints/{mid}*/checkpoint.pth")[0]
    sd=torch.load(ck,map_location=dev); sd=sd["model"] if isinstance(sd,dict) and "model" in sd else sd
    model.load_state_dict(sd); model.eval(); return cfg,model
def set_mask(model,mask,dev):
    for m in model.modules():
        if isinstance(m,qm.QuantumMixLayer):
            m.intervention_mask=None if mask is None else torch.as_tensor(mask,dtype=torch.float32,device=dev)
@torch.no_grad()
def mse_of(cfg,model,loader,dev,mask):
    set_mask(model,mask,dev); tot=0.0;n=0
    for bx,by,bxm,bym in loader:
        bx=bx.float().to(dev);bm=bxm.float().to(dev);by=by.float().to(dev);bmy=bym.float().to(dev)
        dec=torch.cat([by[:,:cfg.label_len],torch.zeros_like(by[:,-cfg.pred_len:])],1).to(dev)
        o=model(bx,bm,dec,bmy); o=o[0] if isinstance(o,(list,tuple)) else o
        e=(o[:,-cfg.pred_len:,:]-by[:,-cfg.pred_len:,:])**2; tot+=float(e.sum()); n+=e.numel()
    return tot/max(n,1)
if __name__=="__main__":
    mid=sys.argv[1]; gpu=sys.argv[2] if len(sys.argv)>2 else "0"
    dev=f"cuda:{gpu}"; cfg,model=load(mid,dev)
    _,l0=data_provider(cfg,"test"); loader=DataLoader(l0.dataset,batch_size=64,shuffle=False,num_workers=0)
    V=cfg.enc_in; off=np.ones((V,V),dtype=np.float32); np.fill_diagonal(off,1.0)
    for i in range(V):
        for j in range(V):
            if i!=j: off[i,j]=0.0
    b=mse_of(cfg,model,loader,dev,None); n=mse_of(cfg,model,loader,dev,off)
    out={"id":mid,"mse":b,"no_offdiag_mse":n,"delta_pct":(n-b)/b*100}
    json.dump(out,open(f"/home/youjun/paper/ch5_supp_20260909/grid_{mid}.json","w"))
    print(f"[{mid}] MSE={b:.4f} no_offdiag={n:.4f} Δ={(n-b)/b*100:+.2f}%",flush=True)
