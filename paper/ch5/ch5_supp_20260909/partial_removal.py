# -*- coding: utf-8 -*-
"""按比例移除耦合：最强一半 / 最弱一半 / 随机一半，测测试 MSE 变化。"""
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
    renorm=os.environ.get("RENORM","0")=="1"
    for m in model.modules():
        if isinstance(m,qm.QuantumMixLayer):
            m.intervention_mask=None if mask is None else torch.as_tensor(mask,dtype=torch.float32,device=dev)
            m.intervention_renorm=renorm
@torch.no_grad()
def mse_of(cfg,model,loader,dev,mask):
    set_mask(model,mask,dev); tot=0.0;n=0
    for bx,by,bxm,bym in loader:
        bx=bx.float().to(dev);bm=bxm.float().to(dev);by=by.float().to(dev);bmy=bym.float().to(dev)
        dec=torch.cat([by[:,:cfg.label_len],torch.zeros_like(by[:,-cfg.pred_len:])],1).to(dev)
        o=model(bx,bm,dec,bmy); o=o[0] if isinstance(o,(list,tuple)) else o
        e=(o[:,-cfg.pred_len:,:]-by[:,-cfg.pred_len:,:])**2; tot+=float(e.sum()); n+=e.numel()
    return tot/max(n,1)
def mask_for(V,pairs):
    m=np.ones((V,V),dtype=np.float32)
    for i,j in pairs: m[i,j]=0.0; m[j,i]=0.0
    return m
if __name__=="__main__":
    mid=sys.argv[1]; frac=float(sys.argv[2]) if len(sys.argv)>2 else 0.5; gpu=sys.argv[3] if len(sys.argv)>3 else "0"
    dev=f"cuda:{gpu}"; cfg,model=load(mid,dev)
    _,l0=data_provider(cfg,"test"); loader=DataLoader(l0.dataset,batch_size=64,shuffle=False,num_workers=0)
    V=cfg.enc_in
    # 模型自身平均 K（前20批）
    Kacc=None;nb=0
    for bx,by,bxm,bym in loader:
        bx=bx.float().to(dev);bm=bxm.float().to(dev);by=by.float().to(dev);bmy=bym.float().to(dev)
        dec=torch.cat([by[:,:cfg.label_len],torch.zeros_like(by[:,-cfg.pred_len:])],1).to(dev)
        _=model(bx,bm,dec,bmy); K=model._last_K
        if K is None: continue
        K=K.detach().float().mean(0).cpu().numpy(); Kacc=K if Kacc is None else Kacc+K; nb+=1
        if nb>=20: break
    F=Kacc/nb; ii=np.triu_indices(V,1); vals=F[ii]
    k=int(round(frac*len(vals)))
    order=np.argsort(-vals)
    top=[(ii[0][o],ii[1][o]) for o in order[:k]]
    bot=[(ii[0][o],ii[1][o]) for o in order[-k:]]
    base=mse_of(cfg,model,loader,dev,None)
    mt=mse_of(cfg,model,loader,dev,mask_for(V,top))
    mb=mse_of(cfg,model,loader,dev,mask_for(V,bot))
    rng=np.random.default_rng(0); rr=[]
    for s in range(3):
        idx=rng.choice(len(vals),k,replace=False); rr.append(mse_of(cfg,model,loader,dev,mask_for(V,[(ii[0][o],ii[1][o]) for o in idx])))
    print(f"[{mid}|{int(frac*100)}%] base={base:.4f} 移除最强{int(frac*100)}%={mt:.4f}({(mt-base)/base*100:+.2f}%) "
          f"移除最弱{int(frac*100)}%={mb:.4f}({(mb-base)/base*100:+.2f}%) 随机={np.mean(rr):.4f}({(np.mean(rr)-base)/base*100:+.2f}%)",flush=True)
    json.dump({"id":mid,"frac":frac,"base":base,"top_delta":(mt-base)/base*100,"bot_delta":(mb-base)/base*100,
               "rand_delta":(np.mean(rr)-base)/base*100},open(f"/home/youjun/paper/ch5_supp_20260909/partial_{mid}_{int(frac*100)}.json","w"))
