import os,sys,re,torch,numpy as np,argparse,glob,importlib
_WS = os.environ.get("QCC_WS", "/home/youjun/dataops_ws")  # 工作副本路径（含 run.py/logs/checkpoints）
os.chdir(_WS); sys.path.insert(0, _WS)
src=open("run.py").read(); st=src.index("parser = argparse.ArgumentParser"); en=src.index("args = parser.parse_args()")
def cfg_from_log(log):
    txt=open(log,encoding="utf-8",errors="ignore").read(); ns=re.search(r"Namespace\((.+?)\)\n",txt,re.S).group(1)
    nd={"argparse":argparse}; exec("\n".join(l[4:] if l.startswith("    ") else l for l in src[st:en].split("\n")),nd); p=nd["parser"]
    a=p.parse_args(["--is_training","1","--model_id","d","--model","Q_S_Mamba","--data","custom"])
    for m in re.finditer(r"(\w+)=('([^']*)'|True|False|None|[\d.e\-]+)",ns):
        k,v=m.group(1),m.group(2)
        if v=="True":a.__dict__[k]=True
        elif v=="False":a.__dict__[k]=False
        elif v=="None":a.__dict__[k]=None
        else:
            try:a.__dict__[k]=int(v)
            except ValueError:
                try:a.__dict__[k]=float(v)
                except ValueError:a.__dict__[k]=v.strip("'")
    return a
from data_provider.data_factory import data_provider
def load(mid,mn,dev):
    cfg=cfg_from_log(f"logs/{mid}.log"); cfg.model=mn
    M=importlib.import_module(f"model.{mn}").Model; model=M(cfg).to(dev)
    ck=glob.glob(f"checkpoints/{mid}*/checkpoint.pth")[0]
    sd=torch.load(ck,map_location=dev); sd=sd["model"] if isinstance(sd,dict) and "model" in sd else sd
    model.load_state_dict(sd); model.eval(); return cfg,model
def preds(cfg,model,dev):
    _,loader=data_provider(cfg,"test"); P=[];T=[]
    with torch.no_grad():
        for bx,by,bxm,bym in loader:
            bx=bx.float().to(dev);bm=bxm.float().to(dev);by=by.float().to(dev);bmy=bym.float().to(dev)
            dec=torch.zeros_like(by[:,-cfg.pred_len:]).float(); dec=torch.cat([by[:,:cfg.label_len],dec],1).to(dev)
            o=model(bx,bm,dec,bmy); o=o[0] if isinstance(o,(list,tuple)) else o
            P.append(o[:,-cfg.pred_len:,-1].cpu().numpy()); T.append(by[:,-cfg.pred_len:,-1].cpu().numpy())
    return np.concatenate(P),np.concatenate(T)
dev="cuda:0"
def corr(a,b):
    a=a-a.mean();b=b-b.mean();return float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
JOBS=[("ETTh1","qf2_ETTh1_96_q_base_s2024","bsl_ETTh1_96_s2024"),
      ("ETTm2","qf1_ETTm2_96_q_freq_s2023","sm_ETTm2_96_rt_s2024"),
      ("Beijing","qb_beijing_96_rt_s2023","bsl_beijing_96_s2023")]
for ds,qmid,smid in JOBS:
    cfq,mq=load(qmid,"Q_S_Mamba",dev); cfs,ms=load(smid,"S_Mamba",dev)
    Pq,Tr=preds(cfq,mq,dev); Ps,_=preds(cfs,ms,dev)
    qm=((Pq-Tr)**2).mean(1); sm=((Ps-Tr)**2).mean(1)
    qc=np.array([corr(Tr[i],Pq[i]) for i in range(len(Tr))]); sc=np.array([corr(Tr[i],Ps[i]) for i in range(len(Tr))])
    good=np.where((qm<=np.percentile(qm,35))&(qc>=0.8)&(sm>=np.percentile(sm,55)))[0]
    order=sorted(good,key=lambda i:(sm[i]-qm[i]),reverse=True)
    print(f"=== {ds}-96 : {len(good)} candidates (q_p35={np.percentile(qm,35):.4f} s_p55={np.percentile(sm,55):.4f}) ===")
    for i in order[:6]:
        print(f"   idx={i:5d} q={qm[i]:.4f} s={sm[i]:.4f} d={sm[i]-qm[i]:+.4f} corr_q={qc[i]:.3f} corr_s={sc[i]:.3f}")
