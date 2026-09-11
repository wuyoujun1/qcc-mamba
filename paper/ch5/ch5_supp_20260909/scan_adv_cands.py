import os,sys,re,torch,numpy as np,argparse,glob,importlib
_WS = os.environ.get("QCC_WS", "/home/youjun/dataops_ws")  # 工作副本路径（含 run.py/logs/checkpoints）
os.chdir(_WS); sys.path.insert(0, _WS)
def cfg_from_log(log):
    txt=open(log,encoding="utf-8",errors="ignore").read()
    ns=re.search(r"Namespace\((.+?)\)\n",txt,re.S).group(1)
    src=open("/home/youjun/dataops_ws/run.py").read(); st=src.index("parser = argparse.ArgumentParser"); en=src.index("args = parser.parse_args()")
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
jobs=[("qf2_ETTh1_96_q_base_s2024","bsl_ETTh1_96_s2024",96,60),
      ("k720_T02","sm_ETTh1_720_rt_s2024",720,360)]
for qmid,smid,H,seg in jobs:
    cfq,mq=load(qmid,"Q_S_Mamba",dev); cfs,ms=load(smid,"S_Mamba",dev)
    Pq,Tr=preds(cfq,mq,dev); Ps,_=preds(cfs,ms,dev)
    qm=((Pq-Tr)**2).mean(1); sm=((Ps-Tr)**2).mean(1)
    q_seg=((Pq[:,:seg]-Tr[:,:seg])**2).mean(1); s_seg=((Ps[:,:seg]-Tr[:,:seg])**2).mean(1)
    def corr(a,b): 
        a=a-a.mean();b=b-b.mean();return (a*b).sum()/ (np.linalg.norm(a)*np.linalg.norm(b)+1e-9)
    qc=np.array([corr(Tr[i],Pq[i]) for i in range(len(Tr))])
    medq=np.median(qm)
    # 96: QCC 质量好(q<=p40) & 前seg段 SM明显差；720: qcc 低误差&高corr & sm高
    if H==96:
        good=np.where((qm<=np.percentile(qm,40))&(s_seg-q_seg> (np.std(s_seg-q_seg))*0.5) & (sm>medq))[0]
    else:
        good=np.where((qm<=np.percentile(qm,30))&(qc>=0.9)&(sm>=np.percentile(sm,55)))[0]
    order=sorted(good,key=lambda i:(sm[i]-qm[i]),reverse=True)
    print(f"=== H={H}: 候选窗口 top {min(8,len(order))} (med_q={medq:.4f}) ===")
    for i in order[:8]:
        print(f"  idx={i:5d} q={qm[i]:.4f} s={sm[i]:.4f} d={sm[i]-qm[i]:+.4f} q_seg={q_seg[i]:.4f} s_seg={s_seg[i]:.4f} corr_q={qc[i]:.3f}")
