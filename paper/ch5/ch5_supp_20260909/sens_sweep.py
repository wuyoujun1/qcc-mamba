import os,re,subprocess,argparse
_WS = os.environ.get("QCC_WS", "/home/youjun/dataops_ws")  # 工作副本路径（含 run.py/logs/checkpoints）
os.chdir(_WS)
src=open("run.py").read(); st=src.index("parser = argparse.ArgumentParser"); en=src.index("args = parser.parse_args()")
ns={"argparse":argparse}; exec("\n".join(l[4:] if l.startswith("    ") else l for l in src[st:en].split("\n")),ns)
parser=ns["parser"]
def readns(log):
    txt=open(log,encoding="utf-8",errors="ignore").read()
    m=re.search(r"Namespace\((.+?)\)\n",txt,re.S); d={}
    for mm in re.finditer(r"(\w+)=('([^']*)'|True|False|None|[\d.eE\-]+)",m.group(1)):
        k,v=mm.group(1),mm.group(2)
        if v=="True":d[k]=True
        elif v=="False":d[k]=False
        elif v=="None":d[k]=None
        else:
            try:d[k]=int(v)
            except:
                try:d[k]=float(v)
                except:d[k]=v.strip("'")
    return d
defaults={a.dest:a.default for a in parser._actions}
def argv(d):
    out=[]
    for k,v in d.items():
        if k not in defaults: continue
        if isinstance(v,bool):
            if v and not defaults.get(k): out += [f"--{k}"]
        elif v is None or v==defaults.get(k): continue
        else: out += [f"--{k}",str(v)]
    return out
TPL_ETTH1=readns("logs/qf2_ETTh1_96_q_base_s2024.log")
TPL_ETTM2=readns("logs/qf1_ETTm2_96_q_freq_s2023.log")
CELLS=[("ETTh1",96,TPL_ETTH1),("ETTh1",192,TPL_ETTH1),("ETTh1",336,TPL_ETTH1),
       ("ETTh1",720,TPL_ETTH1),("ETTm2",96,TPL_ETTM2),("ETTm2",192,TPL_ETTM2)]
VAR=[("nq3",dict(n_qubits=3)),("nq7",dict(n_qubits=7)),
     ("g0",dict(gate_init=0.0)),("g1",dict(gate_init=1.0)),("base",{})]
cmds=[]
for ds,H,tpl in CELLS:
    for vname,ov in VAR:
        d=dict(tpl); d.update(model_id=f"sens_{ds}_{H}_{vname}", pred_len=H, seed=2024, do_predict=False)
        d.update(ov)
        cmds.append((d["model_id"],argv(d)))
with open("logs/_sens_sweep_cmds.sh","w") as f:
    for mid,av in cmds:
        f.write(f"cd /home/youjun/dataops_ws && OMP_NUM_THREADS=8 /tmp/qcc-env/bin/python run.py {' '.join(av)} > logs/{mid}.log 2>&1\n")
print("total cmds:",len(cmds))
for mid,av in cmds[:3]: print(mid, av[:6],"...",len(av))
