import os,re,subprocess,glob,time
os.chdir("/home/youjun/dataops_ws")
src=open("run.py").read(); st=src.index("parser = argparse.ArgumentParser"); en=src.index("args = parser.parse_args()")
import argparse
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
TM={
 ("ETTh1",192):"ms_ETTh196_s2024", ("ETTh1",336):"ms_ETTh196_s2024",
 ("ETTm2",192):"ms_ETTm296_s2024",
 ("ETTh2",96):"q_ETTh2_96_rt_s2023", ("ETTm1",96):"q_ETTm1_96_rt_s2023",
}
jobs=[]
for (ds,H),tmpl in TM.items():
    d=readns(f"logs/{tmpl}.log")
    d.update(model="S_Mamba", seed=2023, pred_len=H, do_predict=False)
    d["model_id"]=f"sm_{ds}_{H}_fill_s2023"
    if ds in ("ETTh2","ETTm1"):  # 保留 QCC cfg 的数据字段即可，仅模型换 S_Mamba
        pass
    jobs.append((d["model_id"], argv(d)))
with open("logs/_sm_fill_cmds.sh","w") as f:
    for mid,av in jobs: f.write(f"cd /home/youjun/dataops_ws && OMP_NUM_THREADS=8 /tmp/qcc-env/bin/python run.py {' '.join(av)} > logs/{mid}.log 2>&1\n")
print("\n".join(f"{mid}: {len(av)} flags" for mid,av in jobs))
open("/home/youjun/paper/ch5_supp_20260909/_sm_fill_cmds.sh","w").write(open("logs/_sm_fill_cmds.sh").read())
