from __future__ import annotations
import argparse, json, os, shutil, threading, time
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import disable_progress_bars

MODEL_ID="Qwen/Qwen-Image-2.1"
def root()->Path:return Path(os.getenv("AI_STUDIO_MODELS",Path.home()/".desktop-ai-studio"/"models"))
def model_dir()->Path:return root()/"qwen-image-2.1"
def size_bytes(p:Path)->int:return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0
def status():
 p=model_dir();installed=False
 if (p/"model_index.json").exists():
  try:
   files,total=repo_manifest()
   complete=total>0 and all((p/name).is_file() and (p/name).stat().st_size==expected for name,expected in files if expected>0)
   installed=complete
  except Exception:
   installed=False
 return {"id":"qwen-image-2.1","name":"Qwen Image 2.1","modelId":MODEL_ID,"path":str(p),"installed":installed,"sizeBytes":size_bytes(p)}
def emit_progress(downloaded:int,total:int):
 pct=round(min(downloaded,total)*100/total,1) if total else 0
 print(json.dumps({"event":"progress","downloadedBytes":min(downloaded,total),"totalBytes":total,"percent":pct}),flush=True)
def repo_manifest():
 info=HfApi().model_info(MODEL_ID,files_metadata=True)
 files=[(s.rfilename,int(s.size or 0)) for s in (info.siblings or [])]
 return files,sum(size for _,size in files)
def downloaded_bytes(files):
 base=model_dir();total=0
 for name,expected in files:
  p=base/name
  if p.is_file():
   try: total+=min(p.stat().st_size,expected)
   except OSError: pass
 return total
def install():
 root().mkdir(parents=True,exist_ok=True)
 files,total=repo_manifest()
 emit_progress(downloaded_bytes(files),total)
 disable_progress_bars()
 error=[]
 def worker():
  try:snapshot_download(repo_id=MODEL_ID,local_dir=model_dir())
  except Exception as e:error.append(e)
 t=threading.Thread(target=worker,daemon=True);t.start()
 while t.is_alive():
  emit_progress(downloaded_bytes(files),total);t.join(0.5)
 if error:raise error[0]
 emit_progress(total,total)
 return status()
def remove():shutil.rmtree(model_dir(),ignore_errors=True);return status()
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("command",choices=["status","install","remove"]);a=parser.parse_args()
 try:print(json.dumps({"ok":True,"model":globals()[a.command]()},ensure_ascii=False),flush=True)
 except Exception as e:print(json.dumps({"ok":False,"error":str(e)},ensure_ascii=False),flush=True)
