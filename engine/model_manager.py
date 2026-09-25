from __future__ import annotations
import argparse, json, os, shutil, threading, time
from pathlib import Path
from tqdm.auto import tqdm
from huggingface_hub import HfApi, snapshot_download

MODEL_ID="Qwen/Qwen-Image-2.1"
COMPLETE_MARKER=".ai-studio-complete.json"
_progress_lock=threading.Lock()
_progress={"n":0,"total":0,"last_n":-1,"last_total":-1}

def root()->Path:return Path(os.getenv("AI_STUDIO_MODELS",Path.home()/".desktop-ai-studio"/"models"))
def model_dir()->Path:return root()/"qwen-image-2.1"
def marker()->Path:return model_dir()/COMPLETE_MARKER
def size_bytes(p:Path)->int:return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and ".cache" not in f.parts) if p.exists() else 0
def status():
 p=model_dir()
 return {"id":"qwen-image-2.1","name":"Qwen Image 2.1","modelId":MODEL_ID,"path":str(p),"installed":marker().exists() and (p/"model_index.json").exists(),"sizeBytes":size_bytes(p)}
def emit(event:str,**data):print(json.dumps({"event":event,**data},ensure_ascii=False),flush=True)
def emit_progress(n:int,total:int):
 pct=round(min(n,total)*100/total,1) if total else None
 emit("progress",downloadedBytes=n,totalBytes=total,percent=pct)
def repo_manifest(result:dict):
 try:
  info=HfApi().model_info(MODEL_ID,files_metadata=True)
  result["total"]=sum(int(s.size or 0) for s in (info.siblings or []))
  emit("manifest",totalBytes=result["total"])
 except Exception as e:
  result["error"]=str(e);emit("metadata-error",message=str(e))

class JsonProgress(tqdm):
 """Bridge Hugging Face's aggregate snapshot tqdm into JSON-lines."""
 def __init__(self,*args,**kwargs):
  super().__init__(*args,disable=True,**kwargs)
  with _progress_lock:
   # Current huggingface_hub creates aggregate transfer/reconstruction bars
   # with total=0 and grows total as files are scheduled.
   if self.total:
    _progress["total"]=max(_progress["total"],int(self.total))
 def update(self,n=1):
  if not n:return
  with _progress_lock:
   _progress["n"]+=int(n)
 def refresh(self,*args,**kwargs):return True
 def close(self):return None
 def set_description(self,*args,**kwargs):return None
 def set_description_str(self,*args,**kwargs):return None
 def set_postfix_str(self,*args,**kwargs):return None
 def set_transfer_postfix_str(self,*args,**kwargs):return None
 def update_transfer(self,n=1):
  # HF/Xet reports network transfer separately. Reconstruction progress is
  # the stable denominator-compatible signal used for the model percentage.
  return None

def install():
 root().mkdir(parents=True,exist_ok=True);marker().unlink(missing_ok=True)
 meta={};download_error=[]
 with _progress_lock:
  _progress.update(n=0,total=0,last_n=-1,last_total=-1)
 mt=threading.Thread(target=repo_manifest,args=(meta,),daemon=True);mt.start()
 def download():
  try:
   emit("phase",phase="snapshot-start",message="snapshot_download 已啟動")
   snapshot_download(repo_id=MODEL_ID,local_dir=model_dir(),max_workers=1,etag_timeout=15,tqdm_class=JsonProgress)
  except Exception as e:
   emit("download-error",message=repr(e));download_error.append(e)
 dt=threading.Thread(target=download,daemon=True);dt.start()
 last=None
 while dt.is_alive():
  manifest_total=int(meta.get("total",0))
  with _progress_lock:
   n=int(_progress["n"]);hf_total=int(_progress["total"])
  total=manifest_total or hf_total
  state=(n,total)
  if state!=last:
   emit_progress(n,total);last=state
  dt.join(0.25)
 if download_error:raise download_error[0]
 mt.join(timeout=5)
 total=int(meta.get("total",0))
 emit_progress(total,total) if total else emit_progress(size_bytes(model_dir()),0)
 marker().write_text(json.dumps({"modelId":MODEL_ID,"completedAt":time.time(),"sizeBytes":size_bytes(model_dir())}),encoding="utf-8")
 return status()
def remove():shutil.rmtree(model_dir(),ignore_errors=True);return status()
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("command",choices=["status","install","remove"]);a=parser.parse_args()
 try:print(json.dumps({"ok":True,"model":globals()[a.command]()},ensure_ascii=False),flush=True)
 except Exception as e:print(json.dumps({"ok":False,"error":str(e)},ensure_ascii=False),flush=True)
