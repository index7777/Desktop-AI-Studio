from __future__ import annotations
import argparse, json, os, shutil, threading, time
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import disable_progress_bars

MODEL_ID="Qwen/Qwen-Image-2.1"
COMPLETE_MARKER=".ai-studio-complete.json"
def root()->Path:return Path(os.getenv("AI_STUDIO_MODELS",Path.home()/".desktop-ai-studio"/"models"))
def model_dir()->Path:return root()/"qwen-image-2.1"
def marker()->Path:return model_dir()/COMPLETE_MARKER
def size_bytes(p:Path)->int:return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and ".cache" not in f.parts) if p.exists() else 0
def transfer_bytes(p:Path)->int:
 if not p.exists():return 0
 total=0
 for f in p.rglob("*"):
  if not f.is_file():continue
  try:
   # Include Hugging Face local_dir cache/incomplete payloads, but exclude tiny metadata/lock files.
   if ".cache" not in f.parts or f.name.endswith(".incomplete"):total+=f.stat().st_size
  except OSError:pass
 return total
def status():
 p=model_dir()
 return {"id":"qwen-image-2.1","name":"Qwen Image 2.1","modelId":MODEL_ID,"path":str(p),"installed":marker().exists() and (p/"model_index.json").exists(),"sizeBytes":size_bytes(p)}
def emit(event:str,**data):print(json.dumps({"event":event,**data},ensure_ascii=False),flush=True)
def repo_manifest(result:dict):
 try:
  info=HfApi().model_info(MODEL_ID,files_metadata=True)
  files=[(s.rfilename,int(s.size or 0)) for s in (info.siblings or []) if int(s.size or 0)>0]
  result["files"]=files;result["total"]=sum(x[1] for x in files)
  emit("manifest",totalBytes=result["total"])
 except Exception as e:
  result["error"]=str(e);emit("metadata-error",message=str(e))
def downloaded_bytes(files):
 base=model_dir();total=0
 for name,expected in files:
  p=base/name
  if p.is_file():
   try:total+=min(p.stat().st_size,expected)
   except OSError:pass
 return total
def install():
 root().mkdir(parents=True,exist_ok=True);marker().unlink(missing_ok=True);disable_progress_bars()
 meta={};download_error=[]
 mt=threading.Thread(target=repo_manifest,args=(meta,),daemon=True);mt.start()
 def download():
  try:
   emit("phase",phase="snapshot-start",message="snapshot_download 已啟動")
   snapshot_download(repo_id=MODEL_ID,local_dir=model_dir(),max_workers=1,etag_timeout=15)
  except Exception as e:
   emit("download-error",message=repr(e));download_error.append(e)
 dt=threading.Thread(target=download,daemon=True);dt.start()
 emit("progress",downloadedBytes=transfer_bytes(model_dir()),totalBytes=0,percent=None)
 last=(-1,-1)
 while dt.is_alive():
  files=meta.get("files");total=int(meta.get("total",0))
  downloaded=downloaded_bytes(files) if files else transfer_bytes(model_dir())
  state=(downloaded,total)
  if state!=last:
   pct=round(min(downloaded,total)*100/total,1) if total else None
   emit("progress",downloadedBytes=downloaded,totalBytes=total,percent=pct);last=state
  dt.join(0.5)
 if download_error:raise download_error[0]
 mt.join(timeout=5)
 total=int(meta.get("total",0));files=meta.get("files")
 downloaded=downloaded_bytes(files) if files else size_bytes(model_dir())
 emit("progress",downloadedBytes=downloaded,totalBytes=total,percent=100.0 if total and downloaded>=total else None)
 marker().write_text(json.dumps({"modelId":MODEL_ID,"completedAt":time.time(),"sizeBytes":size_bytes(model_dir())}),encoding="utf-8")
 return status()
def remove():shutil.rmtree(model_dir(),ignore_errors=True);return status()
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("command",choices=["status","install","remove"]);a=parser.parse_args()
 try:print(json.dumps({"ok":True,"model":globals()[a.command]()},ensure_ascii=False),flush=True)
 except Exception as e:print(json.dumps({"ok":False,"error":str(e)},ensure_ascii=False),flush=True)
