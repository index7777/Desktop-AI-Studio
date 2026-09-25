from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID="Qwen/Qwen-Image-2.1"
def root()->Path:return Path(os.getenv("AI_STUDIO_MODELS",Path.home()/".desktop-ai-studio"/"models"))
def model_dir()->Path:return root()/"qwen-image-2.1"
def size_bytes(p:Path)->int:return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0
def status():p=model_dir();return {"id":"qwen-image-2.1","name":"Qwen Image 2.1","modelId":MODEL_ID,"path":str(p),"installed":(p/"model_index.json").exists(),"sizeBytes":size_bytes(p)}
def install():root().mkdir(parents=True,exist_ok=True);snapshot_download(repo_id=MODEL_ID,local_dir=model_dir());return status()
def remove():shutil.rmtree(model_dir(),ignore_errors=True);return status()
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("command",choices=["status","install","remove"]);a=parser.parse_args()
 try:print(json.dumps({"ok":True,"model":globals()[a.command]()},ensure_ascii=False))
 except Exception as e:print(json.dumps({"ok":False,"error":str(e)},ensure_ascii=False));raise
