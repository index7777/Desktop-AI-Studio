from __future__ import annotations
import os,secrets,time
from pathlib import Path
from typing import Callable
import torch
from diffusers import QwenImage21Pipeline
from protocol import GenerateRequest
DEFAULT_MODEL="Qwen/Qwen-Image-2.1"
def model_source()->str:
 explicit=os.getenv("AI_STUDIO_MODEL")
 if explicit:return explicit
 local=Path(os.getenv("AI_STUDIO_MODELS",Path.home()/".desktop-ai-studio"/"models"))/"qwen-image-2.1"
 return str(local) if (local/"model_index.json").exists() else DEFAULT_MODEL
class QwenEngine:
 def __init__(self)->None:self.pipe:QwenImage21Pipeline|None=None
 def load(self,progress:Callable[[str],None]|None=None)->None:
  if self.pipe is not None:return
  if not torch.cuda.is_available():raise RuntimeError("Qwen Image 2.1 MVP requires a CUDA-capable NVIDIA GPU.")
  if progress:progress("loading-model")
  self.pipe=QwenImage21Pipeline.from_pretrained(model_source(),dtype=torch.bfloat16).to("cuda")
  if progress:progress("model-ready")
 def generate(self,req:GenerateRequest,progress:Callable[[str],None]|None=None)->dict:
  self.load(progress);assert self.pipe is not None
  seed=req.seed if req.seed>=0 else secrets.randbelow(2**31-1);g=torch.Generator("cuda").manual_seed(seed);Path(req.output_path).parent.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
  kwargs=dict(prompt=req.prompt,width=req.width,height=req.height,num_inference_steps=req.steps,generator=g,true_cfg_scale=req.true_cfg_scale)
  if req.negative_prompt:kwargs["negative_prompt"]=req.negative_prompt
  if progress:progress("generating")
  image=self.pipe(**kwargs).images[0];image.save(req.output_path)
  return {"path":str(Path(req.output_path).resolve()),"seed":seed,"elapsedSeconds":round(time.perf_counter()-started,3),"width":req.width,"height":req.height}
