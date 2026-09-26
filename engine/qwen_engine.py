from __future__ import annotations
import gc, json, os, secrets, time
from pathlib import Path
from typing import Callable
import torch
from PIL import Image
from diffusers import QwenImage21Pipeline

try:
    from .protocol import GenerateRequest
except ImportError:
    # Support running engine scripts directly (python engine/main.py).
    from protocol import GenerateRequest

DEFAULT_MODEL = "Qwen/Qwen-Image-2.1"

def model_source() -> str:
    explicit = os.getenv("AI_STUDIO_MODEL")
    if explicit:
        return explicit
    local = Path(os.getenv("AI_STUDIO_MODELS", Path.home()/".desktop-ai-studio"/"models"))/"qwen-image-2.1"
    return str(local) if (local/"model_index.json").exists() else DEFAULT_MODEL

def vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

def process_memory_gb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    except Exception:
        return 0.0

def system_memory() -> dict:
    try:
        import psutil
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        return {
            "processRamGB": round(process_memory_gb(), 2),
            "systemAvailableGB": round(vm.available / (1024 ** 3), 2),
            "systemUsedPercent": round(vm.percent, 1),
            "swapUsedGB": round(sm.used / (1024 ** 3), 2),
            "swapTotalGB": round(sm.total / (1024 ** 3), 2),
        }
    except Exception:
        return {"processRamGB": round(process_memory_gb(), 2)}

def hardware_profile() -> str:
    requested = os.getenv("AI_STUDIO_HARDWARE_PROFILE", "auto").strip().lower()
    if requested in {"low-vram", "normal"}:
        return requested
    return "low-vram" if vram_gb() < 12 else "normal"

class QwenEngine:
    def __init__(self) -> None:
        self.pipe: QwenImage21Pipeline | None = None
        self.profile = hardware_profile()

    def load(self, progress: Callable[[str], None] | None = None) -> None:
        if self.pipe is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen Image 2.1 requires a CUDA-capable NVIDIA GPU.")
        if progress:
            progress(f"loading-model:{self.profile}")
            progress(f"memory-before-load:{system_memory()}")

        source = model_source()
        if self.profile == "low-vram":
            if progress:
                progress("loading-strategy:staged-components")
            source_path = Path(source)
            if not source_path.exists():
                raise RuntimeError("low-vram staged loading requires the locally installed Qwen model.")

            index = json.loads((source_path / "model_index.json").read_text(encoding="utf-8"))
            components = {}
            for name in ("processor", "scheduler", "text_encoder", "transformer", "vae"):
                library, class_name = index[name]
                if progress:
                    progress(f"component-start:{name}:{system_memory()}")
                module = __import__(library, fromlist=[class_name])
                cls = getattr(module, class_name)
                kwargs = {}
                if name in {"text_encoder", "transformer", "vae"}:
                    kwargs["dtype"] = torch.bfloat16
                    kwargs["low_cpu_mem_usage"] = True
                component = cls.from_pretrained(str(source_path), subfolder=name, **kwargs)
                if progress:
                    progress(f"component-ready:{name}:{system_memory()}")

                if name in {"text_encoder", "transformer", "vae"}:
                    from accelerate import disk_offload
                    offload_dir = source_path / ".runtime-offload" / name
                    offload_dir.mkdir(parents=True, exist_ok=True)
                    if progress:
                        progress(f"component-offload-start:{name}:{system_memory()}")
                    disk_offload(
                        component,
                        offload_dir=str(offload_dir),
                        execution_device=torch.device("cuda"),
                    )
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if progress:
                        progress(f"component-offload-ready:{name}:{system_memory()}")

                components[name] = component

            if progress:
                progress(f"assembling-pipeline:{system_memory()}")
            self.pipe = QwenImage21Pipeline(
                scheduler=components["scheduler"],
                vae=components["vae"],
                text_encoder=components["text_encoder"],
                processor=components["processor"],
                transformer=components["transformer"],
            )
            if hasattr(self.pipe, "enable_vae_tiling"):
                self.pipe.enable_vae_tiling()
            if hasattr(self.pipe, "enable_vae_slicing"):
                self.pipe.enable_vae_slicing()
        else:
            self.pipe = QwenImage21Pipeline.from_pretrained(source, dtype=torch.bfloat16)
            self.pipe.enable_model_cpu_offload()

        if progress:
            progress(f"memory-after-load:{system_memory()}")

        if progress:
            progress(f"model-ready:{self.profile}")
            progress(f"memory-after-offload:{system_memory()}")

    def component_self_test(self, progress: Callable[[str], None] | None = None) -> dict:
        source = Path(model_source())
        if not source.exists():
            raise RuntimeError("component-self-test requires the local Qwen model directory.")
        index = json.loads((source / "model_index.json").read_text(encoding="utf-8"))
        tested = []
        for name, spec in index.items():
            if name.startswith("_") or not isinstance(spec, list) or len(spec) != 2:
                continue
            library, class_name = spec
            if progress:
                progress(f"component-start:{name}:{library}.{class_name}:{system_memory()}")
            module = __import__(library, fromlist=[class_name])
            cls = getattr(module, class_name)
            component = cls.from_pretrained(str(source), subfolder=name, torch_dtype=torch.bfloat16)
            tested.append(name)
            if progress:
                progress(f"component-ok:{name}:{system_memory()}")
            del component
        return {"components": tested, "memory": system_memory()}

    def self_test(self, prompt: str = "a simple red apple on a white background", progress: Callable[[str], None] | None = None) -> dict:
        self.load(progress)
        assert self.pipe is not None
        if progress:
            progress("testing-text-encoder")
            progress(f"memory-before-encode:{system_memory()}")
        started = time.perf_counter()
        prompt_embeds, prompt_embeds_mask, image_pad_mask = self.pipe.encode_prompt(
            image=None,
            prompt=prompt,
            device=torch.device("cuda"),
            num_images_per_prompt=1,
        )
        if progress:
            progress(f"memory-after-encode:{system_memory()}")
        return {
            "hardwareProfile": self.profile,
            "vramGB": round(vram_gb(), 2),
            "promptType": type(prompt).__name__,
            "promptEmbedsShape": list(prompt_embeds.shape),
            "promptMaskShape": list(prompt_embeds_mask.shape),
            "imagePadMaskShape": list(image_pad_mask.shape) if image_pad_mask is not None else None,
            "elapsedSeconds": round(time.perf_counter() - started, 3),
            "memory": system_memory(),
        }

    def generate(self, req: GenerateRequest, progress: Callable[[str], None] | None = None) -> dict:
        self.load(progress)
        assert self.pipe is not None
        seed = req.seed if req.seed >= 0 else secrets.randbelow(2**31 - 1)
        generator = torch.Generator("cuda").manual_seed(seed)
        Path(req.output_path).parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()

        kwargs = dict(
            prompt=req.prompt,
            num_inference_steps=req.steps,
            generator=generator,
            true_cfg_scale=req.true_cfg_scale,
        )
        if req.input_path:
            kwargs["image"] = Image.open(req.input_path).convert("RGB")
        else:
            kwargs.update(width=req.width, height=req.height)
        if req.negative_prompt:
            kwargs["negative_prompt"] = req.negative_prompt

        if progress:
            progress("generating")
        image = self.pipe(**kwargs).images[0]
        image.save(req.output_path)
        return {
            "path": str(Path(req.output_path).resolve()),
            "seed": seed,
            "elapsedSeconds": round(time.perf_counter() - started, 3),
            "width": image.width,
            "height": image.height,
            "hardwareProfile": self.profile,
            "vramGB": round(vram_gb(), 2),
        }
