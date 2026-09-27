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
                if progress:
                    progress(f"component-from-pretrained-start:{name}")
                component = cls.from_pretrained(str(source_path), subfolder=name, **kwargs)
                if progress:
                    progress(f"component-from-pretrained-ready:{name}:{system_memory()}")

                if name in {"text_encoder", "transformer", "vae"}:
                    from accelerate import disk_offload
                    offload_dir = source_path / ".runtime-offload" / name
                    offload_dir.mkdir(parents=True, exist_ok=True)
                    if progress:
                        progress(f"component-offload-start:{name}:{offload_dir}:{system_memory()}")
                    disk_offload(
                        component,
                        offload_dir=str(offload_dir),
                        execution_device=torch.device("cuda"),
                    )
                    if progress:
                        progress(f"component-offload-hooked:{name}:{system_memory()}")
                    gc.collect()
                    if progress:
                        progress(f"component-gc-ready:{name}:{system_memory()}")
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
            progress(f"model-ready:{self.profile}")
            progress(f"memory-after-offload:{system_memory()}")

    def tokenizer_self_test(self, text: str = "hello world") -> dict:
        source = Path(model_source())
        index = json.loads((source / "model_index.json").read_text(encoding="utf-8"))
        library, class_name = index["processor"]
        module = __import__(library, fromlist=[class_name])
        cls = getattr(module, class_name)
        processor = cls.from_pretrained(str(source), subfolder="processor")
        tokenizer = processor.tokenizer
        import transformers, tokenizers
        result = {
            "pythonExecutable": os.sys.executable,
            "pythonVersion": os.sys.version,
            "transformersVersion": transformers.__version__,
            "tokenizersVersion": tokenizers.__version__,
            "processorClass": type(processor).__name__,
            "tokenizerClass": type(tokenizer).__name__,
            "textType": type(text).__name__,
            "textRepr": repr(text),
        }
        encoded = tokenizer(text, add_special_tokens=False)
        result["inputIdsLength"] = len(encoded["input_ids"])
        return result

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
            "promptMaskShape": list(prompt_embeds_mask.shape) if prompt_embeds_mask is not None else None,
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

        condition_images = None
        if req.input_path:
            condition_images = [Image.open(req.input_path).convert("RGB")]

        # Keep the official Qwen Image 2.1 prompt template and hidden-state
        # extraction, but tokenize through the already-loaded processor's
        # tokenizer for text-only requests. A standalone tokenizer sanity test
        # in this exact environment passes, so this avoids ProcessorMixin's
        # text routing while preserving the checkpoint's expected template.
        if condition_images is None:
            rendered_prompt = self.pipe.prompt_template_t2i.format(req.prompt or " ")
            # tokenizer_self_test succeeds before pipeline construction. Reload
            # only the lightweight processor here so prompt tokenization cannot
            # inherit mutable fast-tokenizer state from pipeline initialization.
            source_path = Path(model_source())
            index = json.loads((source_path / "model_index.json").read_text(encoding="utf-8"))
            processor_library, processor_class = index["processor"]
            processor_module = __import__(processor_library, fromlist=[processor_class])
            clean_processor = getattr(processor_module, processor_class).from_pretrained(
                str(source_path), subfolder="processor"
            )
            tokenizer = clean_processor.tokenizer
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "left"
            try:
                if progress:
                    progress(f"tokenizer-clean:{type(tokenizer).__name__}:{type(rendered_prompt).__name__}")
                token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(rendered_prompt))
                input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
                attention_mask = torch.ones_like(input_ids)
                model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            finally:
                tokenizer.padding_side = original_padding_side

            forward_kwargs = {
                "input_ids": model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"],
                "output_hidden_states": True,
            }
            if progress:
                progress("encoding-prompt:tokenizer-direct")

            text_model = getattr(self.pipe.text_encoder.model, "language_model", self.pipe.text_encoder.model)
            handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
            try:
                outputs = self.pipe.text_encoder(**forward_kwargs)
            finally:
                handle.remove()

            hidden_states = outputs.hidden_states[-1]
            valid_hidden = list(self.pipe._extract_masked_hidden(hidden_states, model_inputs.attention_mask))
            valid_hidden = [sample[self.pipe._drop_idx :] for sample in valid_hidden]
            attn_mask_list = [torch.ones(sample.size(0), dtype=torch.long, device=sample.device) for sample in valid_hidden]
            max_seq_len = max(sample.size(0) for sample in valid_hidden)
            prompt_embeds = torch.stack([
                torch.cat([sample, sample.new_zeros(max_seq_len - sample.size(0), sample.size(1))])
                for sample in valid_hidden
            ])
            prompt_embeds_mask = torch.stack([
                torch.cat([mask, mask.new_zeros(max_seq_len - mask.size(0))]) for mask in attn_mask_list
            ])
            image_pad_mask = torch.zeros_like(prompt_embeds_mask, dtype=torch.bool)
        else:
            if progress:
                progress("encoding-prompt:multimodal-processor")
            prompt_embeds, prompt_embeds_mask, image_pad_mask = self.pipe.encode_prompt(
                prompt=req.prompt,
                image=condition_images,
                device=torch.device("cuda"),
                num_images_per_prompt=1,
            )

        kwargs = dict(
            prompt=None,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            image_pad_mask=image_pad_mask,
            num_inference_steps=req.steps,
            generator=generator,
            true_cfg_scale=req.true_cfg_scale,
        )
        if req.input_path:
            kwargs["image"] = condition_images
        else:
            kwargs.update(width=req.width, height=req.height)
        if req.negative_prompt and req.true_cfg_scale > 1:
            neg_embeds, neg_mask, neg_image_pad_mask = self.pipe.encode_prompt(
                prompt=req.negative_prompt,
                image=condition_images,
                device=torch.device("cuda"),
                num_images_per_prompt=1,
            )
            kwargs["negative_prompt_embeds"] = neg_embeds
            kwargs["negative_prompt_embeds_mask"] = neg_mask
            kwargs["negative_image_pad_mask"] = neg_image_pad_mask

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
