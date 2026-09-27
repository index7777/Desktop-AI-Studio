from __future__ import annotations
import gc, json, os, re, secrets, time, unicodedata
from functools import lru_cache
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


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2/Qwen byte-to-unicode table, implemented without tokenizers/Rust."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    if len(word) < 2:
        return set()
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class PurePythonQwen2Tokenizer:
    """Minimal Qwen2 byte-level BPE tokenizer implemented in pure Python.

    Transformers 5.x Qwen2Tokenizer is itself backed by the Rust `tokenizers`
    package. In the affected Windows runtime, constructing/encoding with that
    backend can terminate Python without a catchable exception. This class only
    implements the encoding behavior needed by Desktop AI Studio's text prompt
    path and never imports or calls the Rust tokenizer backend.
    """

    PRETOKENIZE_REGEX = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
        r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )

    def __init__(self, processor_dir: Path) -> None:
        self.processor_dir = processor_dir
        self.is_fast = False
        self.vocab = json.loads((processor_dir / "vocab.json").read_text(encoding="utf-8"))
        self.byte_encoder = _bytes_to_unicode()

        merges_path = processor_dir / "merges.txt"
        lines = merges_path.read_text(encoding="utf-8").splitlines()
        merges: list[tuple[str, str]] = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))
        self.bpe_ranks = {pair: rank for rank, pair in enumerate(merges)}

        self.special_token_ids: dict[str, int] = {}
        config_path = processor_dir / "tokenizer_config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            for raw_id, entry in (config.get("added_tokens_decoder") or {}).items():
                if isinstance(entry, dict) and isinstance(entry.get("content"), str):
                    try:
                        self.special_token_ids[entry["content"]] = int(raw_id)
                    except (TypeError, ValueError):
                        pass

        # Some checkpoints also place special tokens directly in vocab.json.
        for token in ("<|endoftext|>", "<|im_start|>", "<|im_end|>"):
            if token in self.vocab:
                self.special_token_ids.setdefault(token, int(self.vocab[token]))

        special_tokens = sorted(self.special_token_ids, key=len, reverse=True)
        self.special_pattern = re.compile(
            "(" + "|".join(re.escape(t) for t in special_tokens) + ")"
        ) if special_tokens else None

        try:
            import regex as regex_module
        except Exception as exc:
            raise RuntimeError(
                "Pure Python Qwen tokenizer requires the 'regex' package (normally installed with Transformers)."
            ) from exc
        self._regex = regex_module.compile(self.PRETOKENIZE_REGEX)

    @lru_cache(maxsize=65536)
    def _bpe(self, token: str) -> tuple[str, ...]:
        word = tuple(token)
        if len(word) <= 1:
            return word

        pairs = _get_pairs(word)
        while pairs:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) <= 1:
                break
            pairs = _get_pairs(word)
        return word

    def _encode_ordinary(self, text: str) -> list[int]:
        text = unicodedata.normalize("NFC", text)
        ids: list[int] = []
        for piece in self._regex.findall(text):
            encoded = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            for bpe_token in self._bpe(encoded):
                token_id = self.vocab.get(bpe_token)
                if token_id is None:
                    raise RuntimeError(f"Qwen BPE token missing from vocab: {bpe_token!r}")
                ids.append(int(token_id))
        return ids

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # Qwen's image prompt template already includes the required special
        # tokens, so add_special_tokens is intentionally ignored here.
        if not text:
            return []
        if self.special_pattern is None:
            return self._encode_ordinary(text)

        ids: list[int] = []
        for part in self.special_pattern.split(text):
            if not part:
                continue
            special_id = self.special_token_ids.get(part)
            if special_id is not None:
                ids.append(special_id)
            else:
                ids.extend(self._encode_ordinary(part))
        return ids


def load_text_tokenizer(source_path: Path) -> PurePythonQwen2Tokenizer:
    return PurePythonQwen2Tokenizer(source_path / "processor")


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
        tokenizer = load_text_tokenizer(source)
        import transformers
        template = (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
            "texture, quantity, text, spatial relationships of the objects and background:\n"
            "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        rendered = template.format(text or " ")
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        return {
            "pythonExecutable": os.sys.executable,
            "pythonVersion": os.sys.version,
            "transformersVersion": transformers.__version__,
            "tokenizerClass": type(tokenizer).__name__,
            "tokenizerIsFast": False,
            "textType": type(text).__name__,
            "renderedType": type(rendered).__name__,
            "renderedLength": len(rendered),
            "inputIdsLength": len(token_ids),
        }

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

        if condition_images is None:
            rendered_prompt = self.pipe.prompt_template_t2i.format(req.prompt or " ")
            source_path = Path(model_source())

            if progress:
                progress("tokenizer-python-load:start")
            tokenizer = load_text_tokenizer(source_path)
            if progress:
                progress(
                    f"tokenizer-python-load:done:{type(tokenizer).__name__}:"
                    f"vocab={len(tokenizer.vocab)}:merges={len(tokenizer.bpe_ranks)}"
                )
                progress("tokenizer-python-encode:start")

            token_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
            if progress:
                progress(f"tokenizer-python-encode:done:ids={len(token_ids)}")

            if not token_ids:
                raise RuntimeError("Tokenizer returned zero token IDs.")

            input_ids = torch.tensor([token_ids], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            if progress:
                progress(f"tokenizer-tensor-cpu:ready:shape={tuple(input_ids.shape)}")

            input_ids = input_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")
            model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if progress:
                progress(f"tokenizer-tensor-cuda:ready:shape={tuple(input_ids.shape)}")

            forward_kwargs = {
                "input_ids": model_inputs["input_ids"],
                "attention_mask": model_inputs["attention_mask"],
                "output_hidden_states": True,
            }
            if progress:
                progress("encoding-prompt:text-encoder:start")

            text_model = getattr(self.pipe.text_encoder.model, "language_model", self.pipe.text_encoder.model)
            handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
            try:
                outputs = self.pipe.text_encoder(**forward_kwargs)
            finally:
                handle.remove()
            if progress:
                progress("encoding-prompt:text-encoder:done")

            hidden_states = outputs.hidden_states[-1]
            if progress:
                progress(f"encoding-prompt:hidden-states:shape={tuple(hidden_states.shape)}")

            valid_hidden = list(self.pipe._extract_masked_hidden(hidden_states, model_inputs["attention_mask"]))
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
            if progress:
                progress(f"encoding-prompt:embeds-ready:shape={tuple(prompt_embeds.shape)}")
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
