"""
Wan2.2 Image-to-Video RunPod Serverless Handler with LoRA support.

Generates short video clips from a single input image.
Supports loading multiple LoRAs (Lightning acceleration + style/motion LoRAs).
Uploads result to Supabase Storage (bucket: generations).
Model is downloaded LAZILY on first request and cached on container local disk.
No Network Volume required — set Container Disk to 120GB+ in RunPod endpoint config.
While the worker stays warm (FlashBoot), the model remains cached.
"""

import os
import sys

# ── Redirect caches to container local disk ──────────────────────────────────
LOCAL_CACHE_PATH = "/workspace/cache"
_hf_cache_dir = os.path.join(LOCAL_CACHE_PATH, "hf_cache")
_tmp_dir = os.path.join(LOCAL_CACHE_PATH, "tmp")
os.makedirs(_hf_cache_dir, exist_ok=True)
os.makedirs(_tmp_dir, exist_ok=True)

os.environ["HF_HOME"] = _hf_cache_dir
os.environ["TRANSFORMERS_CACHE"] = _hf_cache_dir
os.environ["HF_HUB_CACHE"] = _hf_cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = _hf_cache_dir
os.environ["TMPDIR"] = _tmp_dir
os.environ["TEMP"] = _tmp_dir
os.environ["TMP"] = _tmp_dir

import tempfile
tempfile.tempdir = _tmp_dir

print(f"[init] HF cache → {_hf_cache_dir}", flush=True)
print(f"[init] TMPDIR  → {_tmp_dir}", flush=True)

import io
import gc
import time
import base64
import random
import traceback

import torch
import requests
import numpy as np
from PIL import Image

# ── Global config ────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get("MODEL_ID", "Wan-AI/Wan2.2-I2V-14B-480P-Diffusers")
DEVICE = os.environ.get("DEVICE", "cuda")
DTYPE = torch.float16

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

LORA_CACHE_DIR = os.path.join(LOCAL_CACHE_PATH, "lora_cache")
os.makedirs(LORA_CACHE_DIR, exist_ok=True)

MODEL_CACHE_DIR = os.path.join(LOCAL_CACHE_PATH, "wan22-i2v-14b")

print(f"[init] Model: {MODEL_ID}, Device: {DEVICE}", flush=True)
print(f"[init] Supabase URL configured: {bool(SUPABASE_URL)}", flush=True)
print(f"[init] Local cache dir: {MODEL_CACHE_DIR}", flush=True)

import ftfy
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video
from diffusers.pipelines.wan import pipeline_wan_i2v
pipeline_wan_i2v.ftfy = ftfy
print("[init] ftfy monkey-patched into diffusers pipeline", flush=True)

# ── Lazy pipeline holder ─────────────────────────────────────────────────────
# CRITICAL: Do NOT load model at import time. RunPod will kill the container
# if it doesn't report ready quickly. We load on first request instead.
_pipe = None


def get_pipeline():
    """Lazy-load the Wan2.2 pipeline on first call. Cached for subsequent calls."""
    global _pipe
    if _pipe is not None:
        return _pipe

    print("[model] Loading pipeline (first request)...", flush=True)
    start_time = time.time()

    cache_marker = os.path.join(MODEL_CACHE_DIR, ".download_complete")

    if os.path.exists(cache_marker):
        print(f"[model] Loading from local cache: {MODEL_CACHE_DIR}", flush=True)
        _pipe = WanImageToVideoPipeline.from_pretrained(
            MODEL_CACHE_DIR,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        )
    else:
        print(f"[model] Not in cache. Downloading {MODEL_ID}...", flush=True)
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

        from huggingface_hub import snapshot_download
        snapshot_download(
            MODEL_ID,
            local_dir=MODEL_CACHE_DIR,
            resume_download=True,
            max_workers=4,
        )

        with open(cache_marker, "w") as f:
            f.write("ok")
        print("[model] Download complete!", flush=True)

        _pipe = WanImageToVideoPipeline.from_pretrained(
            MODEL_CACHE_DIR,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        )

    # Enable memory optimizations
    try:
        _pipe.enable_model_cpu_offload()
        print("[model] CPU offload enabled", flush=True)
    except Exception as e:
        print(f"[model] CPU offload fallback to .to(DEVICE): {e}", flush=True)
        _pipe = _pipe.to(DEVICE)

    try:
        _pipe.enable_vae_slicing()
        print("[model] VAE slicing enabled", flush=True)
    except Exception:
        pass

    elapsed = time.time() - start_time
    print(f"[model] Pipeline ready! (took {elapsed:.1f}s)", flush=True)
    return _pipe


# ── Supabase helpers ─────────────────────────────────────────────────────────
_lora_cache: dict[str, str] = {}


def download_lora(source: str) -> str:
    """Download a LoRA file from URL or Supabase storage path."""
    if source in _lora_cache:
        local = _lora_cache[source]
        if os.path.exists(local):
            print(f"[lora] Cache hit: {source}", flush=True)
            return local

    if source.startswith("http://") or source.startswith("https://"):
        url = source
    else:
        url = f"{SUPABASE_URL}/storage/v1/object/public/loras/{source}"

    print(f"[lora] Downloading {url}...", flush=True)
    r = requests.get(url, timeout=300)
    r.raise_for_status()

    safe_name = source.replace("/", "_").replace(":", "_").replace("?", "_")[-120:]
    local_path = os.path.join(LORA_CACHE_DIR, safe_name)

    with open(local_path, "wb") as f:
        f.write(r.content)

    _lora_cache[source] = local_path
    size_mb = len(r.content) / 1024 / 1024
    print(f"[lora] Saved {local_path} ({size_mb:.1f} MB)", flush=True)
    return local_path


def upload_to_supabase(data: bytes, storage_path: str, bucket: str = "generations",
                       content_type: str = "video/mp4", max_retries: int = 3):
    """Upload bytes to Supabase Storage with retries."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("[upload] Supabase not configured, skipping upload", flush=True)
        return None

    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{storage_path}"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": content_type,
        "x-upsert": "true",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, data=data, timeout=120)
            if resp.status_code in (200, 201):
                print(f"[upload] Success: {storage_path} ({len(data)} bytes)", flush=True)
                return storage_path
            print(f"[upload] Attempt {attempt+1} failed: {resp.status_code} {resp.text[:200]}", flush=True)
        except Exception as e:
            print(f"[upload] Attempt {attempt+1} error: {e}", flush=True)
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    print(f"[upload] All retries failed for {storage_path}", flush=True)
    return None


def apply_loras(pipeline, lora_configs: list[dict]):
    """Load and set multiple LoRAs into the pipeline."""
    try:
        pipeline.unload_lora_weights()
    except Exception:
        pass

    if not lora_configs:
        return

    adapter_names = []
    adapter_weights = []

    for cfg in lora_configs:
        local_path = download_lora(cfg["path"])
        adapter_name = cfg["adapter_name"]
        pipeline.load_lora_weights(local_path, adapter_name=adapter_name)
        adapter_names.append(adapter_name)
        adapter_weights.append(cfg["weight"])
        print(f"[lora] Loaded '{adapter_name}' weight={cfg['weight']}", flush=True)

    pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
    print(f"[lora] Active adapters: {adapter_names} weights: {adapter_weights}", flush=True)


# ── Request handler ──────────────────────────────────────────────────────────
def handler(job):
    """Process an image-to-video generation request with optional LoRAs."""
    pipe = None
    try:
        # Lazy-load model on first request
        pipe = get_pipeline()

        inp = job["input"]
        job_id = job.get("id", "unknown")

        image_b64 = inp.get("image_base64")
        image_url = inp.get("image_url")
        prompt = inp.get("prompt", "")
        negative_prompt = inp.get("negative_prompt", "blurry, low quality, distorted")
        num_frames = min(int(inp.get("num_frames", 33)), 81)
        guidance_scale = float(inp.get("guidance_scale", 5.0))
        num_inference_steps = min(int(inp.get("num_inference_steps", 30)), 50)
        fps = int(inp.get("fps", 16))
        seed = int(inp.get("seed", -1))
        width = int(inp.get("width", 848))
        height = int(inp.get("height", 480))
        user_id = inp.get("user_id", "unknown")
        project_id = inp.get("project_id", "global")

        lightning_lora_url = inp.get("lightning_lora_url")
        lightning_lora_weight = float(inp.get("lightning_lora_weight", 1.0))
        style_lora_url = inp.get("style_lora_url")
        style_lora_weight = float(inp.get("style_lora_weight", 0.8))
        extra_lora_url = inp.get("extra_lora_url")
        extra_lora_weight = float(inp.get("extra_lora_weight", 0.7))

        # Load input image
        if image_b64:
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif image_url:
            resp = requests.get(image_url, timeout=30)
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        else:
            return {"status": "error", "error": "No input image provided"}

        image = image.resize((width, height), Image.LANCZOS)
        print(f"[handler] Image loaded: {image.size}, frames={num_frames}, steps={num_inference_steps}", flush=True)

        # Build LoRA list
        lora_configs = []
        if lightning_lora_url:
            lora_configs.append({"path": lightning_lora_url, "weight": lightning_lora_weight, "adapter_name": "lightning"})
        if style_lora_url:
            lora_configs.append({"path": style_lora_url, "weight": style_lora_weight, "adapter_name": "style"})
        if extra_lora_url:
            lora_configs.append({"path": extra_lora_url, "weight": extra_lora_weight, "adapter_name": "extra"})

        if lora_configs:
            print(f"[handler] Applying {len(lora_configs)} LoRA(s)...", flush=True)
            apply_loras(pipe, lora_configs)
        else:
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass

        # Generate
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        # Use CPU generator — compatible with model CPU offload
        generator = torch.Generator(device="cpu").manual_seed(seed)

        output = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            height=height,
            width=width,
        )

        frames = output.frames[0]
        print(f"[handler] Generated {len(frames)} frames", flush=True)

        # Export to MP4
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        export_to_video(frames, tmp_path, fps=fps)

        with open(tmp_path, "rb") as f:
            video_bytes = f.read()

        os.unlink(tmp_path)
        print(f"[handler] Video encoded: {len(video_bytes)} bytes", flush=True)

        # Upload to Supabase
        timestamp = int(time.time())
        storage_path = f"{user_id}/{project_id}/video_{job_id}_{timestamp}.mp4"
        uploaded_path = upload_to_supabase(video_bytes, storage_path)

        # Cleanup
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass

        del frames, output
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result = {
            "status": "success",
            "video": uploaded_path or storage_path,
            "storage": bool(uploaded_path),
            "seed": seed,
            "metadata": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_frames": num_frames,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "fps": fps,
                "width": width,
                "height": height,
                "model": MODEL_ID,
                "loras": [{"name": c["adapter_name"], "weight": c["weight"]} for c in lora_configs],
            },
        }

        print(f"[handler] Done: {result['video']}", flush=True)
        return result

    except Exception as e:
        traceback.print_exc()
        if pipe is not None:
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": "error", "error": str(e)}


print("[init] Worker starting (model will load on first request)...", flush=True)

import runpod
runpod.serverless.start({"handler": handler})
