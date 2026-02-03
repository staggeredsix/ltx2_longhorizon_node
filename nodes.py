import json
import math
import os
import time
from typing import Any, Callable

try:
    import requests  # type: ignore
except Exception:
    requests = None

from .ffmpeg_utils import concat_and_mux, concat_mp4


def _adjust_num_frames(num_frames: int) -> int:
    if num_frames < 1:
        return 1
    if (num_frames - 1) % 8 == 0:
        return num_frames
    return ((num_frames - 1) // 8 + 1) * 8 + 1


def _round_down_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return max(multiple, (value // multiple) * multiple)


def _resolve_callable() -> Callable[..., Any]:
    direct = os.getenv("LTX2_LONG_HORIZON_CALLABLE", "").strip()
    if direct:
        module_name, _, attr = direct.partition(":")
        if not module_name or not attr:
            raise RuntimeError("LTX2_LONG_HORIZON_CALLABLE must be module:callable")
        module = __import__(module_name, fromlist=[attr])
        callable_obj = getattr(module, attr)
        if not callable(callable_obj):
            raise RuntimeError(f"{direct} is not callable")
        return callable_obj

    candidates = [
        ("ltx_nodes", "render_ltx2_chunk"),
        ("ltx_nodes", "LTX2_OneClip"),
        ("ltx2_nodes", "render_ltx2_chunk"),
        ("ltx2_nodes", "LTX2_OneClip"),
    ]
    for module_name, attr in candidates:
        try:
            module = __import__(module_name, fromlist=[attr])
            obj = getattr(module, attr, None)
        except Exception:
            continue
        if obj is None:
            continue
        if callable(obj):
            return obj
        if hasattr(obj, "generate"):
            return getattr(obj, "generate")
    raise RuntimeError(
        "Unable to locate LTX-2 template callable. "
        "Set LTX2_LONG_HORIZON_CALLABLE=module:callable or use API fallback."
    )


def _api_fallback(
    workflow_path: str,
    *,
    seed: int,
    fps: int,
    frames: int,
) -> dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is not available for API fallback.")
    api_url = os.getenv("COMFY_API_URL", "http://127.0.0.1:8188")
    if not workflow_path or not os.path.exists(workflow_path):
        raise RuntimeError("LTX2_WORKFLOW_JSON is required for API fallback.")
    with open(workflow_path, "r", encoding="utf-8") as handle:
        workflow = json.load(handle)
    prompt = workflow.get("prompt") or workflow
    for node in prompt.values():
        inputs = node.get("inputs", {})
        if "seed" in inputs:
            inputs["seed"] = seed
        if "fps" in inputs:
            inputs["fps"] = fps
        if "num_frames" in inputs:
            inputs["num_frames"] = frames
    response = requests.post(f"{api_url}/prompt", json={"prompt": prompt}, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Comfy API error: {response.status_code} {response.text}")
    return response.json()


class LTX2_LongHorizon_Controller:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "frames_per_chunk": ("INT", {"default": 73, "min": 1, "max": 4096}),
                "mode": (["commercial", "continuous"],),
                "target_seconds": ("FLOAT", {"default": 35.0, "min": 0.0, "max": 600.0}),
                "seed_base": ("INT", {"default": 10, "min": 0, "max": 2**31 - 1}),
                "seed_stride": ("INT", {"default": 1, "min": 1, "max": 2**31 - 1}),
                "chain_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0}),
                "chain_frames": ("INT", {"default": 3, "min": 1, "max": 8}),
                "drop_prefix": ("INT", {"default": 0, "min": 0, "max": 8}),
                "blend_frames": ("INT", {"default": 3, "min": 0, "max": 16}),
                "reset_interval": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "output_dir": ("STRING", {"default": "outputs"}),
                "basename": ("STRING", {"default": "ltx2_long_horizon"}),
                "keep_chunks": ("BOOLEAN", {"default": True}),
                "rolling_chunks": ("INT", {"default": 8, "min": 1, "max": 1000}),
                "stop_file": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("final_or_latest_mp4_path", "chunk_paths_json", "status")
    FUNCTION = "run"
    CATEGORY = "LTX2"

    def run(
        self,
        model,
        vae,
        audio_vae,
        positive,
        negative,
        fps: int,
        frames_per_chunk: int,
        mode: str,
        target_seconds: float,
        seed_base: int,
        seed_stride: int,
        chain_strength: float,
        chain_frames: int,
        drop_prefix: int,
        blend_frames: int,
        reset_interval: int,
        output_dir: str,
        basename: str,
        keep_chunks: bool,
        rolling_chunks: int,
        stop_file: str,
    ):
        frames_per_chunk = _adjust_num_frames(frames_per_chunk)
        os.makedirs(output_dir, exist_ok=True)
        chunk_paths: list[str] = []
        wav_paths: list[str] = []
        latest_path = os.path.join(output_dir, f"{basename}_latest.mp4")
        status = "ok"

        def _should_stop() -> bool:
            return bool(stop_file and os.path.exists(stop_file))

        if mode == "commercial":
            total_chunks = max(1, int(math.ceil((target_seconds * fps) / float(frames_per_chunk))))
            total_chunks = max(1, total_chunks)
        else:
            total_chunks = 1

        try:
            chunk_fn = _resolve_callable()
            use_api = False
        except Exception as exc:
            workflow_path = os.getenv("LTX2_WORKFLOW_JSON", "")
            if workflow_path:
                chunk_fn = None
                use_api = True
            else:
                raise exc

        chunk_index = 0
        while True:
            chunk_index += 1
            if mode == "commercial" and chunk_index > total_chunks:
                break
            if mode == "continuous" and _should_stop():
                status = "stopped"
                break

            seed = seed_base + (chunk_index - 1) * seed_stride
            out_path = os.path.join(output_dir, f"{basename}_chunk{chunk_index}.mp4")

            if use_api:
                _api_fallback(os.getenv("LTX2_WORKFLOW_JSON", ""), seed=seed, fps=fps, frames=frames_per_chunk)
                if not os.path.exists(out_path):
                    raise RuntimeError("API fallback did not create expected chunk mp4.")
                audio_path = None
            else:
                result = chunk_fn(
                    model=model,
                    vae=vae,
                    audio_vae=audio_vae,
                    positive=positive,
                    negative=negative,
                    fps=fps,
                    num_frames=frames_per_chunk,
                    seed=seed,
                    chain_strength=chain_strength,
                    chain_frames=chain_frames,
                    drop_prefix=drop_prefix,
                    blend_frames=blend_frames,
                    reset_interval=reset_interval,
                    output_path=out_path,
                )
                if isinstance(result, (list, tuple)) and len(result) >= 1:
                    out_path = result[0]
                    audio_path = result[1] if len(result) > 1 else None
                elif isinstance(result, dict):
                    out_path = result.get("video_path") or out_path
                    audio_path = result.get("audio_path")
                else:
                    audio_path = None

            if not os.path.exists(out_path):
                raise RuntimeError(f"Chunk mp4 missing: {out_path}")
            chunk_paths.append(out_path)
            if audio_path and os.path.exists(audio_path):
                wav_paths.append(audio_path)

            if mode == "continuous":
                ring = chunk_paths[-max(1, rolling_chunks):]
                concat_mp4(ring, latest_path)
            if mode == "continuous" and _should_stop():
                status = "stopped"
                break

        final_path = latest_path
        if mode == "commercial":
            final_path = os.path.join(output_dir, f"{basename}.mp4")
            concat_and_mux(chunk_paths, wav_paths, fps, final_path)

        if not keep_chunks:
            for path in chunk_paths:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            for path in wav_paths:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

        return (final_path, json.dumps(chunk_paths), status)


NODE_CLASS_MAPPINGS = {
    "LTX2_LongHorizon_Controller": LTX2_LongHorizon_Controller,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX2_LongHorizon_Controller": "LTX2: Long Horizon Controller",
}
