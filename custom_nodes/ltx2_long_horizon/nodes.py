import importlib
import json
import math
import os
import time
from typing import Any, Callable

from .ffmpeg_utils import concat_and_mux, concat_mp4


API_PORT_START = 8188
API_PORT_END = 8192
DEFAULT_API_TIMEOUT_SEC = 600


def _get_requests():
    try:
        return importlib.import_module("requests")
    except Exception as exc:
        raise RuntimeError("requests is required for ComfyUI API mode.") from exc


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


def _resolve_direct_callable() -> Callable[..., Any] | None:
    direct = os.getenv("LTX2_LONG_HORIZON_CALLABLE", "").strip()
    if not direct:
        return None
    module_name, _, attr = direct.partition(":")
    if not module_name or not attr:
        raise RuntimeError("LTX2_LONG_HORIZON_CALLABLE must be module:callable")
    module = importlib.import_module(module_name)
    callable_obj = getattr(module, attr)
    if not callable(callable_obj):
        raise RuntimeError(f"{direct} is not callable")
    return callable_obj


def _probe_api(api_url: str) -> bool:
    requests = _get_requests()
    for endpoint in ("system_stats", "queue"):
        try:
            response = requests.get(f"{api_url}/{endpoint}", timeout=2)
        except Exception:
            continue
        if response.status_code == 200:
            return True
    return False


def _discover_api_url(comfy_api_url: str) -> str:
    comfy_api_url = (comfy_api_url or "").strip()
    if comfy_api_url:
        if not _probe_api(comfy_api_url):
            raise RuntimeError(f"ComfyUI API not reachable at {comfy_api_url}.")
        return comfy_api_url
    for port in range(API_PORT_START, API_PORT_END + 1):
        candidate = f"http://127.0.0.1:{port}"
        if _probe_api(candidate):
            return candidate
    raise RuntimeError(
        "ComfyUI API not reachable; start ComfyUI with --listen/--port."
    )


def _looks_like_prompt(data: dict[str, Any]) -> bool:
    if not isinstance(data, dict) or not data:
        return False
    for value in data.values():
        if not isinstance(value, dict):
            return False
        if "class_type" not in value:
            return False
    return True


def _extract_prompt(workflow_data: dict[str, Any]) -> dict[str, Any]:
    if "prompt" in workflow_data and isinstance(workflow_data["prompt"], dict):
        return workflow_data["prompt"]
    if _looks_like_prompt(workflow_data):
        return workflow_data
    raise RuntimeError(
        "Workflow JSON does not look like a ComfyUI prompt. "
        "Export the workflow from ComfyUI and pass its JSON."
    )


def _load_workflow_json(workflow_json_text: str, workflow_json_path: str) -> dict[str, Any]:
    workflow_json_text = (workflow_json_text or "").strip()
    workflow_json_path = (workflow_json_path or "").strip()
    if workflow_json_text:
        try:
            data = json.loads(workflow_json_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("workflow_json_text is not valid JSON.") from exc
        return _extract_prompt(data)

    if workflow_json_path:
        if not os.path.exists(workflow_json_path):
            raise RuntimeError(f"workflow_json_path not found: {workflow_json_path}")
        with open(workflow_json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return _extract_prompt(data)

    env_path = os.getenv("LTX2_WORKFLOW_JSON", "").strip()
    if env_path:
        if not os.path.exists(env_path):
            raise RuntimeError(f"LTX2_WORKFLOW_JSON not found: {env_path}")
        with open(env_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return _extract_prompt(data)

    bundled = os.path.join(
        os.path.dirname(__file__),
        "workflows",
        "ltx2_official_template.json",
    )
    if os.path.exists(bundled):
        with open(bundled, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return _extract_prompt(data)

    raise RuntimeError(
        "No workflow JSON found. Export the official LTX-2 workflow JSON from "
        "ComfyUI and pass it via workflow_json_path or workflow_json_text."
    )


def _update_prompt_inputs(prompt: dict[str, Any], seed: int, fps: int, frames: int) -> None:
    for node in prompt.values():
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        if "seed" in inputs:
            inputs["seed"] = seed
        if "fps" in inputs:
            inputs["fps"] = int(fps)
        if "frame_rate" in inputs:
            if isinstance(inputs["frame_rate"], int):
                inputs["frame_rate"] = int(fps)
            else:
                inputs["frame_rate"] = float(fps)
        for key in ("num_frames", "frames_number", "length"):
            if key in inputs:
                inputs[key] = int(frames)


def _update_savevideo_nodes(prompt: dict[str, Any], basename: str, chunk_index: int) -> None:
    target_id = os.getenv("LTX2_SAVE_NODE_ID", "").strip()
    if target_id and target_id not in prompt:
        raise RuntimeError(f"LTX2_SAVE_NODE_ID {target_id} not found in workflow prompt.")
    target_ids = {target_id} if target_id else set(prompt.keys())
    for node_id, node in prompt.items():
        if node_id not in target_ids:
            continue
        class_type = str(node.get("class_type", ""))
        if "SaveVideo" not in class_type:
            continue
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            continue
        inputs["filename_prefix"] = f"{basename}_chunk{chunk_index:04d}"
        if "format" in inputs:
            inputs["format"] = "mp4"


def _extract_prompt_id(response_json: dict[str, Any]) -> str:
    prompt_id = response_json.get("prompt_id")
    if isinstance(prompt_id, list) and prompt_id:
        prompt_id = prompt_id[0]
    if not prompt_id:
        raise RuntimeError(f"Unexpected /prompt response: {response_json}")
    return str(prompt_id)


def _collect_history_items(obj: Any, items: list[dict[str, Any]]) -> None:
    if isinstance(obj, dict):
        if "filename" in obj:
            items.append(obj)
        for value in obj.values():
            _collect_history_items(value, items)
    elif isinstance(obj, list):
        for value in obj:
            _collect_history_items(value, items)


def _extract_history_files(history_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries: Iterable[Any]
    if "outputs" in history_payload:
        entries = [history_payload]
    else:
        entries = history_payload.values()
    items: list[dict[str, Any]] = []
    for entry in entries:
        outputs = entry.get("outputs") if isinstance(entry, dict) else None
        if outputs is None:
            continue
        _collect_history_items(outputs, items)
    mp4_items = [item for item in items if str(item.get("filename", "")).lower().endswith(".mp4")]
    wav_items = [item for item in items if str(item.get("filename", "")).lower().endswith(".wav")]
    return mp4_items, wav_items


def _resolve_output_path(
    filename: str,
    subfolder: str | None,
    base_dir: str,
    fallback_roots: Iterable[str],
) -> str:
    subfolder = subfolder or ""
    direct_path = os.path.join(base_dir, subfolder, filename)
    if os.path.exists(direct_path):
        return direct_path
    for root in fallback_roots:
        if not root:
            continue
        candidate = os.path.join(root, base_dir, subfolder, filename)
        if os.path.exists(candidate):
            return candidate
    return direct_path


def _guess_comfy_root() -> str | None:
    root = os.getenv("COMFYUI_ROOT", "").strip()
    if root:
        return root
    return None


def _api_wait_for_outputs(
    api_url: str,
    prompt_id: str,
    timeout_sec: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requests = _get_requests()
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            response = requests.get(f"{api_url}/history/{prompt_id}", timeout=10)
        except Exception:
            time.sleep(0.5)
            continue
        if response.status_code != 200:
            time.sleep(0.5)
            continue
        payload = response.json()
        mp4_items, wav_items = _extract_history_files(payload)
        if mp4_items or wav_items:
            return mp4_items, wav_items
        time.sleep(0.5)
    raise RuntimeError("Timed out waiting for ComfyUI history outputs.")


def api_run_chunk(
    *,
    workflow_json_text: str,
    workflow_json_path: str,
    comfy_api_url: str,
    comfy_output_dir: str,
    seed: int,
    fps: int,
    frames: int,
    basename: str,
    chunk_index: int,
    timeout_sec: int,
) -> tuple[str, str | None]:
    prompt = _load_workflow_json(workflow_json_text, workflow_json_path)
    _update_prompt_inputs(prompt, seed=seed, fps=fps, frames=frames)
    _update_savevideo_nodes(prompt, basename, chunk_index)
    api_url = _discover_api_url(comfy_api_url)

    requests = _get_requests()
    response = requests.post(f"{api_url}/prompt", json={"prompt": prompt}, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Comfy API error: {response.status_code} {response.text}")
    prompt_id = _extract_prompt_id(response.json())

    mp4_items, wav_items = _api_wait_for_outputs(api_url, prompt_id, timeout_sec)
    if not mp4_items:
        raise RuntimeError("ComfyUI history returned no mp4 outputs.")

    base_dir = (comfy_output_dir or "").strip() or os.getenv("COMFY_OUTPUT_DIR", "").strip() or "output"
    fallback_roots = [os.getcwd()]
    comfy_root = _guess_comfy_root()
    if comfy_root:
        fallback_roots.append(comfy_root)

    mp4_item = mp4_items[-1]
    mp4_path = _resolve_output_path(
        str(mp4_item.get("filename")),
        mp4_item.get("subfolder"),
        base_dir,
        fallback_roots,
    )

    wav_path = None
    if wav_items:
        wav_item = wav_items[-1]
        wav_path = _resolve_output_path(
            str(wav_item.get("filename")),
            wav_item.get("subfolder"),
            base_dir,
            fallback_roots,
        )

    return mp4_path, wav_path


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
            },
            "optional": {
                "workflow_json_path": ("STRING", {"default": ""}),
                "workflow_json_text": ("STRING", {"default": "", "multiline": True}),
                "comfy_api_url": ("STRING", {"default": ""}),
                "comfy_output_dir": ("STRING", {"default": ""}),
                "timeout_sec": ("INT", {"default": 0, "min": 0, "max": 86400}),
            },
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
        workflow_json_path: str = "",
        workflow_json_text: str = "",
        comfy_api_url: str = "",
        comfy_output_dir: str = "",
        timeout_sec: int = 0,
    ):
        frames_per_chunk = _adjust_num_frames(frames_per_chunk)
        output_dir = output_dir or "outputs"
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

        chunk_fn = _resolve_direct_callable()
        use_api = chunk_fn is None

        if use_api:
            timeout = timeout_sec or int(os.getenv("LTX2_API_TIMEOUT_SEC", DEFAULT_API_TIMEOUT_SEC))
        else:
            timeout = 0

        chunk_index = 0
        while True:
            chunk_index += 1
            if mode == "commercial" and chunk_index > total_chunks:
                break
            if mode == "continuous" and _should_stop():
                status = "stopped"
                break

            seed = seed_base + (chunk_index - 1) * seed_stride

            if use_api:
                out_path, audio_path = api_run_chunk(
                    workflow_json_text=workflow_json_text,
                    workflow_json_path=workflow_json_path,
                    comfy_api_url=comfy_api_url,
                    comfy_output_dir=comfy_output_dir,
                    seed=seed,
                    fps=fps,
                    frames=frames_per_chunk,
                    basename=basename,
                    chunk_index=chunk_index,
                    timeout_sec=timeout,
                )
            else:
                out_path = os.path.join(output_dir, f"{basename}_chunk{chunk_index:04d}.mp4")
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
                ring = chunk_paths[-max(1, rolling_chunks) :]
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
