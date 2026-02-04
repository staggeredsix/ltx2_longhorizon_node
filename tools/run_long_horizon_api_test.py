#!/usr/bin/env python3
import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from custom_nodes.ltx2_long_horizon.ffmpeg_utils import concat_and_mux
from custom_nodes.ltx2_long_horizon.nodes import api_run_chunk


def _resolve_workflow_arg(workflow_json: str) -> tuple[str, str]:
    if not workflow_json:
        return "", ""
    workflow_json = workflow_json.strip()
    if os.path.exists(workflow_json):
        return "", workflow_json
    try:
        json.loads(workflow_json)
    except json.JSONDecodeError as exc:
        raise SystemExit("--workflow-json must be a file path or valid JSON text.") from exc
    return workflow_json, ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LTX2 long-horizon chunks via ComfyUI API.")
    parser.add_argument("--api-url", default="", help="ComfyUI API URL (leave empty for auto-discovery).")
    parser.add_argument("--workflow-json", default="", help="Workflow JSON file path or inline JSON.")
    parser.add_argument("--basename", default="ltx2_api_test", help="Output basename.")
    parser.add_argument("--output-dir", default="outputs", help="Final output directory.")
    parser.add_argument("--chunks", type=int, default=1, help="Number of chunks to run.")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second.")
    parser.add_argument("--frames-per-chunk", type=int, default=73, help="Frames per chunk.")
    parser.add_argument("--timeout-sec", type=int, default=600, help="API timeout per chunk.")
    args = parser.parse_args()

    workflow_text, workflow_path = _resolve_workflow_arg(args.workflow_json)

    chunk_paths = []
    wav_paths = []

    for chunk_index in range(1, max(1, args.chunks) + 1):
        mp4_path, wav_path = api_run_chunk(
            workflow_json_text=workflow_text,
            workflow_json_path=workflow_path,
            comfy_api_url=args.api_url,
            comfy_output_dir="",
            seed=10 + (chunk_index - 1),
            fps=args.fps,
            frames=args.frames_per_chunk,
            basename=args.basename,
            chunk_index=chunk_index,
            timeout_sec=args.timeout_sec,
        )
        chunk_paths.append(mp4_path)
        if wav_path:
            wav_paths.append(wav_path)

    os.makedirs(args.output_dir, exist_ok=True)
    final_path = os.path.join(args.output_dir, f"{args.basename}.mp4")
    concat_and_mux(chunk_paths, wav_paths, args.fps, final_path)
    print(final_path)


if __name__ == "__main__":
    main()
