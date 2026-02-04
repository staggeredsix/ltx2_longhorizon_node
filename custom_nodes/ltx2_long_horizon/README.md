# LTX2 Long Horizon Controller (ComfyUI Custom Node)

This folder contains a standalone ComfyUI custom node package intended to be copied into a ComfyUI install.

## Install

Use the helper script to symlink and install dependencies:

```
./install_comfy_node.sh /path/to/ComfyUI
```

Manual alternative:

```
ln -s /path/to/gb10_attitude_gen/custom_nodes/ltx2_long_horizon /path/to/ComfyUI/custom_nodes/ltx2_long_horizon
```

## Node

Node name: **LTX2: Long Horizon Controller**

Outputs:
- `final_or_latest_mp4_path`
- `chunk_paths_json`
- `status`

Modes:
- **commercial**: finite generation; chunked render + final concatenated mp4
- **continuous**: infinite-ish; keeps writing `{basename}_latest.mp4` from a ring buffer until stopped

## Wiring to Official LTX-2 Template

This node is an orchestrator. It does not implement LTX diffusion. It expects to run the official LTX-2 workflow
through ComfyUI's HTTP API by default.

### Export the workflow JSON

1. Open the official LTX-2 workflow in ComfyUI.
2. Click **Save (API Format)** or **Export (API)** to get the workflow JSON.
3. Provide the JSON via:
   - `workflow_json_path` (path to the file), or
   - `workflow_json_text` (paste the JSON text into the node input).

### Default (API) usage — no env vars needed

Fill in **workflow_json_path** (or paste **workflow_json_text**) and leave:
- `comfy_api_url` empty to auto-discover (tries 127.0.0.1:8188–8192)
- `comfy_output_dir` empty if ComfyUI uses its default `./output` directory

The node will inject `seed`, `fps`, and `num_frames` into the workflow and update the SaveVideo
`filename_prefix` per chunk.

### Direct-call (optional)

If you want to bypass the API, set an environment variable pointing to a callable that generates one chunk:

```
LTX2_LONG_HORIZON_CALLABLE=module:callable
```

The callable is expected to accept keyword args like:

```
model, vae, audio_vae, positive, negative, fps, num_frames, seed,
chain_strength, chain_frames, drop_prefix, blend_frames, reset_interval, output_path
```

It should return either:
- `(video_path, audio_path)` tuple, or
- `{"video_path": "...", "audio_path": "..."}` dict, or
- a string path to the mp4.

## Constraints

- Frames per chunk must be `1 + 8k` (the controller snaps/validates).
- Resolution should be aligned (64/128). This controller does not change resolution.
- `stop_file` can be used to halt continuous mode; create the file to stop.

## Output Behavior

Commercial:
- computes total chunks with `ceil((target_seconds * fps) / frames_per_chunk)`
- concatenates chunk mp4s at the end
- muxes audio if wav chunks are available

Continuous:
- keeps only `rolling_chunks` in a ring buffer for the latest mp4
- updates `{basename}_latest.mp4` after each chunk
