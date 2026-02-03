# LTX2 Long Horizon Controller (ComfyUI Custom Node)

This folder contains a standalone ComfyUI custom node package intended to be copied into a ComfyUI install.

## Install

Symlink or copy into your ComfyUI `custom_nodes` directory:

```
./install_comfy_node.sh /path/to/ComfyUI
```

Or manually:

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

This node is an orchestrator. It does not implement LTX diffusion. It expects to call the official LTX-2 template's
"one clip" generator.

### Direct-call (preferred)

Set an environment variable pointing to a callable that can generate one chunk:

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

### API fallback (supported)

If the direct call can't be resolved, the node can call ComfyUI's local API and run a workflow JSON:

```
LTX2_WORKFLOW_JSON=/path/to/official_ltx2_workflow.json
COMFY_API_URL=http://127.0.0.1:8188
```

The workflow should already include the official LTX-2 template nodes. The controller updates `seed`, `fps`,
and `num_frames` inputs if they exist in the workflow JSON.

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
