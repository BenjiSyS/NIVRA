# Long videos, multi-GPU and shared generation

Contract between the Python executor (`ai/`), the controller (`internal/controller/`), the worker agent and
the Nivra App. Numbers marked *estimate* are calculations, not measurements.

## Why segments

An HD model generates one clip of up to `default_seconds` (8 s at 30 fps = 241 frames) at a time. Anything
longer is built from segments. Plain continuation (each segment conditioned on the previous one) is strictly
sequential: extra computers would not make one video faster. Nivra therefore uses **draft → refine → stitch**:

1. **Draft** (`ai.draft`, one worker, sequential): the whole video is generated at **256×144** with the
   normal continuation chain (cheap: about 1/23 of the 720p token count per segment, *estimate*). This fixes
   composition and motion for the full length.
2. **Refine** (`ai.refine_segment`, N jobs, **parallel on any workers / GPUs**): each segment is generated at
   the target resolution starting from the upscaled, VAE-encoded draft frames with partial noise (SDEdit,
   `strength` 0.55 by default) and with its first and last latent frames anchored to the draft (inpainting
   replacement during sampling). Neighbouring segments share their boundary frame, so they meet in the same
   picture without waiting for each other.
3. **Stitch** (`ai.stitch`, CPU): segments are joined (the duplicate boundary frame dropped, a 4-frame
   crossfade across each seam) into the final MP4.

Videos up to `default_seconds` keep using a single `ai.generate` job (unchanged). Length limit: 30 minutes on the network (the draft is one job that cannot resume yet); 3 hours on a single local GPU.
Quality caveat: coherence over minutes/hours depends on the draft; a weakly trained model drifts.

Cost (*estimate*, RTX 2060-class, Medium): 720p 8 s ≈ 20–40 min per segment; 1 h at 720p ≈ 450 segments
≈ 150–300 GPU-hours, divided by the number of GPUs taking part. 1080p ≈ 2.5×, 1440p ≈ 4–5× the 720p cost
per segment (attention grows faster than the pixel count).

## Segment plan (shared by Python and Go)

`segment_frames = frames_for(default_seconds, fps)` (1 + 8k rule, 241 at 8 s/30 fps). Total frames
`F = round(seconds × fps) + 1`. Segment *i* covers frames `[i·(S−1), min((i+1)·(S−1), F−1)]` with
`S = segment_frames`; the last segment is shorter (rounded up to the 1 + 8k rule internally, then trimmed).
`N = ceil((F − 1) / (S − 1))`.

## Job types (Python `nivra.cli exec`)

Inputs (roles) are the model files as for `ai.generate`: `weights`, `manifest`, `vae_weights`.

| Job | Extra inputs | Params | Artifacts | Summary |
|-----|--------------|--------|-----------|---------|
| `ai.draft` | – | `prompt, seed, style?, sampler?, seconds, fps, draft_width=256, draft_height=144` | `draft_video` (MP4, draft size, all frames), `draft_plan` (JSON: `frames, fps, segment_frames, segments:[{index,start,end}]`) | `frames, segments, seconds_*` |
| `ai.refine_segment` | `draft_video` | `prompt, seed, style?, sampler?, width, height, fps, index, start, end, strength=0.55` | `segment_video` (MP4, target size, frames start..end inclusive, CRF 12) | `index, frames, seconds_*` |
| `ai.stitch` | `segment_000` … `segment_NNN` | `fps, crossfade_frames=4, count` | `video` (final MP4) | `frames, seconds` |

`seed` for segment *i* is `seed + i`. Progress events as for `ai.generate`. All three accept `device`
`cuda:N` from the runner like other jobs.

## Controller

* `POST /projects/{pid}/generate`: HD + `seconds > default_seconds` → creates a **generation** (table
  `generations`, migration 0007) and its draft job, returns `{generation_id, job_id: <draft job>, model_version,
  segments}`. Short videos: unchanged (`job_id`).
* On a job's accepted result (same transaction that completes it): draft done → insert the N
  `ai.refine_segment` jobs (input `draft_video` = the draft's artifact, priority above new generations so
  started videos finish first); all segments done → insert `ai.stitch`; stitch done → generation `completed`
  with `artifact_id`. A child job that fails permanently fails the generation and cancels its other jobs;
  cancelling the generation cancels all its unfinished jobs. Idempotent: a duplicate completion never inserts
  a second set of children (unique `(generation_id, stage, index)`).
* `GET /projects/{pid}/generations/{gid}` → `{id, status (queued|running|completed|failed|cancelled),
  stage (draft|refine|stitch|done), progress 0..1, segments_total, segments_done, workers:[names],
  artifact_id?, failure_reason?, created_at}`; progress = 0.1·draft + 0.85·mean(segment progress) +
  0.05·stitch. `POST .../generations/{gid}/cancel`.
* Width/height limits for HD: up to 2560×1440. `seconds` up to 1800 on the network (3 hours locally).

## Workers with several GPUs

A worker computer runs **one worker slot per CUDA GPU** in the same process (slot 0 is the normal worker;
slot *k* uses its own identity and data folder `<data_dir>/gpu<k>` and is pinned to GPU *k*). Enrolling with a
project code enrolls every slot ("PC name (GPU k)"). The scheduler is unchanged: each slot is a device with one
lease, so N GPUs take N segments at once. CPU-only computers have one slot.

## Nivra App

* History items for long videos follow the generation (status, stage, `segments_done/segments_total`, workers,
  progress, ETA from the aggregated progress) and download the final artifact.
* Local models: the app runs the same three stages itself, refining segments on all local CUDA GPUs in
  parallel (one at a time on CPU).
* Before sending a long video the app shows the segment count and a time estimate.
