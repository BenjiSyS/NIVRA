# Nivra App (Nivra.exe)

`Nivra.exe` is the desktop app. It runs the local worker agent and serves a chat-style generation
interface at `http://127.0.0.1:8741/app` (loopback only), shown in an app window (Microsoft Edge or
Chrome in app mode; falls back to the default browser). One controller coordinates; every app
installation can generate through it and can also contribute its hardware as a worker.

"Chat" means: you type a prompt, the video appears as the reply. There is no conversational language
model; the text goes to the video model.

## Local app API (loopback only; non-GET requests need `X-Nivra-Local: 1`)

| Route | Body → Response |
|-------|-----------------|
| `GET /local/api/app/config` | → `{controller_url, has_token, user: {username, role} \| null, project: {id, name} \| null, connected, error}` |
| `PUT /local/api/app/config` | `{controller_url, token?, project?}` → same as GET. The token is verified against the controller (`/auth/me`) and stored only in the app data folder (owner-only file); it is never returned to the page. Empty `token` keeps the stored one; `"-"` clears it. |
| `GET /local/api/app/models` | → `{models: [{id, name, architecture, family: "small" \| "hd", active, parameter_count}]}` (from the controller) |
| `POST /local/api/app/generate` | `{prompt, model_version?, seed?, steps?, cfg_scale?, fps?, width?, height?, seconds?, sampler?}` → `{id}` (controller job id) |
| `GET /local/api/app/history` | → `{items: [{id, prompt, params, created_at, status (queued \| running \| downloading \| done \| failed \| cancelled), progress 0..1, stage, worker, error, video_url \| null, seconds_total}]}` newest first |
| `POST /local/api/app/history/{id}/cancel` | cancels the controller job |
| `DELETE /local/api/app/history/{id}` | removes the local entry and its video file |
| `GET /local/api/app/videos/{id}.mp4` | the downloaded video (Range supported) |

Existing worker routes (`/local/api/status`, `start`, `pause`, `resume`, `stop`, `connect`, `settings`,
`logs`) stay available; the app shows them as "This computer as a worker".

## Behaviour
* After `POST generate` the app polls the controller job, downloads the MP4 when complete into
  `<data_dir>/videos/`, and keeps history in `<data_dir>/app-history.json` (survives restarts; polling
  resumes for unfinished items).
* The API token is a personal Nivra API token (Studio → Settings → API tokens).
* Hardware is detected by the worker agent (NVIDIA GPU via `nvidia-smi`, PyTorch CUDA, CPU); the controller
  only sends jobs that fit each machine's RAM/VRAM budget (HD models declare `min_vram_mb`).
