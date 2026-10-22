# Nivra App interface

The Nivra App is served by the local worker at `http://127.0.0.1:8741/app` (loopback only). The API it
uses is described in `docs/architecture/APP.md`.

## First run

Open the app. If no controller or API token is stored, or the controller cannot be reached, the
connection screen opens:

1. Enter the **Controller URL** (for example `https://studio.example.com`).
2. Paste an **API token**. Create one in Nivra Studio, Settings, API tokens. The token is verified, stored
   only on this computer, and never shown again. Leave the field empty later to keep the stored token.
3. **Save and verify** shows the user, role and project, or the error.

## Making a video

Type a prompt in the box at the bottom. Enter sends, Shift+Enter adds a line. Your prompt appears as a
message and the reply is a card: progress and the worker doing the job, then the video with Download and
the parameters used. Failed jobs show the error and a Retry button that re-submits the same parameters.

* **Model picker**: lists the controller's models. The default is preselected. HD models carry an "HD"
  badge; the others are small research models and their output is rough.
* **Options**: seed (random if empty, shown on the result), steps, guidance (cfg scale), frames per
  second. For HD models also resolution (1280x720, 854x480, 448x256), length (2, 4 or 8 seconds) and
  sampler (ab2 or euler). Empty steps and guidance use the model's defaults.

## History

The left sidebar lists earlier videos (newest first) with status and age. It refreshes every 2 seconds
while a job is unfinished and every 15 seconds otherwise. History and videos live in this computer's Nivra
data folder.

## This computer as a worker

The sidebar link shows the worker's state, project, CPU, memory, GPU and AI executor status, with
Start, Pause, Resume and Stop buttons. A computer only receives jobs while the worker is running.

## Developing

`cd web && npm run dev` serves the pages with a proxy to the local worker; `npm run build` writes
`internal/webui/dist` (entries `studio.html`, `worker.html`, `app.html`).
