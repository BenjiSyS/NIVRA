# Web development (Nivra Studio and Nivra Worker site)

Both websites live in `web/` and are built into `internal/webui/dist/`, which the Go binaries embed
(`//go:embed all:dist`). The controller serves the build with a fallback to `studio.html`; the worker
serves the same build with a fallback to `worker.html`.

| Entry | Source | Talks to |
|-------|--------|----------|
| Studio (`studio.html`) | `web/src/studio` | `/api/v1/*` on the controller (cookie session + `X-Nivra-CSRF`, one `EventSource` per project) |
| Worker (`worker.html`) | `web/src/worker` | `/local/api/*` on the worker (`X-Nivra-Local: 1` on non-GET) |
| Shared | `web/src/shared` | hash router, fetch wrapper, formatting, UI primitives, SVG line chart, virtualised table, base CSS |

The authoritative contract is `docs/api/API.md` (section 3 for Studio, section 4 for the worker).

## Build

```sh
cd web
npm ci
npm run build      # tsc --noEmit (strict) && vite build && postbuild
```

`postbuild` restores `internal/webui/dist/.keep` (Vite empties the output directory) and checks that
`studio.html`, `worker.html` and `assets/` exist. After a build, rebuild the Go binary so it embeds the new
files: `go build ./cmd/nivra`.

`npm run typecheck` runs only the TypeScript check.

## Develop with hot reload

Run a controller on `127.0.0.1:8740` and a worker on `127.0.0.1:8741`, then:

```sh
cd web
npm run dev        # http://127.0.0.1:5173/studio.html and /worker.html
```

The Vite dev server proxies `/api` to the controller and `/local` to the worker, so the pages behave as they
do when embedded. Sign in to Studio on the dev origin once; the session cookie is per host, not per port.

## Rules the code follows

* The Content-Security-Policy is strict (`default-src 'self'`, no external fonts, scripts or CDNs, no inline
  scripts). Everything is bundled; fonts are the system stack.
* Only the dependencies in `package.json` (React 18, Vite, TypeScript, the React plugin). The router, SVG chart
  and virtualised table are in `src/shared`.
* No mock data. Every control calls a real endpoint; empty states say what is missing.
* Passwords are never stored. The CSRF token is held in memory only; `localStorage` holds UI preferences
  (selected project, table density).
* Live data: `src/studio/store.ts` keeps a per-project cache, applies `worker.updated`, `worker.metrics`,
  `job.updated`, `command.updated`, `training.updated`, `result.finalized` and `log` events, and refetches
  everything on `resync`, reconnect or error. Components subscribe per domain and re-render at most every
  ~120 ms; charts additionally throttle to one render per second.

## Adding a Studio page

1. Create `src/studio/pages/<Name>.tsx` with a default export.
2. Add a `lazy()` import and a `case` in `src/studio/App.tsx`, plus an entry in `NAV`.
3. Use `api` from `src/studio/api.ts` for requests and `useLive(domain)` / `useRev(domain)` for cached state.
