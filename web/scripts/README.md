# Live demo build

The live demo page (`/demo/`, `/ja/demo/`) performs the pieces in `samples/demo/`
in the visitor's browser. Its design is `docs/design-notes/demo-site.md`.

The page exists only when `web/static/demo-assets/` is present. Without it the site
builds as before: no demo route, navigation link or sitemap entry.

## Local build

From the repository root, with Rust's `wasm32-unknown-unknown` target and
`wasm-bindgen-cli 0.2.130` installed:

```sh
node web/scripts/build-demo.mjs
cd web
npm ci
npm run build
```

`build-demo.mjs` calls `crates/conchordal-web/build.sh` for the engine and GUI. It
respects `CARGO`, `CARGO_TARGET_DIR` and `CARGO_BUILD_JOBS`; on blanche use the
assigned cargo wrapper and resource limits in `crates/conchordal-web/README.md`.
It writes `web/static/demo-assets/build.json` and one directory named after the tag
(or `v<version>-dev`), the commit and, for uncommitted changes, `-dirty` with a
content hash, so a browser never mixes cached files of two local builds. The
directory holds the wasm and JS modules, the pieces with `pieces.json`, and the
third-party notices (`licenses.txt`). Only the wasm `name` section is removed.
`--skip-wasm` reuses wasm built by this script from unchanged sources.

The generated directory is ignored by git and must not be committed.

## Checks

```sh
node web/scripts/demo-smoke.mjs web/dist 20      # every piece, 20 s each, headless Chrome
node web/metrics-worker/test.mjs                 # the counter endpoint keeps only the allowlist
```

The smoke check plays each piece with Chrome's fake audio output. It fails on a
dropout, a display that receives no snapshot, a CSP violation, an exception, a
cross-origin request, a counter field outside the allowlist, or a POST while
sending is turned off.

## Release and deployment

- Publishing a GitHub release runs `.github/workflows/demo_assets.yml`. It builds
  the assets from the tag, runs both checks, attaches
  `conchordal-demo-<tag>.tar.gz` to the release and reruns the site deployment.
- `web/src/data/demo-release.json` names the release the site serves. Changing that
  tag is the act of publishing a new demo version; `null` serves no demo.
- `web_deploy.yml` downloads that release's archive before building the site. If it
  is missing (for example, right after tagging) the site builds without the demo.

## Counter endpoint

`web/metrics-worker/` is a Cloudflare Worker routed at
`www.conchordal.org/demo/metrics`; every other path stays on GitHub Pages. It
rebuilds each record from the allowlist in `src/record.js`, reduces the
User-Agent to browser and OS families, keeps no IP address and writes one
Analytics Engine data point (kept for three months). Deploy it by hand, with the
author's Cloudflare login and permission:

```sh
cd web/metrics-worker
npx wrangler deploy
```

For a local look, serve `web/dist` over HTTP, for example
`python3 -m http.server 8765 --bind 127.0.0.1 --directory web/dist`, and open
`/demo/`. The local server does not accept the counter POST; the page ignores that.
