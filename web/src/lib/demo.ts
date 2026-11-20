import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

// Assets come from web/scripts/build-demo.mjs or from the pinned release (web_deploy.yml).
// Without them the site has no demo page, link or sitemap entry.
const assets = join(process.cwd(), 'static/demo-assets');

export interface DemoPiece {
  id: string;
  score: string;
  config?: string;
  duration_sec: number;
  title: string;
  authored: string;
  generated: string;
  authored_harmony: string | null;
  scoreText: string;
  configText: string | null;
}

export interface Demo {
  build: { dir: string; tag: string | null; version: string; commit: string; dirty: boolean; wasmBytes: number };
  base: string;
  pieces: DemoPiece[];
}

let cached: Demo | null | undefined;
export function demo(): Demo | null {
  if (cached !== undefined) return cached;
  const buildFile = join(assets, 'build.json');
  if (!existsSync(buildFile)) return (cached = null);
  const build = JSON.parse(readFileSync(buildFile, 'utf8'));
  const dir = join(assets, build.dir);
  const read = (name: string) => readFileSync(join(dir, 'pieces', name), 'utf8');
  const pieces = JSON.parse(readFileSync(join(dir, 'pieces.json'), 'utf8')).map((piece: DemoPiece) => ({
    ...piece, scoreText: read(piece.score), configText: piece.config ? read(piece.config) : null,
  }));
  return (cached = { build, base: `/demo-assets/${build.dir}/`, pieces });
}
