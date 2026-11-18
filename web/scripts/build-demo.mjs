// Assemble the live demo's assets into web/static/demo-assets/.
// The release workflow packs that directory; the site build reads it when present.
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { copyFile, mkdir, readFile, readdir, rename, rm, writeFile } from 'node:fs/promises';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const host = join(root, 'crates/conchordal-web/www');
const pieces = join(root, 'samples/demo');
const destination = join(root, 'web/static/demo-assets');
const skipWasm = process.argv.includes('--skip-wasm');
if (process.argv.slice(2).some(arg => arg !== '--skip-wasm')) {
  throw new Error('Usage: node web/scripts/build-demo.mjs [--skip-wasm]');
}
const git = (...args) => execFileSync('git', args, { cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();

// Reuse only artifacts built by this script from the same engine and GUI sources.
const hash = createHash('sha256');
async function fingerprint(digest, base, path) {
  const entries = await readdir(join(base, path), { withFileTypes: true });
  for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    const relative = path ? `${path}/${entry.name}` : entry.name;
    if (entry.isDirectory()) await fingerprint(digest, base, relative);
    else { digest.update(relative); digest.update(await readFile(join(base, relative))); }
  }
}
for (const directory of ['src', 'gui/src', 'crates/conchordal-web/src']) await fingerprint(hash, root, directory);
for (const path of ['Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'gui/Cargo.toml', 'gui/Cargo.lock',
  'crates/conchordal-web/Cargo.toml', 'crates/conchordal-web/Cargo.lock', 'crates/conchordal-web/build.sh']) {
  if (existsSync(join(root, path))) { hash.update(path); hash.update(await readFile(join(root, path))); }
}
const sourceHash = hash.digest('hex');
const stamp = join(host, 'pkg/demo-source.sha256');
if (skipWasm) {
  if (!existsSync(stamp) || (await readFile(stamp, 'utf8')).trim() !== sourceHash) {
    throw new Error('Engine or GUI sources changed. Run without --skip-wasm.');
  }
} else {
  execFileSync('bash', [join(root, 'crates/conchordal-web/build.sh')], { cwd: root, stdio: 'inherit' });
  await writeFile(stamp, `${sourceHash}\n`);
}

// Keep executable sections byte-identical; omit only the optional function names.
function withoutNames(bytes) {
  let cursor = 8;
  const chunks = [bytes.subarray(0, 8)];
  const uleb = () => {
    let value = 0, shift = 0, byte;
    do { byte = bytes[cursor++]; value += (byte & 127) * 2 ** shift; shift += 7; } while (byte & 128);
    return value;
  };
  while (cursor < bytes.length) {
    const start = cursor, id = bytes[cursor++], size = uleb(), end = cursor + size;
    let name;
    if (id === 0) {
      const length = uleb();
      name = bytes.subarray(cursor, cursor + length).toString('utf8');
    }
    if (id !== 0 || name !== 'name') chunks.push(bytes.subarray(start, end));
    cursor = end;
  }
  return Buffer.concat(chunks);
}

// Third-party notices for every crate linked into either wasm module.
function licenseNotices() {
  const packages = new Map();
  for (const manifest of ['crates/conchordal-web/Cargo.toml', 'gui/Cargo.toml']) {
    const metadata = JSON.parse(execFileSync('cargo', ['metadata', '--format-version', '1', '--locked',
      '--filter-platform', 'wasm32-unknown-unknown', '--manifest-path', join(root, manifest)],
    { cwd: root, encoding: 'utf8', maxBuffer: 256 * 1024 * 1024 }));
    const byId = new Map(metadata.packages.map(p => [p.id, p]));
    const nodes = new Map(metadata.resolve.nodes.map(n => [n.id, n]));
    const pending = [metadata.resolve.root];
    const seen = new Set();
    while (pending.length) {
      const id = pending.pop();
      if (seen.has(id)) continue;
      seen.add(id);
      for (const dep of nodes.get(id).deps) {
        if (dep.dep_kinds.some(kind => kind.kind === null)) pending.push(dep.pkg);
      }
    }
    for (const id of seen) {
      const p = byId.get(id);
      if (p.source) packages.set(`${p.name} ${p.version}`, p);
    }
  }
  const sections = [];
  for (const key of [...packages.keys()].sort()) {
    const p = packages.get(key);
    const dir = dirname(p.manifest_path);
    const files = readdirSync(dir).filter(name => /^(LICEN[CS]E|COPYING|NOTICE|COPYRIGHT)/i.test(name)).sort();
    const texts = files.map(name => readFileSync(join(dir, name), 'utf8').trim());
    sections.push([`== ${key} (${p.license ?? p.license_file ?? 'license not stated'})`,
      p.repository ?? '', ...texts].filter(Boolean).join('\n\n'));
  }
  return `Third-party software in the Conchordal browser instrument\n\n${sections.join('\n\n\n')}\n`;
}

const cargo = await readFile(join(root, 'Cargo.toml'), 'utf8');
const version = cargo.match(/^version\s*=\s*"([^"]+)"/m)[1];
const commit = git('rev-parse', '--short=8', 'HEAD');
const dirty = git('status', '--porcelain', '--', '.', ':!web/static/demo-assets') !== '';
let tag = process.env.DEMO_TAG || null;
if (!tag) { try { tag = git('describe', '--exact-match', '--tags', 'HEAD'); } catch { tag = null; } }
let dir = `${tag ?? `v${version}-dev`}-${commit}${dirty ? '-dirty' : ''}`;
const out = join(destination, dir);

await rm(destination, { recursive: true, force: true });
await mkdir(join(out, 'pieces'), { recursive: true });
for (const name of ['player.js', 'engine-worker.js', 'audio-worklet.js']) {
  await copyFile(join(host, name), join(out, name));
}
let wasmBytes = 0;
for (const directory of ['pkg', 'gui-pkg']) {
  await mkdir(join(out, directory));
  for (const name of await readdir(join(host, directory))) {
    if (name.endsWith('.js')) await copyFile(join(host, directory, name), join(out, directory, name));
    if (name.endsWith('.wasm')) {
      const bytes = withoutNames(await readFile(join(host, directory, name)));
      await writeFile(join(out, directory, name), bytes);
      wasmBytes += bytes.length;
      console.log(`${directory}/${name}: ${bytes.length} bytes`);
    }
  }
}
const manifest = JSON.parse(await readFile(join(pieces, 'pieces.json'), 'utf8'));
for (const piece of manifest) {
  if (!/^[a-z0-9-]+$/.test(piece.id)) throw new Error(`Invalid piece id: ${piece.id}`);
  for (const key of ['score', 'config']) {
    if (piece[key] === undefined) continue;
    if (piece[key].includes('/')) throw new Error(`${piece.id}: ${key} must be a file in samples/demo`);
    await copyFile(join(pieces, piece[key]), join(out, 'pieces', piece[key]));
  }
}
await writeFile(join(out, 'pieces.json'), `${JSON.stringify(manifest, null, 2)}\n`);
await writeFile(join(out, 'licenses.txt'), licenseNotices());
if (dirty) {
  // Uncommitted builds share one commit; a content hash keeps a browser from mixing
  // cached files of two builds under one URL.
  const content = createHash('sha256');
  await fingerprint(content, out, '');
  dir = `${dir}-${content.digest('hex').slice(0, 8)}`;
  await rename(out, join(destination, dir));
}
await writeFile(join(destination, 'build.json'),
  `${JSON.stringify({ dir, tag, version, commit, dirty, wasmBytes }, null, 2)}\n`);
console.log(`Demo assets ${dir}: ${manifest.length} pieces`);
