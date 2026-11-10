// SHA-256 over the dashboard's sources: for each file (sorted by path relative to web/dashboard),
// "<path>\0<bytes>\0". The Swift test (DashboardBundleTests) computes the same thing.
import { createHash } from 'node:crypto';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const root = new URL('..', import.meta.url).pathname;
export const sourceFiles = ['index.html', 'package.json', 'vite.config.ts', 'src', 'public'];

function walk(p) {
  return statSync(p).isDirectory() ? readdirSync(p).flatMap((n) => walk(join(p, n))) : [p];
}

export function sourceHash() {
  const files = sourceFiles.flatMap((f) => walk(join(root, f)))
    .map((p) => relative(root, p))
    .filter((p) => !p.endsWith('.test.ts'))
    .sort();
  const h = createHash('sha256');
  for (const f of files) {
    h.update(f);
    h.update('\0');
    h.update(readFileSync(join(root, f)));
    h.update('\0');
  }
  return h.digest('hex');
}
