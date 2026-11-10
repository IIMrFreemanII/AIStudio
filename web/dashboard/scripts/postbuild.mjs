// After `vite build`: pre-compress text assets (.br, .gz) so the server sends them as-is, and record a hash of the
// sources so a test can tell when the committed bundle is older than its sources.
import { createHash } from 'node:crypto';
import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { brotliCompressSync, constants, gzipSync } from 'node:zlib';
import { sourceHash } from './source-hash.mjs';

const out = new URL('../../../Sources/MLXComfyServer/Resources/dashboard/', import.meta.url).pathname;

function walk(dir) {
  return readdirSync(dir).flatMap((n) => {
    const p = join(dir, n);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });
}

let raw = 0, br = 0;
for (const file of walk(out)) {
  if (!/\.(js|css|html|svg|json)$/.test(file) || file.endsWith('build-info.json')) continue;
  const data = readFileSync(file);
  if (data.length < 1024) continue;
  const b = brotliCompressSync(data, { params: { [constants.BROTLI_PARAM_QUALITY]: 11 } });
  writeFileSync(file + '.br', b);
  writeFileSync(file + '.gz', gzipSync(data, { level: 9 }));
  raw += data.length;
  br += b.length;
  console.log(`  ${relative(out, file)}  ${(data.length / 1024).toFixed(1)} KB → br ${(b.length / 1024).toFixed(1)} KB`);
}
writeFileSync(join(out, 'build-info.json'), JSON.stringify({ sourceHash: sourceHash(), built: new Date().toISOString() }, null, 2) + '\n');
console.log(`dashboard: ${(raw / 1024).toFixed(0)} KB, ${(br / 1024).toFixed(1)} KB brotli`);
