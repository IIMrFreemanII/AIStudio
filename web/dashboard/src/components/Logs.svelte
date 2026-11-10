<script lang="ts">
  import { clock } from '../lib/format';
  import type { LogLine } from '../lib/protocol';

  let { logs }: { logs: LogLine[] } = $props();
  let level = $state<'all' | 'warning' | 'error'>('all');
  let paused = $state(false);
  let frozen = $state<LogLine[]>([]);
  let box: HTMLDivElement;

  const order = { debug: 0, info: 1, warning: 2, error: 3 };
  const shown = $derived.by(() => {
    const src = paused ? frozen : logs;
    const min = level === 'all' ? 0 : order[level];
    return src.filter((l) => order[l.level] >= min).slice(-300);
  });
  $effect(() => {
    void shown;
    if (!paused && box) queueMicrotask(() => (box.scrollTop = box.scrollHeight));
  });
</script>

<div class="card">
  <div class="card-head">
    <h2>Logs</h2>
    <div class="controls">
      <select bind:value={level} aria-label="Log level">
        <option value="all">All</option><option value="warning">Warnings and errors</option><option value="error">Errors</option>
      </select>
      <button onclick={() => { if (!paused) frozen = logs.slice(); paused = !paused; }}>{paused ? 'Resume' : 'Pause'}</button>
    </div>
  </div>
  <div class="log mono" bind:this={box} role="log" aria-live="polite">
    {#each shown as l, i (i + ':' + l.time)}
      <div class="line {l.level}"><span class="muted">{clock(l.time)}</span> {l.level === 'error' ? '✕' : l.level === 'warning' ? '!' : ' '} {l.message}</div>
    {:else}
      <div class="muted">No log lines.</div>
    {/each}
  </div>
</div>

<style>
  .controls { display: flex; gap: 6px; }
  .log { height: 240px; overflow: auto; background: var(--surface-2); border-radius: 8px; padding: 8px 10px; }
  .line { white-space: pre-wrap; word-break: break-word; }
  .line.error { color: var(--critical); }
  .line.warning { color: var(--serious); }
</style>
