<script lang="ts">
  import { cancelRun, clearQueue } from '../lib/api';
  import { clock, duration, mem } from '../lib/format';
  import type { RunSummary } from '../lib/protocol';
  import Status from './Status.svelte';

  let { queue, history }: { queue: RunSummary[]; history: RunSummary[] } = $props();
  const waiting = $derived(queue.filter((r) => r.status === 'waiting'));
  const tone = { done: 'good', failed: 'critical', cancelled: 'neutral', running: 'neutral', waiting: 'neutral' } as const;
</script>

<div class="card">
  <div class="card-head">
    <h2>Queue &amp; history</h2>
    <button class="danger" disabled={!waiting.length} onclick={() => clearQueue()}>Clear queue ({waiting.length})</button>
  </div>
  {#if waiting.length}
    <div class="scroll">
      <table>
        <thead><tr><th>Run</th><th>Workflow</th><th>Client</th><th>Queued</th><th></th></tr></thead>
        <tbody>
          {#each waiting as r (r.run)}
            <tr>
              <td class="num">{r.run}</td><td>{r.workflow || 'Untitled'}</td><td class="secondary">{r.client}</td>
              <td class="secondary">{clock(r.created)}</td>
              <td class="num"><button onclick={() => cancelRun(r.run)}>Cancel</button></td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
  {#if history.length}
    <div class="scroll">
      <table>
        <thead><tr><th>Run</th><th>Workflow</th><th>Status</th><th class="num">Time</th><th class="num">Peak</th><th>Outputs</th><th>Finished</th></tr></thead>
        <tbody>
          {#each history as r (r.run)}
            <tr>
              <td class="num">{r.run}</td>
              <td>{r.workflow || 'Untitled'}<br /><span class="muted small">{r.client}</span></td>
              <td title={r.error ?? ''}><Status tone={tone[r.status]} label={r.status} />
                {#if r.error}<div class="s-critical small err">{r.error}</div>{/if}</td>
              <td class="num">{duration(r.seconds)}</td>
              <td class="num">{r.peakMemory ? mem(r.peakMemory) : '—'}</td>
              <td class="outs">
                {#each r.outputs.filter((o) => o.kind === 'image').slice(0, 3) as o (o.name)}
                  <img src="/v1/runs/{r.run}/outputs/{encodeURIComponent(o.name)}" alt="Output {o.name} of run {r.run}" loading="lazy" />
                {/each}
                {#each r.outputs.filter((o) => o.kind !== 'image') as o (o.name)}
                  <a href="/v1/runs/{r.run}/outputs/{encodeURIComponent(o.name)}" class="small">{o.kind}</a>
                {/each}
              </td>
              <td class="secondary small">{r.finished ? clock(r.finished) : ''}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {:else if !waiting.length}
    <p class="empty">No runs yet.</p>
  {/if}
</div>

<style>
  .small { font-size: 12px; }
  .err { max-width: 320px; white-space: normal; }
  .outs { display: flex; gap: 4px; align-items: center; flex-wrap: wrap; }
  .outs img { width: 40px; height: 40px; object-fit: cover; border-radius: 4px; background: var(--surface-2); }
  .outs a { color: var(--accent); }
  .scroll + .scroll { margin-top: 12px; }
</style>
