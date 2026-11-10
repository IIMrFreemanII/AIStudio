<script lang="ts">
  import { cancelRun } from '../lib/api';
  import { duration } from '../lib/format';
  import { NodeState, type Live } from '../lib/live.svelte';
  import type { RunSummary } from '../lib/protocol';

  let { live, run }: { live: Live; run: RunSummary | undefined } = $props();
  const names = ['waiting', 'running', 'cached', 'done', 'failed'];

  const rows = $derived.by(() => {
    void live.revision;
    if (!run) return [];
    const r = live.runs.get(run.run);
    return run.nodes.map((id, i) => ({
      id, type: run.nodeTypes[i], state: r?.state[i] ?? 0, step: r?.step[i] ?? 0, total: r?.total[i] ?? 0,
      seconds: r?.seconds[i] ?? 0, error: r?.errors.get(i), status: r?.status.get(i),
    }));
  });
  const current = $derived(rows.find((r) => r.state === NodeState.running));
  let busy = $state(false);
  let now = $state(Date.now() / 1000);
  $effect(() => { const t = setInterval(() => (now = Date.now() / 1000), 500); return () => clearInterval(t); });
</script>

<div class="card live">
  <div class="card-head">
    <h2>Live run</h2>
    {#if run}
      <button class="danger" disabled={busy} onclick={async () => { busy = true; try { await cancelRun(run.run); } finally { busy = false; } }}>
        Cancel run {run.run}
      </button>
    {/if}
  </div>
  {#if !run}
    <p class="empty">Nothing running. Runs queued from MLXComfyUI show here with live previews.</p>
  {:else}
    <div class="body">
      <div class="preview">
        {#if live.previewURL && live.previewRun === run.run}
          <img src={live.previewURL} alt="Live preview of run {run.run}, step {live.previewStep[0]} of {live.previewStep[1]}" />
        {:else}
          <div class="placeholder muted">Preview appears during sampling</div>
        {/if}
      </div>
      <div class="info">
        <p><strong>{run.workflow || 'Untitled'}</strong> <span class="muted">from {run.client}</span></p>
        <p class="secondary">
          {#if current}
            {current.id} ({current.type}){current.total ? ` · step ${current.step} of ${current.total}` : ''}{current.status ? ` · ${current.status}` : (current.total ? '' : '…')}
          {:else}
            Starting…
          {/if}
          · {duration(now - (run.started ?? now))} elapsed
        </p>
        {#if current && current.total}
          <progress max={current.total} value={current.step}></progress>
        {/if}
        <div class="scroll">
          <table>
            <thead><tr><th>Node</th><th>Type</th><th>State</th><th class="num">Time</th></tr></thead>
            <tbody>
              {#each rows as r (r.id)}
                <tr class:active={r.state === NodeState.running}>
                  <td class="mono">{r.id}</td>
                  <td>{r.type}</td>
                  <td class:s-critical={r.state === NodeState.failed} title={r.error}>
                    {r.state === NodeState.failed ? '✕ ' : r.state === NodeState.done || r.state === NodeState.cached ? '✓ ' : ''}{names[r.state]}
                  </td>
                  <td class="num">{r.seconds ? duration(r.seconds) : ''}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .live { container-type: inline-size; }
  .body { display: grid; grid-template-columns: minmax(0, 240px) minmax(0, 1fr); gap: 16px; }
  @container (max-width: 620px) { .body { grid-template-columns: 1fr; } .preview { max-width: 320px; } }
  .preview { aspect-ratio: 1; background: var(--surface-2); border-radius: 8px; overflow: hidden; display: grid; place-items: center; }
  .preview img { width: 100%; height: 100%; object-fit: contain; }
  .placeholder { font-size: 12px; padding: 12px; text-align: center; }
  .info p { margin: 0 0 4px; }
  progress { width: 100%; height: 6px; accent-color: var(--series-1); margin: 4px 0 8px; }
  tr.active td { background: var(--accent-soft); }
</style>
