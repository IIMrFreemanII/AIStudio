<script lang="ts">
  import { saveSettings } from '../lib/api';
  import type { ServerSettings } from '../lib/protocol';

  let { settings }: { settings: ServerSettings } = $props();
  let budget = $state(0);
  let hz = $state(0);
  let side = $state(0);
  let saved = $state<string | null>(null);
  let dirty = $state(false);
  $effect(() => {
    if (!dirty) { budget = Math.round(settings.budgetFraction * 100); hz = settings.previewHz; side = settings.previewMaxSide; }
  });

  async function save(e: SubmitEvent) {
    e.preventDefault();
    try {
      await saveSettings({ budgetFraction: budget / 100, previewHz: hz, previewMaxSide: side });
      saved = 'Saved';
      dirty = false;
    } catch (err) {
      saved = `✕ ${err}`;
    }
  }
</script>

<form class="card" onsubmit={save} oninput={() => { dirty = true; saved = null; }}>
  <div class="card-head"><h2>Settings</h2><button class="primary" disabled={!dirty}>Save</button></div>
  <label><span>Model memory budget</span><input type="range" min="30" max="95" bind:value={budget} /><span class="num">{budget}% of RAM</span></label>
  <label><span>Default preview rate</span><input type="number" min="0" max="30" step="0.5" bind:value={hz} /><span class="muted">frames/s</span></label>
  <label><span>Default preview size</span><input type="number" min="64" max="2048" step="64" bind:value={side} /><span class="muted">px</span></label>
  {#if saved}<p class="small secondary">{saved}</p>{/if}
  <dl>
    <dt>Listening on</dt><dd class="mono">{settings.host}:{settings.port}</dd>
    <dt>Models</dt><dd class="mono">{settings.modelsDirectory}</dd>
    <dt>Data</dt><dd class="mono">{settings.dataDirectory}</dd>
    <dt>Token</dt><dd class="secondary">Not shown here. Rotate it on the server: replace the token file and restart.</dd>
  </dl>
</form>

<style>
  label { display: grid; grid-template-columns: 160px minmax(0, 1fr) auto; gap: 10px; align-items: center; margin-bottom: 8px; font-size: 13px; }
  @media (max-width: 520px) { label { grid-template-columns: 1fr auto; } label > span:first-child { grid-column: 1 / -1; } }
  input[type='number'] { width: 100%; }
  dl { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 4px 12px; margin: 12px 0 0; font-size: 13px; }
  dt { color: var(--text-muted); }
  dd { margin: 0; overflow-wrap: anywhere; }
  .small { font-size: 12px; }
</style>
