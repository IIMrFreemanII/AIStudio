<script lang="ts">
  import { preloadModel, unloadModel } from '../lib/api';
  import { ago, bytes, mem } from '../lib/format';
  import type { DashboardState } from '../lib/protocol';

  let { data }: { data: DashboardState } = $props();
  let busy = $state<string | null>(null);
  let error = $state<string | null>(null);

  async function act(name: string, f: () => Promise<unknown>) {
    busy = name;
    error = null;
    try { await f(); } catch (e) { error = String(e); } finally { busy = null; }
  }
  const budget = $derived(data.info.memory.budget);
</script>

<div class="card">
  <div class="card-head">
    <h2>Models</h2>
    <button disabled={!data.loadedModels.some((m) => !m.pinned)} onclick={() => act('all', () => unloadModel())}>Unload all</button>
  </div>
  {#if error}<p class="s-critical small" role="alert">✕ {error}</p>{/if}
  <h3 class="sub">Loaded</h3>
  {#if data.loadedModels.length}
    <div class="scroll">
    <table>
      <thead><tr><th>Model</th><th class="num">Resident</th><th>Last used</th><th></th></tr></thead>
      <tbody>
        {#each data.loadedModels as m (m.key)}
          <tr>
            <td>{m.name}{#if m.pinned}<span class="badge">in use</span>{/if}</td>
            <td class="num">{mem(m.bytes)}</td>
            <td class="secondary">{ago(m.lastUsed)}</td>
            <td class="num"><button disabled={m.pinned || busy !== null} onclick={() => act(m.key, () => unloadModel(m.key))}>Unload</button></td>
          </tr>
        {/each}
      </tbody>
    </table>
    </div>
  {:else}
    <p class="empty">None. Models load on first use and stay until memory is needed.</p>
  {/if}
  <h3 class="sub">Installed</h3>
  {#if data.info.models.length}
    <div class="scroll">
      <table>
        <thead><tr><th>Folder</th><th class="num">Weights</th><th>Fits budget</th><th></th></tr></thead>
        <tbody>
          {#each data.info.models as m (m.name)}
            <tr>
              <td class="mono">{m.name}</td>
              <td class="num">{bytes(m.bytes)}</td>
              <td>{#if m.bytes <= budget}<span class="s-good">✓ yes</span>{:else}<span class="s-serious">! over budget</span>{/if}</td>
              <td class="num">
                {#if m.loaded}<span class="muted">loaded</span>{:else}
                  <button disabled={busy !== null} onclick={() => act(m.name, () => preloadModel(m.name))}>{busy === m.name ? 'Loading…' : 'Preload'}</button>
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {:else}
    <p class="empty">No models in {data.settings.modelsDirectory}. Add them with <code class="mono">mlxcomfy-cli download</code>.</p>
  {/if}
</div>

<style>
  .sub { font-size: 12px; font-weight: 600; color: var(--text-muted); margin: 10px 0 4px; }
  .small { font-size: 12px; }
  .badge { margin-left: 6px; }
</style>
