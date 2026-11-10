<script lang="ts">
  import { purgeInputs } from '../lib/api';
  import { ago, bytes, rate } from '../lib/format';
  import type { DashboardState } from '../lib/protocol';

  let { data }: { data: DashboardState } = $props();
  let message = $state<string | null>(null);
</script>

<div class="card">
  <div class="card-head"><h2>Clients &amp; transfers</h2></div>
  {#if data.clients.length}
    <div class="scroll">
      <table>
        <thead><tr><th>Client</th><th>Address</th><th>Connected</th><th class="num">In</th><th class="num">Out</th><th class="num">Previews</th></tr></thead>
        <tbody>
          {#each data.clients as c (c.session)}
            <tr>
              <td>{c.name} <span class="badge">{c.role}</span></td>
              <td class="mono">{c.address}</td>
              <td class="secondary">{ago(c.connected)}</td>
              <td class="num">{bytes(c.bytesIn)}</td>
              <td class="num">{bytes(c.bytesOut)}</td>
              <td class="num">{c.previewHz ? `${c.previewHz}/s` : 'off'}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {:else}
    <p class="empty">No clients connected.</p>
  {/if}
  <h3 class="sub">Transfers</h3>
  {#if data.transfers.length}
    <div class="scroll">
    <table>
      <tbody>
        {#each data.transfers as t (t.id)}
          <tr>
            <td>{t.direction === 'upload' ? '↑' : '↓'} <span class="mono">{t.name.slice(0, 16)}…</span></td>
            <td class="num">{bytes(t.bytes)}{t.total ? ` of ${bytes(t.total)}` : ''}</td>
            <td class="num">{rate(t.bytesPerSecond)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
    </div>
  {:else}
    <p class="empty">No transfers in progress.</p>
  {/if}
  <h3 class="sub">Storage</h3>
  <div class="scroll">
  <table>
    <tbody>
      <tr>
        <td>Uploaded inputs (cache)</td><td class="num">{bytes(data.inputCacheBytes)}</td>
        <td class="num"><button class="danger" onclick={async () => { const r = await purgeInputs(); message = `Freed ${bytes(r.freed)}`; }}>Purge</button></td>
      </tr>
      {#each data.disks as d (d.path)}
        <tr><td class="secondary small path">{d.path}</td><td class="num">{bytes(d.free)} free</td><td class="num muted">of {bytes(d.total, 0)}</td></tr>
      {/each}
    </tbody>
  </table>
  </div>
  {#if message}<p class="secondary small">{message}</p>{/if}
</div>

<style>
  .sub { font-size: 12px; font-weight: 600; color: var(--text-muted); margin: 12px 0 4px; }
  .small { font-size: 12px; }
  .path { overflow-wrap: anywhere; }
</style>
