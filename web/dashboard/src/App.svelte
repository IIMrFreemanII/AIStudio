<script lang="ts">
  import { onDestroy } from 'svelte';
  import { logout } from './lib/api';
  import { Live } from './lib/live.svelte';
  import Clients from './components/Clients.svelte';
  import LiveRun from './components/LiveRun.svelte';
  import Login from './components/Login.svelte';
  import Logs from './components/Logs.svelte';
  import Models from './components/Models.svelte';
  import Overview from './components/Overview.svelte';
  import Performance from './components/Performance.svelte';
  import Runs from './components/Runs.svelte';
  import Settings from './components/Settings.svelte';
  import Status from './components/Status.svelte';

  let signedIn = $state<boolean | null>(null);
  let live = $state<Live | null>(null);

  async function check() {
    const r = await fetch('/v1/state', { credentials: 'same-origin' }).catch(() => null);
    signedIn = r?.ok ?? false;
    if (signedIn) start();
  }
  function start() {
    live?.close();
    const l = new Live();
    l.connect();
    live = l;
  }
  $effect(() => { if (live?.connection === 'unauthorized') { live.close(); live = null; signedIn = false; } });
  check();
  onDestroy(() => live?.close());

  const s = $derived(live?.state);
  const current = $derived(s?.queue.find((r) => r.status === 'running'));
  const connTone = $derived(live?.connection === 'open' ? 'good' : live?.connection === 'connecting' ? 'warning' : 'critical');
</script>

{#if signedIn === false}
  <Login onLogin={() => { signedIn = true; start(); }} />
{:else if live}
  <header>
    <div class="title">
      <h1>MLXComfy Server</h1>
      {#if s}<span class="secondary">{s.info.hostName} · {s.info.chip} · {Math.round(s.info.memory.total / 2 ** 30)} GB · v{s.info.serverVersion}</span>{/if}
    </div>
    <div class="right">
      <Status tone={connTone} label={live.connection === 'open' ? 'Live' : live.connection === 'connecting' ? 'Connecting' : 'Reconnecting'} />
      <button onclick={async () => { await logout().catch(() => {}); live?.close(); live = null; signedIn = false; }}>Sign out</button>
    </div>
  </header>
  {#if live.serverError}<p class="banner s-critical" role="alert">✕ {live.serverError}</p>{/if}
  {#if s}
    <main>
      <Overview data={s} ring={live.metrics} revision={live.revision} />
      <div class="two">
        <LiveRun {live} run={current} />
        <Models data={s} />
      </div>
      <Runs queue={s.queue} history={s.history} />
      <Performance ring={live.metrics} revision={live.revision} budget={s.info.memory.budget} />
      <div class="two">
        <Clients data={s} />
        <Settings settings={s.settings} />
      </div>
      <Logs logs={live.logs} />
    </main>
  {:else}
    <p class="empty pad">Connecting to the server…</p>
  {/if}
{/if}

<style>
  header {
    display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap;
    padding: 12px 16px; background: var(--surface-1); border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 2;
  }
  .title { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; min-width: 0; }
  h1 { font-size: 16px; }
  .title span { font-size: 12px; }
  .right { display: flex; align-items: center; gap: 10px; }
  main { display: flex; flex-direction: column; gap: 12px; padding: 12px 16px 32px; max-width: 1400px; margin: 0 auto; }
  .two { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 460px), 1fr)); gap: 12px; }
  .banner { margin: 8px 16px 0; }
  .pad { padding: 16px; }
</style>
