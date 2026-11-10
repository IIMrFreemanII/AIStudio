<script lang="ts">
  import type { DashboardState } from '../lib/protocol';
  import { Metric, type SeriesRing } from '../lib/protocol';
  import { duration, mem } from '../lib/format';
  import Status from './Status.svelte';

  let { data, ring, revision }: { data: DashboardState; ring: SeriesRing; revision: number } = $props();

  const latest = $derived.by(() => {
    void revision;
    const i = ring.count - 1;
    return (m: number) => (i >= 0 ? ring.columns[m][i] : 0);
  });
  const memory = $derived(data.info.memory);
  const usedFraction = $derived(memory.total ? latest(Metric.memoryActive) / memory.total : 0);
  const thermalTone = $derived(
    ({ nominal: 'good', fair: 'warning', serious: 'serious', critical: 'critical' } as const)[
      data.thermalState as 'nominal' | 'fair' | 'serious' | 'critical'] ?? 'neutral');
  const running = $derived(data.queue.filter((r) => r.status === 'running').length);
  const waiting = $derived(data.queue.filter((r) => r.status === 'waiting').length);
  let now = $state(Date.now() / 1000);
  $effect(() => { const t = setInterval(() => (now = Date.now() / 1000), 1000); return () => clearInterval(t); });
</script>

<section class="tiles" aria-label="Overview">
  <div class="tile card">
    <span class="label">Memory (MLX active)</span>
    <span class="value">{mem(latest(Metric.memoryActive))}</span>
    <div class="meter" role="meter" aria-valuemin="0" aria-valuemax={memory.total} aria-valuenow={latest(Metric.memoryActive)}
         aria-label="Memory used">
      <div class="fill" style:width="{Math.min(100, usedFraction * 100)}%"></div>
      <div class="budget" style:left="{Math.min(100, (memory.budget / memory.total) * 100)}%" title="Model budget"></div>
    </div>
    <span class="sub">cache {mem(latest(Metric.memoryCache))} · budget {mem(memory.budget)} of {mem(memory.total)}</span>
  </div>
  <div class="tile card">
    <span class="label">GPU (whole Mac)</span>
    <span class="value">{Math.round(latest(Metric.gpuUtilization))}%</span>
    <span class="sub">{data.info.gpu} · server CPU {Math.round(latest(Metric.cpuProcess))}%</span>
  </div>
  <div class="tile card">
    <span class="label">Queue</span>
    <span class="value">{running ? 'Running' : 'Idle'}</span>
    <span class="sub">{waiting} waiting · {data.history.length} recent runs</span>
  </div>
  <div class="tile card">
    <span class="label">Health</span>
    <span class="value small"><Status tone={thermalTone} label={`Thermal ${data.thermalState}`} /></span>
    <span class="sub">up {duration(now - data.started)} · {data.clients.filter((c) => c.role === 'app').length} app clients</span>
  </div>
</section>

<style>
  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }
  .tile { display: flex; flex-direction: column; gap: 4px; }
  .label { font-size: 12px; color: var(--text-muted); }
  .value { font-size: 26px; font-weight: 600; font-variant-numeric: tabular-nums; line-height: 1.15; }
  .value.small { font-size: 15px; padding: 6px 0; }
  .sub { font-size: 12px; color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .meter { position: relative; height: 6px; background: var(--surface-2); border-radius: 3px; margin: 4px 0 2px; }
  .fill { height: 100%; background: var(--series-1); border-radius: 3px; }
  .budget { position: absolute; top: -3px; bottom: -3px; width: 2px; background: var(--text-muted); }
</style>
