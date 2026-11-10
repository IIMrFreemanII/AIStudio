<script lang="ts">
  import { mem, rate } from '../lib/format';
  import { Metric, type SeriesRing } from '../lib/protocol';
  import Chart from './Chart.svelte';

  let { ring, revision, budget }: { ring: SeriesRing; revision: number; budget: number } = $props();
  const pct = (v: number) => `${Math.round(v)}%`;
  const ms = (v: number) => (v < 10 ? `${v.toFixed(1)} ms` : `${Math.round(v)} ms`);
  const perSec = (v: number) => (v < 10 ? v.toFixed(1) : String(Math.round(v)));
</script>

<div class="card">
  <div class="card-head"><h2>Performance · last 10 minutes</h2></div>
  <div class="grid">
    <Chart title="Memory" {ring} {revision} format={(v) => mem(v, 0)} refLine={{ value: budget, label: 'Budget' }}
           series={[{ label: 'MLX active', metric: Metric.memoryActive, slot: 1 }, { label: 'Cache', metric: Metric.memoryCache, slot: 2 }]} />
    <Chart title="Utilization" {ring} {revision} format={pct} max={100}
           series={[{ label: 'GPU', metric: Metric.gpuUtilization, slot: 1 }, { label: 'Server CPU', metric: Metric.cpuProcess, slot: 2 }]} />
    <Chart title="Network" {ring} {revision} format={rate}
           series={[{ label: 'Out', metric: Metric.bytesOutPerSecond, slot: 1 }, { label: 'In', metric: Metric.bytesInPerSecond, slot: 2 }]} />
    <Chart title="Event latency (publish → socket)" {ring} {revision} format={ms}
           series={[{ label: 'p50', metric: Metric.eventLatencyP50, slot: 1 }, { label: 'p99', metric: Metric.eventLatencyP99, slot: 2 }]} />
    <Chart title="Events per second" {ring} {revision} format={perSec}
           series={[{ label: 'Events', metric: Metric.eventsPerSecond, slot: 1 }]} />
    <Chart title="Preview frames per second" {ring} {revision} format={perSec}
           series={[{ label: 'Previews', metric: Metric.previewsPerSecond, slot: 1 }]} />
  </div>
</div>

<style>
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px 20px; }
</style>
