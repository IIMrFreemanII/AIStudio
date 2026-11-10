<script lang="ts">
  import uPlot from 'uplot';
  import 'uplot/dist/uPlot.min.css';
  import { onDestroy, onMount } from 'svelte';
  import type { SeriesRing } from '../lib/protocol';

  interface Series { label: string; metric: number; slot: 1 | 2 | 3; scale?: number }
  let {
    title, ring, series, revision, format, max, refLine,
  }: {
    title: string; ring: SeriesRing; series: Series[]; revision: number;
    format: (v: number) => string; max?: number; refLine?: { value: number; label: string };
  } = $props();

  let el: HTMLDivElement;
  let plot: uPlot | null = null;
  let observer: ResizeObserver | null = null;
  const media = matchMedia('(prefers-color-scheme: dark)');

  function color(name: string): string {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function data(): uPlot.AlignedData {
    const t = ring.timeView();
    const ys = series.map((s) => {
      const col = ring.view(s.metric);
      if (!s.scale || s.scale === 1) return col;
      const out = new Float32Array(col.length);
      for (let i = 0; i < col.length; i++) out[i] = col[i] * s.scale;
      return out;
    });
    const out: (Float64Array | Float32Array | number[])[] = [t, ...ys];
    if (refLine) out.push(new Array(t.length).fill(refLine.value));
    return out as unknown as uPlot.AlignedData;
  }

  function build() {
    plot?.destroy();
    const grid = { stroke: color('--grid'), width: 1 };
    const axis = { stroke: color('--text-muted'), grid, ticks: { show: false }, font: '11px system-ui' };
    const opts: uPlot.Options = {
      width: el.clientWidth || 300,
      height: 150,
      cursor: { drag: { x: false, y: false }, points: { size: 8 } },
      legend: { live: true },
      scales: { x: { time: true }, y: { range: (_u, _min, mx) => [0, Math.max(max ?? 0, mx * 1.1) || 1] } },
      axes: [
        { ...axis, space: 80, values: (_u, vals) => vals.map((v) => new Date(v * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })) },
        { ...axis, size: 56, values: (_u, vals) => vals.map((v) => format(v)) },
      ],
      series: [
        { label: '', value: (_u, v) => (v == null ? '' : new Date(v * 1000).toLocaleTimeString()) },
        ...series.map((s) => ({
          label: s.label,
          stroke: color(`--series-${s.slot}`),
          width: 2,
          points: { show: false },
          value: (_u: uPlot, v: number | null) => (v == null ? '—' : format(v)),
        })),
        ...(refLine ? [{ label: refLine.label, stroke: color('--text-muted'), width: 1, dash: [4, 4], points: { show: false },
                         value: () => format(refLine!.value) }] : []),
      ],
    };
    plot = new uPlot(opts, data(), el);
  }

  onMount(() => {
    build();
    observer = new ResizeObserver(() => plot?.setSize({ width: el.clientWidth, height: 150 }));
    observer.observe(el);
    media.addEventListener('change', build);
  });
  onDestroy(() => {
    observer?.disconnect();
    media.removeEventListener('change', build);
    plot?.destroy();
  });

  $effect(() => {
    void revision;
    plot?.setData(data());
  });
</script>

<figure class="chart">
  <figcaption>{title}</figcaption>
  <div bind:this={el}></div>
</figure>

<style>
  .chart { margin: 0; min-width: 0; }
  figcaption { font-size: 12px; font-weight: 600; color: var(--text-secondary); margin-bottom: 4px; }
  .chart :global(.u-legend) { font-size: 11px; color: var(--text-secondary); text-align: left; }
  .chart :global(.u-legend .u-marker) { border-radius: 2px; }
  .chart :global(.u-legend .u-series:first-child) { display: none; }
</style>
