<script lang="ts">
  // A status always reads as icon + label, never color alone.
  let { tone, label }: { tone: 'good' | 'warning' | 'serious' | 'critical' | 'neutral'; label: string } = $props();
  const icon = { good: '✓', warning: '!', serious: '!', critical: '✕', neutral: '•' };
</script>

<span class="badge s-{tone}"><span aria-hidden="true">{icon[tone]}</span>{label}</span>
