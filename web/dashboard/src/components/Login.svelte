<script lang="ts">
  import { login, HttpError } from '../lib/api';

  let { onLogin }: { onLogin: () => void } = $props();
  let token = $state('');
  let error = $state<string | null>(null);
  let busy = $state(false);

  async function submit(e: SubmitEvent) {
    e.preventDefault();
    busy = true;
    error = null;
    try {
      await login(token.trim());
      token = '';
      onLogin();
    } catch (err) {
      error = err instanceof HttpError && err.status === 401 ? 'That token is not valid for this server.' : String(err);
    } finally {
      busy = false;
    }
  }
</script>

<main class="login">
  <form class="card" onsubmit={submit}>
    <h1>MLXComfy Server</h1>
    <p class="secondary">Sign in with the server's token: run <code class="mono">mlxcomfy-cli serve --show-token</code> on the server.</p>
    <label>
      <span class="muted">Token</span>
      <input type="password" autocomplete="current-password" bind:value={token} required minlength="16" />
    </label>
    {#if error}<p class="s-critical" role="alert">✕ {error}</p>{/if}
    <button class="primary" disabled={busy || token.trim().length < 16}>{busy ? 'Signing in…' : 'Sign in'}</button>
    <p class="muted small">The session lasts 12 hours. The token is not stored in this browser.</p>
  </form>
</main>

<style>
  .login { min-height: 100vh; display: grid; place-items: center; padding: 16px; }
  form { width: min(400px, 100%); display: flex; flex-direction: column; gap: 12px; }
  h1 { font-size: 20px; }
  label { display: flex; flex-direction: column; gap: 4px; }
  p { margin: 0; }
  .small { font-size: 12px; }
  code { background: var(--surface-2); padding: 1px 5px; border-radius: 4px; }
</style>
