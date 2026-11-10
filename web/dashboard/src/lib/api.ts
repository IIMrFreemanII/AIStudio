// REST calls with the dashboard's session cookie; changes carry the CSRF header the server checks.

function csrf(): string {
  const m = document.cookie.match(/(?:^|;\s*)mlxc_csrf=([^;]+)/);
  return m ? decodeURIComponent(m[1]) : '';
}

export class HttpError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
  }
}

export async function api<T = unknown>(path: string, init: { method?: string; body?: unknown } = {}): Promise<T> {
  const method = init.method ?? 'GET';
  const headers: Record<string, string> = {};
  if (method !== 'GET' && method !== 'HEAD') headers['X-AIStudio-CSRF'] = csrf();
  if (init.body !== undefined) headers['Content-Type'] = 'application/json';
  const r = await fetch(path, {
    method,
    headers,
    body: init.body === undefined ? undefined : JSON.stringify(init.body),
    credentials: 'same-origin',
  });
  if (!r.ok) throw new HttpError(r.status, (await r.text()) || r.statusText);
  const type = r.headers.get('content-type') ?? '';
  return (type.includes('json') ? await r.json() : undefined) as T;
}

export const login = (token: string) => api('/v1/session', { method: 'POST', body: { token } });
export const logout = () => api('/v1/session', { method: 'DELETE' });
export const cancelRun = (run: number) => api(`/v1/runs/${run}/cancel`, { method: 'POST' });
export const clearQueue = () => api('/v1/queue', { method: 'DELETE' });
export const preloadModel = (name: string) => api('/v1/models/preload', { method: 'POST', body: { name } });
export const unloadModel = (key?: string) => api('/v1/models/unload', { method: 'POST', body: key ? { key } : {} });
export const purgeInputs = () => api<{ freed: number }>('/v1/files', { method: 'DELETE' });
export const saveSettings = (s: { budgetFraction?: number; previewHz?: number; previewMaxSide?: number }) =>
  api('/v1/settings', { method: 'POST', body: s });
