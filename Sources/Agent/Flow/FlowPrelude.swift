/// The JavaScript every flow script runs on top of: the public API (`agent`, `parallel`, `pipeline`, `phase`, `log`,
/// `verifyEach`, `implementAndCheck`) built on the few native functions FlowRuntime installs
/// (`__agent`, `__phase`, `__log`).
enum FlowPrelude {
    static let source = #"""
var DEFAULT_VERIFY = "Check this finding adversarially against the code: try to prove it wrong. It is real only if you can confirm it in the files. Say whether it is real and why.";

function __stringify(v) { return v === undefined ? undefined : JSON.stringify(v); }

function phase(title) { __phase(String(title)); }
function log(msg) { __log(String(msg)); }

function agent(prompt, opts) {
  if (typeof prompt !== 'string') throw new TypeError('agent(prompt, opts): prompt must be a string');
  opts = opts == null ? {} : opts;
  if (typeof opts !== 'object') throw new TypeError('agent(prompt, opts): opts must be an object');
  var mode = opts.mode === undefined ? 'read' : opts.mode;
  if (mode !== 'read' && mode !== 'edit') throw new TypeError("agent: mode must be 'read' or 'edit'");
  var maxSteps = opts.maxSteps;
  if (maxSteps != null) {
    if (typeof maxSteps !== 'number' || !isFinite(maxSteps) || maxSteps < 1) throw new TypeError('agent: maxSteps must be a positive number');
    maxSteps = Math.floor(maxSteps);
  }
  var hasSchema = opts.schema != null;
  if (hasSchema && typeof opts.schema !== 'object') throw new TypeError('agent: schema must be a JSON schema object');
  var optsJSON = JSON.stringify({
    label: opts.label == null ? undefined : String(opts.label),
    phase: opts.phase == null ? undefined : String(opts.phase),
    schema: hasSchema ? opts.schema : undefined,
    mode: mode,
    maxSteps: maxSteps == null ? undefined : maxSteps,
    effort: opts.effort == null ? undefined : String(opts.effort)
  });
  return new Promise(function (resolve, reject) {
    __agent(prompt, optsJSON,
      function (kind, text, json, branch) {
        if (kind === 'json') resolve(JSON.parse(json));
        else if (kind === 'edit') resolve({ text: text, branch: branch == null ? null : branch });
        else resolve(text);
      },
      function (message) { reject(new Error(message)); });
  });
}

function parallel(thunks) {
  if (!Array.isArray(thunks)) throw new TypeError("parallel takes functions: () => agent(...)");
  for (var i = 0; i < thunks.length; i++) {
    if (typeof thunks[i] !== 'function') throw new TypeError("parallel takes functions: () => agent(...)");
  }
  return Promise.all(thunks.map(function (f) { return f(); }));
}

function pipeline(items, ...stages) {
  return Promise.all(Array.from(items).map(async function (item, i) {
    var v = item;
    for (var s = 0; s < stages.length; s++) v = await stages[s](v, item, i);
    return v;
  }));
}

function verifyEach(items, opts = {}) {
  var schema = { type: 'object', properties: { real: { type: 'boolean' }, reason: { type: 'string' } }, required: ['real', 'reason'] };
  return Promise.all(Array.from(items).map(async function (item, i) {
    var verdict = await agent((opts.prompt || DEFAULT_VERIFY) + "\n\nThe finding:\n" + JSON.stringify(item, null, 2), {
      label: (opts.label || 'verify') + ':' + (i + 1),
      phase: opts.phase,
      schema: schema
    });
    var plain = item !== null && typeof item === 'object' && !Array.isArray(item);
    return plain ? Object.assign({}, item, { verdict: verdict }) : { item: item, verdict: verdict };
  }));
}

async function implementAndCheck(task, opts = {}) {
  if (!opts || !opts.check) throw new TypeError('implementAndCheck needs opts.check: the command that verifies the change');
  var attempts = opts.attempts === undefined || opts.attempts === null ? 2 : opts.attempts;
  var prompt = task + "\n\nWhen the change is made, run `" + opts.check + "` and fix what fails, up to " + attempts +
    " more tries. End your reply with a line that is exactly PASSED or FAILED.";
  var r = await agent(prompt, { mode: 'edit', label: opts.label || 'implement', phase: opts.phase });
  return { branch: r.branch, passed: /\bPASSED\s*$/.test(r.text), report: r.text };
}
"""#
}
