const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const html = fs.readFileSync(path.join(__dirname, '../web/annotator.html'), 'utf8');
const source = html.slice(html.indexOf('    async function runExport('), html.indexOf('    const G1_STAGE_LABEL'));

async function run({missing = true, accept = false, dryRun = false, checkOK = true} = {}) {
  const calls = [], confirmations = [];
  const context = vm.createContext({
    state: {sessionId: '20260914_162219', jobs: []},
    el: {repoIdInput: {value: 'test/repo'}, outputDirInput: {value: '/test/output'}, exportMeta: {}},
    saveAnnotations: async () => {calls.push('save');},
    setStatus() {}, setLog() {}, renderJobList() {},
    api: async (url, options) => {
      assert.equal(url, '/api/export_lerobot');
      const payload = JSON.parse(options.body);
      calls.push(payload);
      return {job_id: String(calls.length)};
    },
    monitorJob: async (id, options) => ({ok: checkOK, stdout: JSON.stringify({
      requires_hand_confirmation: missing, warnings: missing ? [{message: 'Missing hand observations; fill with zeros'}] : []
    })}),
    window: {confirm: message => {confirmations.push(message); return accept;}},
  });
  vm.runInContext(source, context);
  await context.runExport(dryRun);
  return {calls, confirmations};
}

test('cancel warning saves annotations but never starts actual export', async () => {
  const {calls, confirmations} = await run();
  assert.equal(calls[0], 'save');
  assert.equal(calls.length, 2);
  assert.equal(calls[1].dry_run, true);
  assert.equal(confirmations.length, 1);
});
test('accept warning explicitly authorizes missing hands for actual export', async () => {
  const {calls, confirmations} = await run({accept: true});
  assert.equal(calls.length, 3);
  assert.equal(calls[2].dry_run, false);
  assert.equal(calls[2].allow_missing_hands, true);
  assert.equal(confirmations.length, 1);
});
test('complete hands export without a missing-hand confirmation', async () => {
  const {calls, confirmations} = await run({missing: false});
  assert.equal(calls.length, 3);
  assert.equal(calls[2].allow_missing_hands, false);
  assert.equal(confirmations.length, 0);
});
test('check button never exports or prompts for approval', async () => {
  const {calls, confirmations} = await run({dryRun: true});
  assert.equal(calls.length, 2);
  assert.equal(confirmations.length, 0);
});
test('failed coverage or schema check cannot be overridden by hand confirmation', async () => {
  const {calls, confirmations} = await run({checkOK: false, accept: true});
  assert.equal(calls.length, 2);
  assert.equal(confirmations.length, 0);
});
