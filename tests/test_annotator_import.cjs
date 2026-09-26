const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../web/annotator.html'), 'utf8');
const source = html.slice(html.indexOf('    async function syncPicoDownload()'), html.indexOf('    async function redoCurrentSession()'));

test('completed-session label accepts current and legacy server statuses', () => {
  const helper = html.slice(html.indexOf('    function isProcessedSession('), html.indexOf('    function nextStep('));
  const context = vm.createContext({});
  vm.runInContext(helper, context);
  assert.equal(context.isProcessedSession({processing_status: 'processed'}), true);
  assert.equal(context.isProcessedSession({processing_status: '\u5df2\u5904\u7406'}), true);
  assert.equal(context.isProcessedSession({processing_status: 'unprocessed'}), false);
});

async function run(ok) {
  const logs = [], loaded = [], statuses = [];
  const state = {sessions: [{session_id: 'current'}], sessionId: 'current', episodes: [{unsaved: true}]};
  const context = vm.createContext({
    state,
    el: {syncDateInput: {value: '20260916'}, syncSourceInput: {value: ''}, syncBtn: {}},
    setStatus: text => statuses.push(text), setLog: lines => logs.push(lines.join('\n')),
    commandText: cmd => cmd.join(' '), nextStep: text => text,
    updateSessionOptions() {}, syncWarningLines: () => [],
    loadSession: async id => loaded.push(id),
    api: async (url, options) => {
      assert.equal(JSON.parse(options.body).date, '20260916');
      // A stale payload from an older server must also be safe to display.
      return {ok, command: ['python', 'sync'], stderr: ok ? '' : 'Input/output error',
        rows: [{session_id: 'old', paired: 'True', tracking_status: 'copied'}],
        imported_session_ids: ['old'], pending_session_ids: ['old'], sessions: [{session_id: 'old'}]};
    },
  });
  vm.runInContext(source, context);
  await context.syncPicoDownload();
  return {state, logs, loaded, statuses};
}

test('failed import cannot display stale success or replace unsaved annotation context', async () => {
  const {state, logs, loaded, statuses} = await run(false);
  assert.match(logs.at(-1), /Input\/output error/);
  assert.doesNotMatch(logs.at(-1), /imported:|matched sessions:|WARNING|Imported/);
  assert.equal(state.sessionId, 'current');
  assert.equal(state.sessions[0].session_id, 'current');
  assert.equal(state.episodes[0].unsaved, true);
  assert.deepEqual(loaded, []);
  assert.equal(statuses.at(-1), 'Import failed');
});

test('successful import displays the result and selects the imported session', async () => {
  const {logs, loaded, statuses} = await run(true);
  assert.match(logs.at(-1), /imported: old/);
  assert.deepEqual(loaded, ['old']);
  assert.match(statuses.at(-1), /Import complete: 1/);
});
