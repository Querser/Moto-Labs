import assert from "node:assert/strict";
import test from "node:test";

await import("../../app/static/js/finish_line_editor.js");
const line = globalThis.MotoFinishLine;

test("endpoint A and endpoint B move independently", () => {
  const initial = { x1: .2, y1: .3, x2: .8, y2: .7 };
  assert.deepEqual(line.moveEndpoint(initial, "start", { x: .1, y: .2 }), {
    x1: .1, y1: .2, x2: .8, y2: .7,
  });
  assert.deepEqual(line.moveEndpoint(initial, "end", { x: .9, y: .6 }), {
    x1: .2, y1: .3, x2: .9, y2: .6,
  });
});

test("segment hit testing works at the center and another segment point", () => {
  const initial = { x1: .1, y1: .5, x2: .9, y2: .5 };
  assert.equal(line.hitTest(initial, { x: .5, y: .5 }, 1000, 500), "segment");
  assert.equal(line.hitTest(initial, { x: .7, y: .51 }, 1000, 500), "segment");
  assert.equal(line.hitTest(initial, { x: .5, y: .8 }, 1000, 500), null);
});

test("whole-line dragging preserves length and angle", () => {
  const initial = { x1: .2, y1: .3, x2: .7, y2: .6 };
  const moved = line.translateLine(initial, .15, -.12);
  assert.ok(Math.abs(line.lineLength(initial) - line.lineLength(moved)) < 1e-12);
  assert.ok(Math.abs(line.lineAngle(initial) - line.lineAngle(moved)) < 1e-12);
});

test("whole-line clamp keeps both endpoints inside without distortion", () => {
  const initial = { x1: .1, y1: .2, x2: .8, y2: .7 };
  const moved = line.translateLine(initial, .8, -.8);
  assert.ok(Math.abs(moved.x1 - .3) < 1e-12);
  assert.equal(moved.y1, 0);
  assert.equal(moved.x2, 1);
  assert.ok(Math.abs(moved.y2 - .5) < 1e-12);
  assert.ok(Math.abs(line.lineLength(initial) - line.lineLength(moved)) < 1e-12);
  assert.ok(Math.abs(line.lineAngle(initial) - line.lineAngle(moved)) < 1e-12);
});

test("remote polling cannot revert an active or dirty local edit", () => {
  const state = line.createState();
  line.initialize(state, { x1: .1, y1: .6, x2: .9, y2: .6 });
  state.dragging = true;
  line.setEditable(state, { x1: .2, y1: .5, x2: .9, y2: .6 });
  assert.equal(line.applyRemote(state, { x1: .1, y1: .6, x2: .9, y2: .6 }), false);
  assert.equal(state.editable.x1, .2);
  state.dragging = false;
  assert.equal(line.applyRemote(state, { x1: .1, y1: .6, x2: .9, y2: .6 }), false);
  assert.equal(state.editable.x1, .2);
});

test("stale save response is ignored and failed save preserves the edit", () => {
  const state = line.createState();
  line.initialize(state, line.DEFAULT_LINE);
  line.setEditable(state, { x1: .2, y1: .6, x2: .8, y2: .6 });
  const first = line.beginSave(state);
  line.setEditable(state, { x1: .3, y1: .5, x2: .7, y2: .5 });
  const second = line.beginSave(state);
  assert.equal(line.completeSave(state, first, first.line), false);
  assert.equal(state.editable.x1, .3);
  assert.equal(line.failSave(state, second), true);
  assert.equal(state.dirty, true);
  assert.equal(state.editable.x1, .3);
});

test("successful save restores exactly after a simulated reload", () => {
  const firstState = line.createState();
  line.initialize(firstState, line.DEFAULT_LINE);
  line.setEditable(firstState, { x1: .22, y1: .44, x2: .77, y2: .66 });
  const token = line.beginSave(firstState);
  assert.equal(line.completeSave(firstState, token, token.line), true);
  const reloaded = line.createState();
  line.initialize(reloaded, firstState.persisted);
  assert.deepEqual(reloaded.editable, token.line);
});

test("object-fit contain geometry remains correct after responsive resize", () => {
  const landscape = line.containedImageRect(1000, 500, 1080, 1920);
  assert.ok(Math.abs(landscape.left - 359.375) < 1e-10);
  assert.ok(Math.abs(landscape.top) < 1e-10);
  assert.ok(Math.abs(landscape.width - 281.25) < 1e-10);
  assert.ok(Math.abs(landscape.height - 500) < 1e-10);
  const portrait = line.containedImageRect(500, 1000, 1080, 1920);
  assert.ok(Math.abs(portrait.left) < 1e-10);
  assert.ok(Math.abs(portrait.top - 55.55555555555554) < 1e-10);
  assert.ok(Math.abs(portrait.width - 500) < 1e-10);
  assert.ok(Math.abs(portrait.height - 888.8888888888889) < 1e-10);
});
