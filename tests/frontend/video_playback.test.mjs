import assert from "node:assert/strict";
import test from "node:test";

await import("../../app/static/js/video_playback.js");
const playback = globalThis.MotoVideoPlayback;

test("uploaded video seeks only when analysis drift is visibly large", () => {
  assert.equal(playback.plan({ targetSeconds: 8, currentSeconds: 7.8, rate: 0.5, running: true }).seekTo, null);
  assert.equal(playback.plan({ targetSeconds: 8, currentSeconds: 6, rate: 0.5, running: true }).seekTo, 8);
});

test("uploaded video follows measured analysis speed and pause state", () => {
  const running = playback.plan({ targetSeconds: 4, currentSeconds: 4, rate: 0.42, running: true });
  assert.ok(running.playbackRate >= 0.4 && running.playbackRate <= 0.45);
  assert.equal(running.shouldPlay, true);
  assert.equal(playback.plan({ targetSeconds: 4, currentSeconds: 4, rate: 1, running: false }).shouldPlay, false);
});
