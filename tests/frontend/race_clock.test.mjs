import assert from "node:assert/strict";
import test from "node:test";

await import("../../app/static/js/race_clock.js");
const raceClock = globalThis.MotoRaceClock;

test("live race clock advances smoothly between server samples", () => {
  const clock = raceClock.createState();
  raceClock.sync(clock, {
    raceId: 1, elapsedNs: 2_000_000_000, nowMs: 1000,
    running: true, isVideo: false,
  });
  assert.equal(raceClock.value(clock, 1050), 2_050_000_000);
  assert.equal(raceClock.value(clock, 1123), 2_123_000_000);
});

test("uploaded-video clock estimates source processing rate and interpolates", () => {
  const clock = raceClock.createState();
  raceClock.sync(clock, {
    raceId: 2, elapsedNs: 1_000_000_000, nowMs: 1000,
    running: true, isVideo: true,
  });
  raceClock.sync(clock, {
    raceId: 2, elapsedNs: 1_250_000_000, nowMs: 1500,
    running: true, isVideo: true,
  });
  assert.ok(clock.rate > 0.4 && clock.rate < 0.8);
  const first = raceClock.value(clock, 1550);
  const second = raceClock.value(clock, 1600);
  assert.ok(first > 1_250_000_000);
  assert.ok(second > first);
});

test("paused and completed clocks remain fixed", () => {
  const clock = raceClock.createState();
  raceClock.sync(clock, {
    raceId: 3, elapsedNs: 9_876_000_000, nowMs: 2000,
    running: false, isVideo: true,
  });
  assert.equal(raceClock.value(clock, 9000), 9_876_000_000);
});

test("new race does not inherit the previous processing rate", () => {
  const clock = raceClock.createState();
  raceClock.sync(clock, {
    raceId: 4, elapsedNs: 8_000_000_000, nowMs: 1000,
    running: true, isVideo: true,
  });
  raceClock.sync(clock, {
    raceId: 5, elapsedNs: 0, nowMs: 2000,
    running: false, isVideo: true,
  });
  assert.equal(clock.rate, 0);
  assert.equal(raceClock.value(clock, 5000), 0);
});
