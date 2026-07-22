"use strict";

// Smoothly interpolate an authoritative race clock between backend updates.
// Uploaded video may be processed faster or slower than real time, therefore
// its rate is measured from source-timestamp samples instead of being fixed at 1x.
(function installRaceClock(root) {
  function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, Number(value)));
  }

  function createState() {
    return {
      raceId: null,
      baseNs: 0,
      syncMs: 0,
      rate: 0,
      running: false,
      isVideo: false,
      sampleNs: null,
      sampleMs: null,
    };
  }

  function value(clock, nowMs) {
    if (!clock.running) return clock.baseNs;
    return clock.baseNs + Math.max(0, Number(nowMs) - clock.syncMs) * 1e6 * clock.rate;
  }

  function sync(clock, sample) {
    const nowMs = Number(sample.nowMs);
    const elapsedNs = Math.max(0, Number(sample.elapsedNs || 0));
    const sameRace = clock.raceId === sample.raceId;
    const isVideo = Boolean(sample.isVideo);
    const running = Boolean(sample.running);
    let rate = 0;

    if (running && !isVideo) {
      rate = 1;
    } else if (running && isVideo) {
      rate = sameRace && clock.rate > 0 ? clock.rate : 1;
      if (sameRace && clock.sampleNs !== null && clock.sampleMs !== null) {
        const deltaMs = nowMs - clock.sampleMs;
        const deltaNs = elapsedNs - clock.sampleNs;
        if (deltaMs >= 40 && deltaNs > 0) {
          // Offline inference can be slower or faster than source playback.
          // Limiting extreme estimates prevents one delayed HTTP response from
          // making the visible clock jump uncontrollably between samples.
          const measured = clamp(deltaNs / (deltaMs * 1e6), 0.02, 8);
          rate = clock.rate > 0 ? clock.rate * 0.35 + measured * 0.65 : measured;
        }
      }
    }

    Object.assign(clock, {
      raceId: sample.raceId,
      baseNs: elapsedNs,
      syncMs: nowMs,
      rate,
      running,
      isVideo,
      sampleNs: elapsedNs,
      sampleMs: nowMs,
    });
    return clock;
  }

  root.MotoRaceClock = Object.freeze({ createState, sync, value });
}(globalThis));
