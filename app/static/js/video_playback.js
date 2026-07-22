"use strict";

// Keep native browser video playback close to the authoritative analysis
// position. Analysis may run below or above 1x on CPU, but the browser renders
// continuous frames instead of exposing the inference worker's stop-frame pace.
(function installVideoPlayback(root) {
  function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, Number(value)));
  }

  function plan({ targetSeconds, currentSeconds, rate, running }) {
    const target = Math.max(0, Number(targetSeconds) || 0);
    const current = Math.max(0, Number(currentSeconds) || 0);
    const drift = target - current;
    const baseRate = clamp(Number(rate) || 1, 0.10, 4.0);
    return {
      seekTo: Math.abs(drift) > 0.65 ? target : null,
      playbackRate: clamp(baseRate * (1 + clamp(drift * 0.12, -0.18, 0.18)), 0.10, 4.0),
      shouldPlay: Boolean(running),
    };
  }

  root.MotoVideoPlayback = Object.freeze({ plan });
}(globalThis));
