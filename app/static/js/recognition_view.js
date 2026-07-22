"use strict";

// Pure view-model construction keeps the prominent recognition panel testable
// without tying CV state to DOM implementation details.
(function installRecognitionView(root) {
  function build(runtime = {}) {
    const history = Array.isArray(runtime.recognition_history)
      ? runtime.recognition_history.slice(0, 8)
      : [];
    const latest = history[0] || null;
    return {
      number: latest?.racing_number ?? runtime.recognized_number ?? "—",
      state: latest
        ? (latest.crossed
          ? `Пересечение · круг ${latest.lap_number}`
          : `Распознан · трек ${latest.track_id}`)
        : "Ожидание мотоцикла",
      history,
    };
  }

  root.MotoRecognitionView = Object.freeze({ build });
}(globalThis));
