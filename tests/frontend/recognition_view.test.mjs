import assert from "node:assert/strict";
import test from "node:test";

await import("../../app/static/js/recognition_view.js");
const view = globalThis.MotoRecognitionView;

test("panel updates only from stable runtime state and preserves leading zeros", () => {
  assert.deepEqual(view.build({}), {
    number: "—", state: "Ожидание мотоцикла", history: [],
  });
  const history = [{
    racing_number: "007", timestamp: "2026-01-01T00:00:00Z",
    state: "recognized", track_id: 3, crossed: false, lap_number: null,
  }];
  const result = view.build({ recognized_number: "7", recognition_history: history });
  assert.equal(result.number, "007");
  assert.equal(result.state, "Распознан · трек 3");
});

test("lap event is represented as a finish-line crossing", () => {
  const result = view.build({ recognition_history: [{
    racing_number: "306", track_id: 1, crossed: true, lap_number: 2,
  }] });
  assert.equal(result.number, "306");
  assert.equal(result.state, "Пересечение · круг 2");
});
