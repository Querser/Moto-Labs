"use strict";

// This helper is DOM-independent so stale-response behavior can be tested with Node.js.
globalThis.MotoLapState = Object.freeze({
  mergeLapRows(preferred, fetched) {
    const rows = new Map();
    for (const lap of [...preferred, ...fetched]) {
      if (!rows.has(lap.id)) rows.set(lap.id, lap);
    }
    return [...rows.values()].sort((left, right) => right.id - left.id);
  },
});
