"use strict";

// Geometry and state transitions are DOM-independent so they can be tested
// without a browser and reused by both camera and uploaded-video previews.
(function installFinishLineHelpers(root) {
  const DEFAULT_LINE = Object.freeze({ x1: 0.10, y1: 0.68, x2: 0.90, y2: 0.68 });

  function cloneLine(line) {
    return {
      x1: Number(line.x1), y1: Number(line.y1),
      x2: Number(line.x2), y2: Number(line.y2),
    };
  }

  function clamp(value, minimum = 0, maximum = 1) {
    return Math.max(minimum, Math.min(maximum, Number(value)));
  }

  function normalizeLine(line) {
    return {
      x1: clamp(line.x1), y1: clamp(line.y1),
      x2: clamp(line.x2), y2: clamp(line.y2),
    };
  }

  function linesEqual(left, right, tolerance = 1e-9) {
    return ["x1", "y1", "x2", "y2"]
      .every((key) => Math.abs(Number(left[key]) - Number(right[key])) <= tolerance);
  }

  function lineLength(line) {
    return Math.hypot(line.x2 - line.x1, line.y2 - line.y1);
  }

  function lineAngle(line) {
    return Math.atan2(line.y2 - line.y1, line.x2 - line.x1);
  }

  function pointDistancePixels(left, right, width, height) {
    return Math.hypot((left.x - right.x) * width, (left.y - right.y) * height);
  }

  function segmentDistancePixels(point, line, width, height) {
    const ax = line.x1 * width; const ay = line.y1 * height;
    const bx = line.x2 * width; const by = line.y2 * height;
    const px = point.x * width; const py = point.y * height;
    const dx = bx - ax; const dy = by - ay;
    const denominator = dx * dx + dy * dy;
    const projection = denominator > 0
      ? clamp(((px - ax) * dx + (py - ay) * dy) / denominator)
      : 0;
    return Math.hypot(px - (ax + projection * dx), py - (ay + projection * dy));
  }

  function hitTest(line, point, width, height, endpointTolerance = 28, segmentTolerance = 18) {
    const startDistance = pointDistancePixels(point, { x: line.x1, y: line.y1 }, width, height);
    const endDistance = pointDistancePixels(point, { x: line.x2, y: line.y2 }, width, height);
    if (startDistance <= endpointTolerance || endDistance <= endpointTolerance) {
      return startDistance <= endDistance ? "start" : "end";
    }
    return segmentDistancePixels(point, line, width, height) <= segmentTolerance
      ? "segment"
      : null;
  }

  function moveEndpoint(line, endpoint, point) {
    const result = cloneLine(line);
    if (endpoint === "start") {
      result.x1 = clamp(point.x); result.y1 = clamp(point.y);
    } else if (endpoint === "end") {
      result.x2 = clamp(point.x); result.y2 = clamp(point.y);
    }
    return result;
  }

  function translateLine(line, dx, dy) {
    // Clamp one shared delta. Clamping endpoints separately would change the
    // segment length and angle near a preview boundary.
    const minimumX = Math.min(line.x1, line.x2);
    const maximumX = Math.max(line.x1, line.x2);
    const minimumY = Math.min(line.y1, line.y2);
    const maximumY = Math.max(line.y1, line.y2);
    const clampedX = clamp(dx, -minimumX, 1 - maximumX);
    const clampedY = clamp(dy, -minimumY, 1 - maximumY);
    return {
      x1: line.x1 + clampedX, y1: line.y1 + clampedY,
      x2: line.x2 + clampedX, y2: line.y2 + clampedY,
    };
  }

  function containedImageRect(containerWidth, containerHeight, imageWidth, imageHeight) {
    const safeContainerWidth = Math.max(1, Number(containerWidth));
    const safeContainerHeight = Math.max(1, Number(containerHeight));
    const safeImageWidth = Math.max(1, Number(imageWidth));
    const safeImageHeight = Math.max(1, Number(imageHeight));
    const scale = Math.min(
      safeContainerWidth / safeImageWidth,
      safeContainerHeight / safeImageHeight,
    );
    const width = safeImageWidth * scale;
    const height = safeImageHeight * scale;
    return {
      left: (safeContainerWidth - width) / 2,
      top: (safeContainerHeight - height) / 2,
      width,
      height,
    };
  }

  function createState(initial = DEFAULT_LINE) {
    const line = normalizeLine(initial);
    return {
      persisted: cloneLine(line),
      editable: cloneLine(line),
      initialized: false,
      dragging: false,
      dirty: false,
      saveInProgress: false,
      editRevision: 0,
      saveSequence: 0,
    };
  }

  function initialize(state, line) {
    const normalized = normalizeLine(line);
    state.persisted = cloneLine(normalized);
    state.editable = cloneLine(normalized);
    state.initialized = true;
    state.dragging = false;
    state.dirty = false;
    state.saveInProgress = false;
  }

  function setEditable(state, line) {
    state.editable = normalizeLine(line);
    state.editRevision += 1;
    state.dirty = !linesEqual(state.editable, state.persisted);
  }

  function applyRemote(state, line) {
    // Polling, WebSocket messages, delayed camera responses, and preview
    // metadata must not win while the operator owns a local edit.
    if (state.dragging || state.dirty || state.saveInProgress) return false;
    initialize(state, line);
    return true;
  }

  function beginSave(state) {
    state.saveSequence += 1;
    state.saveInProgress = true;
    return {
      sequence: state.saveSequence,
      editRevision: state.editRevision,
      line: cloneLine(state.editable),
    };
  }

  function completeSave(state, token, persistedLine) {
    if (token.sequence !== state.saveSequence) return false;
    const persisted = normalizeLine(persistedLine);
    state.persisted = cloneLine(persisted);
    state.saveInProgress = false;
    if (!state.dragging && token.editRevision === state.editRevision) {
      state.editable = cloneLine(persisted);
    }
    state.dirty = !linesEqual(state.editable, state.persisted);
    return true;
  }

  function failSave(state, token) {
    if (token.sequence !== state.saveSequence) return false;
    state.saveInProgress = false;
    state.dirty = true;
    return true;
  }

  root.MotoFinishLine = Object.freeze({
    DEFAULT_LINE, cloneLine, normalizeLine, linesEqual, lineLength, lineAngle,
    hitTest, moveEndpoint, translateLine, createState, initialize, setEditable,
    applyRemote, beginSave, completeSave, failSave, containedImageRect,
  });
}(globalThis));
