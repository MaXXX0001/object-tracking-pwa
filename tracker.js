// ByteTrack in plain JS. Boxes are [x1, y1, x2, y2]; detections carry a score as the fifth item.

const POSITION_WEIGHT = 1 / 20;
const VELOCITY_WEIGHT = 1 / 160;
const IMPOSSIBLE_COST = 1000;

// Constant-velocity Kalman filter for one box coordinate; noise scales with box height.
class Kalman1D {
  constructor(position, height) {
    this.position = position;
    this.velocity = 0;
    this.pp = (2 * POSITION_WEIGHT * height) ** 2;
    this.pv = 0;
    this.vv = (10 * VELOCITY_WEIGHT * height) ** 2;
  }

  predict(height) {
    this.position += this.velocity;
    this.pp += 2 * this.pv + this.vv + (POSITION_WEIGHT * height) ** 2;
    this.pv += this.vv;
    this.vv += (VELOCITY_WEIGHT * height) ** 2;
  }

  update(measurement, height) {
    const innovation = measurement - this.position;
    const innovationVariance = this.pp + (POSITION_WEIGHT * height) ** 2;
    const gainPosition = this.pp / innovationVariance;
    const gainVelocity = this.pv / innovationVariance;
    this.position += gainPosition * innovation;
    this.velocity += gainVelocity * innovation;
    const pp = this.pp;
    const pv = this.pv;
    this.pp = pp - gainPosition * pp;
    this.vv -= gainVelocity * pv;
    this.pv = pv - gainPosition * pv;
  }
}

function iou(a, b) {
  const left = Math.max(a[0], b[0]);
  const top = Math.max(a[1], b[1]);
  const right = Math.min(a[2], b[2]);
  const bottom = Math.min(a[3], b[3]);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection;
  return union > 0 ? intersection / union : 0;
}

// Hungarian algorithm for an n x m cost matrix with n <= m; returns the column of each row.
function hungarian(cost, n, m) {
  const INF = 1e18;
  const u = new Float64Array(n + 1);
  const v = new Float64Array(m + 1);
  const p = new Int32Array(m + 1);
  const way = new Int32Array(m + 1);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(m + 1).fill(INF);
    const used = new Uint8Array(m + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = INF;
      let j1 = 0;
      for (let j = 1; j <= m; j++) {
        if (used[j]) continue;
        const current = cost[(i0 - 1) * m + (j - 1)] - u[i0] - v[j];
        if (current < minv[j]) {
          minv[j] = current;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= m; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }
  const rowToColumn = new Int32Array(n).fill(-1);
  for (let j = 1; j <= m; j++) {
    if (p[j]) rowToColumn[p[j] - 1] = j - 1;
  }
  return rowToColumn;
}

// Matches rows to columns by IoU; returns matched pairs plus the leftover row and column indices.
function associate(rowBoxes, columnBoxes, minIou) {
  const rows = rowBoxes.length;
  const columns = columnBoxes.length;
  if (rows === 0 || columns === 0) {
    return {
      matches: [],
      unmatchedRows: rowBoxes.map((_, index) => index),
      unmatchedColumns: columnBoxes.map((_, index) => index),
    };
  }

  const cost = new Float64Array(rows * columns);
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const overlap = iou(rowBoxes[row], columnBoxes[column]);
      cost[row * columns + column] = overlap >= minIou ? 1 - overlap : IMPOSSIBLE_COST;
    }
  }

  const pairs = [];
  if (rows <= columns) {
    const rowToColumn = hungarian(cost, rows, columns);
    rowToColumn.forEach((column, row) => pairs.push([row, column]));
  } else {
    const transposed = new Float64Array(rows * columns);
    for (let row = 0; row < rows; row++) {
      for (let column = 0; column < columns; column++) {
        transposed[column * rows + row] = cost[row * columns + column];
      }
    }
    const columnToRow = hungarian(transposed, columns, rows);
    columnToRow.forEach((row, column) => pairs.push([row, column]));
  }

  const matches = [];
  const matchedRows = new Set();
  const matchedColumns = new Set();
  for (const [row, column] of pairs) {
    if (row < 0 || column < 0 || cost[row * columns + column] >= IMPOSSIBLE_COST) continue;
    matches.push([row, column]);
    matchedRows.add(row);
    matchedColumns.add(column);
  }
  return {
    matches,
    unmatchedRows: rowBoxes.map((_, index) => index).filter((index) => !matchedRows.has(index)),
    unmatchedColumns: columnBoxes.map((_, index) => index).filter((index) => !matchedColumns.has(index)),
  };
}

class Track {
  constructor(detection) {
    const height = Math.max(detection[3] - detection[1], 1);
    this.filters = [0, 1, 2, 3].map((index) => new Kalman1D(detection[index], height));
    this.score = detection[4];
    this.hits = 1;
    this.timeSinceUpdate = 0;
    this.id = null;
  }

  get box() {
    return this.filters.map((filter) => filter.position);
  }

  get height() {
    const [, y1, , y2] = this.box;
    return Math.max(y2 - y1, 1);
  }

  predict() {
    const height = this.height;
    this.filters.forEach((filter) => filter.predict(height));
    this.timeSinceUpdate += 1;
  }

  update(detection) {
    const height = Math.max(detection[3] - detection[1], 1);
    this.filters.forEach((filter, index) => filter.update(detection[index], height));
    this.score = detection[4];
    this.hits += 1;
    this.timeSinceUpdate = 0;
  }
}

export class ByteTracker {
  constructor({ highThreshold = 0.6, minIou = 0.1, lostBuffer = 30, minHits = 2 } = {}) {
    this.highThreshold = highThreshold;
    this.minIou = minIou;
    this.lostBuffer = lostBuffer;
    this.minHits = minHits;
    this.tracks = [];
    this.nextId = 1;
  }

  update(detections) {
    this.tracks.forEach((track) => track.predict());
    const high = detections.filter((detection) => detection[4] >= this.highThreshold);
    const low = detections.filter((detection) => detection[4] < this.highThreshold);

    const first = associate(
      this.tracks.map((track) => track.box),
      high,
      this.minIou,
    );
    for (const [trackIndex, detectionIndex] of first.matches) {
      this.tracks[trackIndex].update(high[detectionIndex]);
    }

    // Only tracks seen in the previous frame get a second chance on weak detections
    const recent = first.unmatchedRows.filter((index) => this.tracks[index].timeSinceUpdate === 1);
    const second = associate(
      recent.map((index) => this.tracks[index].box),
      low,
      this.minIou,
    );
    for (const [recentIndex, detectionIndex] of second.matches) {
      this.tracks[recent[recentIndex]].update(low[detectionIndex]);
    }

    for (const detectionIndex of first.unmatchedColumns) {
      this.tracks.push(new Track(high[detectionIndex]));
    }

    this.tracks = this.tracks.filter((track) => {
      const unconfirmedMiss = track.hits < this.minHits && track.timeSinceUpdate > 0;
      return !unconfirmedMiss && track.timeSinceUpdate <= this.lostBuffer;
    });

    const output = [];
    for (const track of this.tracks) {
      if (track.timeSinceUpdate !== 0 || track.hits < this.minHits) continue;
      if (track.id === null) {
        track.id = this.nextId;
        this.nextId += 1;
      }
      output.push({ id: track.id, box: track.box, score: track.score });
    }
    return output;
  }
}
