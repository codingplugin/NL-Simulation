// Aquila Optimizer (AO) logic for node localization

const FC = 2;

export function euclideanDistance(p1, p2) {
  return Math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2);
}

export function estimateDistance(actualDist, noiseFactor) {
  const noise = (Math.random() * 2 - 1) * actualDist * noiseFactor;
  return actualDist + noise;
}

export function getCentroid(nodes) {
  const n = nodes.length;
  const sum = nodes.reduce((acc, p) => [acc[0] + p[0], acc[1] + p[1]], [0, 0]);
  return [sum[0] / n, sum[1] / n];
}

export function fitnessFunction(position, anchors, estimatedDistances) {
  let error = 0;
  for (let i = 0; i < anchors.length; i++) {
    const dist = euclideanDistance(position, anchors[i]);
    error += (dist - estimatedDistances[i]) ** 2;
  }
  return error / anchors.length;
}

function gamma(x) {
  // Lanczos approximation for Gamma function
  const p = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7];
  let g = 7;
  if (x < 0.5) return Math.PI / (Math.sin(Math.PI * x) * gamma(1 - x));
  x -= 1;
  let a = 0.99999999999980993;
  for (let i = 0; i < p.length; i++) a += p[i] / (x + i + 1);
  let t = x + g + 0.5;
  return Math.sqrt(2 * Math.PI) * Math.pow(t, x + 0.5) * Math.exp(-t) * a;
}

function levyFlight(D, beta = 1.5) {
  // Approximate Levy flight for JS
  const sigma = Math.pow(
    gamma(1 + beta) * Math.sin(Math.PI * beta / 2) /
      (gamma((1 + beta) / 2) * beta * Math.pow(2, (beta - 1) / 2)),
    1 / beta
  );
  const u = Array.from({ length: D }, () => randn_bm() * sigma);
  const v = Array.from({ length: D }, () => randn_bm());
  return u.map((ui, i) => ui / Math.pow(Math.abs(v[i]), 1 / beta));
}

function randn_bm() {
  // Box-Muller transform
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export function aquilaStep(population, best, anchors, estimatedDistances, t, maxIter, areaSize) {
  // One iteration of Aquila Optimizer for all individuals
  const popSize = population.length;
  const Dim = 2;
  const T = maxIter;
  const XM = getCentroid(population);
  const newPop = [];
  for (let i = 0; i < popSize; i++) {
    let newPos;
    if (t <= (2 / 3) * T) {
      if (Math.random() < 0.5) {
        // Expanded exploration
        const X1 = [
          best[0] * (1 - t / T) + (XM[0] - best[0] * Math.random()),
          best[1] * (1 - t / T) + (XM[1] - best[1] * Math.random()),
        ];
        newPos = [
          Math.min(Math.max(X1[0], 0), areaSize),
          Math.min(Math.max(X1[1], 0), areaSize),
        ];
      } else {
        // Narrowed exploration
        const XR = population[Math.floor(Math.random() * popSize)];
        const y = Math.random() * 2 - 1;
        const x = Math.random() * 2 - 1;
        const levy = levyFlight(Dim);
        const X2 = [
          best[0] * levy[0] + XR[0] + (y - x) * Math.random(),
          best[1] * levy[1] + XR[1] + (y - x) * Math.random(),
        ];
        newPos = [
          Math.min(Math.max(X2[0], 0), areaSize),
          Math.min(Math.max(X2[1], 0), areaSize),
        ];
      }
    } else {
      if (Math.random() < 0.5) {
        // Expanded exploitation
        const alpha = 0.1, delta = 0.1;
        const UB = areaSize, LB = 0;
        const X3 = [
          (best[0] - XM[0]) * alpha - Math.random() * (UB - LB) * delta + LB,
          (best[1] - XM[1]) * alpha - Math.random() * (UB - LB) * delta + LB,
        ];
        newPos = [
          Math.min(Math.max(X3[0], 0), areaSize),
          Math.min(Math.max(X3[1], 0), areaSize),
        ];
      } else {
        // Narrowed exploitation
        const G1 = 2 * Math.random() - 1;
        const G2 = 2 * (1 - t / T);
        const QF = Math.pow(t, (2 * Math.random() - 1) / Math.pow(1 - T, 2));
        const levy = levyFlight(Dim);
        const X4 = [
          QF * best[0] - (G2 * population[i][0] * Math.random()) - G2 * levy[0] + Math.random() * G1,
          QF * best[1] - (G2 * population[i][1] * Math.random()) - G2 * levy[1] + Math.random() * G1,
        ];
        newPos = [
          Math.min(Math.max(X4[0], 0), areaSize),
          Math.min(Math.max(X4[1], 0), areaSize),
        ];
      }
    }
    newPop.push(newPos);
  }
  // Find new best
  let bestIdx = 0;
  let bestFit = fitnessFunction(newPop[0], anchors, estimatedDistances);
  for (let i = 1; i < newPop.length; i++) {
    const fit = fitnessFunction(newPop[i], anchors, estimatedDistances);
    if (fit < bestFit) {
      bestFit = fit;
      bestIdx = i;
    }
  }
  return { newPop, newBest: newPop[bestIdx], bestFit };
} 