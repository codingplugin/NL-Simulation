// SOA optimization logic for node localization

const FC = 2;

export function euclideanDistance(p1, p2) {
  return Math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2);
}

// Add Gaussian random function
function gaussianRandom(mean = 0, stddev = 1) {
  let u = 0, v = 0;
  while(u === 0) u = Math.random();
  while(v === 0) v = Math.random();
  return mean + stddev * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export function estimateDistance(actualDist, noiseFactor) {
  const noise = gaussianRandom(0, actualDist * noiseFactor);
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

export function soaStep(population, pBest, anchors, estimatedDistances, t, maxIter) {
  const A = FC - t * (FC / maxIter);
  const newPop = [];
  for (let i = 0; i < population.length; i++) {
    const rd = Math.random();
    const theta = Math.random() * 2 * Math.PI;
    const u = 1, v = 1;
    const r = u * Math.exp(theta * v);
    const C = [A * population[i][0], A * population[i][1]];
    const B = 2 * A * A - rd;
    // Corrected M calculation per paper:
    const M = [B - pBest[0] - population[i][0], B - pBest[1] - population[i][1]];
    const D = [M[0] + C[0], M[1] + C[1]];
    const x = r * Math.cos(theta);
    const y = r * Math.sin(theta);
    const z = r * theta;
    // Remove Math.abs from D in final update
    const newPos = [
      Math.min(Math.max(D[0] * x * y * z + pBest[0], 0), 200),
      Math.min(Math.max(D[1] * x * y * z + pBest[1], 0), 200),
    ];
    newPop.push(newPos);
  }
  // Find new pBest
  let bestIdx = 0;
  let bestFit = fitnessFunction(newPop[0], anchors, estimatedDistances);
  for (let i = 1; i < newPop.length; i++) {
    const fit = fitnessFunction(newPop[i], anchors, estimatedDistances);
    if (fit < bestFit) {
      bestFit = fit;
      bestIdx = i;
    }
  }
  return { newPop, newPBest: newPop[bestIdx], bestFit };
} 