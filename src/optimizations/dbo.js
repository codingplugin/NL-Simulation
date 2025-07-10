// Dung Beetle Optimizer (DBO) logic for node localization

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

export function dboStep(population, best, anchors, estimatedDistances, t, maxIter, areaSize) {
  // DBO parameters
  const k = 0.1; // Deflection coefficient
  const b = 0.3; // Light intensity factor
  const S = 0.5; // Stealing factor
  const Tmax = maxIter;
  const R = 1 - t / Tmax; // Dynamic parameter
  
  const popSize = population.length;
  
  // Population distribution
  const n_ball_rolling = Math.floor(popSize * 0.2); // 20% ball-rolling
  const n_brood = Math.floor(popSize * 0.2); // 20% brood balls
  const n_small = Math.floor(popSize * 0.24); // 24% small dung beetles
  const n_thief = popSize - n_ball_rolling - n_brood - n_small; // Remaining thieves
  
  // Find worst position
  let worstIdx = 0;
  let worstFitness = fitnessFunction(population[0], anchors, estimatedDistances);
  for (let i = 1; i < popSize; i++) {
    const fit = fitnessFunction(population[i], anchors, estimatedDistances);
    if (fit > worstFitness) {
      worstFitness = fit;
      worstIdx = i;
    }
  }
  const Xw = population[worstIdx];
  const Xstar = best; // Local best (assumed same as global best)
  
  const newPop = [];
  for (let i = 0; i < popSize; i++) {
    let newPos;
    const currentPos = population[i];
    
    if (i < n_ball_rolling) {
      // Ball-rolling dung beetles
      if (Math.random() < 0.1) {
        // Dancing behavior (obstacle avoidance)
        const theta = Math.random() * Math.PI;
        if (theta !== 0 && theta !== Math.PI / 2 && theta !== Math.PI) {
          const delta = Math.abs(currentPos[0] - population[i][0]) + Math.abs(currentPos[1] - population[i][1]);
          newPos = [
            currentPos[0] + Math.tan(theta) * delta,
            currentPos[1] + Math.tan(theta) * delta
          ];
        } else {
          newPos = currentPos;
        }
      } else {
        // Ball-rolling behavior
        const alpha = Math.random() < 0.5 ? 1 : -1;
        const deltaX = Math.abs(currentPos[0] - Xw[0]) + Math.abs(currentPos[1] - Xw[1]);
        newPos = [
          currentPos[0] + alpha * k * population[i][0] + b * deltaX,
          currentPos[1] + alpha * k * population[i][1] + b * deltaX
        ];
      }
    } else if (i < n_ball_rolling + n_brood) {
      // Brood balls
      const Lb_star = [
        Math.max(Xstar[0] * (1 - R), 0),
        Math.max(Xstar[1] * (1 - R), 0)
      ];
      const Ub_star = [
        Math.min(Xstar[0] * (1 + R), areaSize),
        Math.min(Xstar[1] * (1 + R), areaSize)
      ];
      const b1 = Math.random();
      const b2 = Math.random();
      newPos = [
        Xstar[0] + b1 * (currentPos[0] - Lb_star[0]) + b2 * (currentPos[0] - Ub_star[0]),
        Xstar[1] + b1 * (currentPos[1] - Lb_star[1]) + b2 * (currentPos[1] - Ub_star[1])
      ];
    } else if (i < n_ball_rolling + n_brood + n_small) {
      // Small dung beetles
      const Lbb = [
        Math.max(best[0] * (1 - R), 0),
        Math.max(best[1] * (1 - R), 0)
      ];
      const Ubb = [
        Math.min(best[0] * (1 + R), areaSize),
        Math.min(best[1] * (1 + R), areaSize)
      ];
      const C1 = (Math.random() - 0.5) * 2; // Normal distribution approximation
      const C2 = Math.random();
      newPos = [
        currentPos[0] + C1 * (currentPos[0] - Lbb[0]) + C2 * (currentPos[0] - Ubb[0]),
        currentPos[1] + C1 * (currentPos[1] - Lbb[1]) + C2 * (currentPos[1] - Ubb[1])
      ];
    } else {
      // Thieves
      const g = (Math.random() - 0.5) * 2; // Normal distribution approximation
      const absDiff = Math.abs(currentPos[0] - Xstar[0]) + Math.abs(currentPos[1] - Xstar[1]) +
                     Math.abs(currentPos[0] - best[0]) + Math.abs(currentPos[1] - best[1]);
      newPos = [
        best[0] + S * g * absDiff,
        best[1] + S * g * absDiff
      ];
    }
    
    // Boundary check
    newPos = [
      Math.min(Math.max(newPos[0], 0), areaSize),
      Math.min(Math.max(newPos[1], 0), areaSize)
    ];
    
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