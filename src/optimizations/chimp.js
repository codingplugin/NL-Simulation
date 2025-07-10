// Chimp Optimization Algorithm (ChOA) logic for node localization

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

function gaussMouseMap(x, iterations) {
  const chaoticValues = [];
  let currentX = x;
  for (let i = 0; i < iterations; i++) {
    if (currentX === 0) {
      currentX = 1;
    } else {
      currentX = 1 / (currentX % 1);
    }
    // Bound the chaotic value to prevent extreme values
    chaoticValues.push(Math.min(Math.max(currentX, 0.1), 10));
  }
  return chaoticValues;
}

export function chimpStep(population, best, anchors, estimatedDistances, t, maxIter, areaSize) {
  const popSize = population.length;
  const minVal = 0, maxVal = areaSize;
  
  // Validate input population and ensure all positions are within bounds
  for (let i = 0; i < population.length; i++) {
    if (!population[i] || population[i].length !== 2 || 
        isNaN(population[i][0]) || isNaN(population[i][1]) ||
        !isFinite(population[i][0]) || !isFinite(population[i][1])) {
      // Reset invalid positions to random values within bounds
      population[i] = [Math.random() * maxVal, Math.random() * maxVal];
    } else {
      // Ensure existing positions are within bounds
      population[i][0] = Math.min(Math.max(population[i][0], minVal), maxVal);
      population[i][1] = Math.min(Math.max(population[i][1], minVal), maxVal);
    }
  }
  
  // Calculate fitness for current population
  const fitness = population.map(p => fitnessFunction(p, anchors, estimatedDistances));
  
  // Sort population by fitness (ascending)
  const sortedIndices = fitness.map((_, i) => i).sort((a, b) => fitness[a] - fitness[b]);
  const sortedPopulation = sortedIndices.map(i => population[i]);
  const sortedFitness = sortedIndices.map(i => fitness[i]);
  
  // Initialize best solutions for attacker, barrier, chaser, driver
  const attacker = sortedPopulation[0].slice();
  const barrier = popSize > 1 ? sortedPopulation[1].slice() : attacker.slice();
  const chaser = popSize > 2 ? sortedPopulation[2].slice() : attacker.slice();
  const driver = popSize > 3 ? sortedPopulation[3].slice() : attacker.slice();
  let bestFitness = sortedFitness[0];
  
  // Initialize chaotic map (Gauss/mouse map for ChOA12)
  const chaoticValue = Math.random();
  const chaoticValues = gaussMouseMap(chaoticValue, maxIter);
  
  // Dynamic coefficient f for four groups (ChOA1, Table 1) - Original equations from paper
  const fValues = [
    1.95 - 2 * (t ** (1/4)) / (maxIter ** (1/3)),  // Group 1
    1.95 - 2 * (t ** (1/3)) / (maxIter ** (1/4)),  // Group 2
    (-3 * (t ** 3) / (maxIter ** 3)) + 1.5,      // Group 3
    (-2 * (t ** 3) / (maxIter ** 3)) + 1.5       // Group 4
  ];
  
  const groupSize = Math.floor(popSize / 4);
  const newPopulation = [];
  const newFitness = [];
  
  // Driving and chasing phase
  for (let i = 0; i < popSize; i++) {
    // Assign chimp to one of the four groups
    const groupIdx = Math.min(Math.floor(i / groupSize), 3);
    const f = fValues[groupIdx];
    
    // Calculate coefficients (Equations 3, 4, 5) - Original equations from paper
    const r1 = [Math.random(), Math.random()];
    const r2 = [Math.random(), Math.random()];
    const a = [2 * f * r1[0] - f, 2 * f * r1[1] - f];  // Equation (3)
    const c = [2 * r2[0], 2 * r2[1]];                   // Equation (4)
    const m = chaoticValues[t];                           // Equation (5)
    
    // Select one of the best solutions (attacker, barrier, chaser, driver)
    const bestPositions = [attacker, barrier, chaser, driver];
    const bestIdx = Math.floor(Math.random() * 4);
    const xPrey = bestPositions[bestIdx];
    
    // Driving and chasing (Equations 1 and 2) - Original equations from paper
    const d = [
      Math.abs(c[0] * xPrey[0] - m * population[i][0]),  // Equation (1)
      Math.abs(c[1] * xPrey[1] - m * population[i][1])
    ];
    
    const newPos = [
      xPrey[0] - a[0] * d[0],  // Equation (2)
      xPrey[1] - a[1] * d[1]
    ];
    
    // Boundary check
    newPos[0] = Math.min(Math.max(newPos[0], minVal), maxVal);
    newPos[1] = Math.min(Math.max(newPos[1], minVal), maxVal);
    
    // Evaluate new position
    const newFitnessVal = fitnessFunction(newPos, anchors, estimatedDistances);
    newPopulation.push(newPos);
    newFitness.push(newFitnessVal);
    
    // Update best solution if better
    if (newFitnessVal < bestFitness) {
      attacker[0] = newPos[0];
      attacker[1] = newPos[1];
      bestFitness = newFitnessVal;
    }
  }
  
  // Attacking phase (Equations 6, 7, 8)
  for (let i = 0; i < popSize; i++) {
    const groupIdx = Math.min(Math.floor(i / groupSize), 3);
    const f = fValues[groupIdx];
    
    const r1 = [Math.random(), Math.random()];
    const r2 = [Math.random(), Math.random()];
    const a1 = [2 * f * Math.random() - f, 2 * f * Math.random() - f];
    const a2 = [2 * f * Math.random() - f, 2 * f * Math.random() - f];
    const a3 = [2 * f * Math.random() - f, 2 * f * Math.random() - f];
    const a4 = [2 * f * Math.random() - f, 2 * f * Math.random() - f];
    
    const c1 = [2 * Math.random(), 2 * Math.random()];
    const c2 = [2 * Math.random(), 2 * Math.random()];
    const c3 = [2 * Math.random(), 2 * Math.random()];
    const c4 = [2 * Math.random(), 2 * Math.random()];
    
    const m1 = chaoticValues[t];
    const m2 = chaoticValues[t];
    const m3 = chaoticValues[t];
    const m4 = chaoticValues[t];
    
    const dAttacker = [
      Math.abs(c1[0] * attacker[0] - m1 * newPopulation[i][0]),  // Equation (6)
      Math.abs(c1[1] * attacker[1] - m1 * newPopulation[i][1])
    ];
    const dBarrier = [
      Math.abs(c2[0] * barrier[0] - m2 * newPopulation[i][0]),
      Math.abs(c2[1] * barrier[1] - m2 * newPopulation[i][1])
    ];
    const dChaser = [
      Math.abs(c3[0] * chaser[0] - m3 * newPopulation[i][0]),
      Math.abs(c3[1] * chaser[1] - m3 * newPopulation[i][1])
    ];
    const dDriver = [
      Math.abs(c4[0] * driver[0] - m4 * newPopulation[i][0]),
      Math.abs(c4[1] * driver[1] - m4 * newPopulation[i][1])
    ];
    

    
    const x1 = [attacker[0] - a1[0] * dAttacker[0], attacker[1] - a1[1] * dAttacker[1]];  // Equation (7)
    const x2 = [barrier[0] - a2[0] * dBarrier[0], barrier[1] - a2[1] * dBarrier[1]];
    const x3 = [chaser[0] - a3[0] * dChaser[0], chaser[1] - a3[1] * dChaser[1]];
    const x4 = [driver[0] - a4[0] * dDriver[0], driver[1] - a4[1] * dDriver[1]];
    
    const newPos = [
      (x1[0] + x2[0] + x3[0] + x4[0]) / 4,  // Equation (8)
      (x1[1] + x2[1] + x3[1] + x4[1]) / 4
    ];
    
    // Boundary check
    newPos[0] = Math.min(Math.max(newPos[0], minVal), maxVal);
    newPos[1] = Math.min(Math.max(newPos[1], minVal), maxVal);
    
    const newFitnessVal = fitnessFunction(newPos, anchors, estimatedDistances);
    
    if (newFitnessVal < newFitness[i]) {
      newPopulation[i] = newPos;
      newFitness[i] = newFitnessVal;
      
      if (newFitnessVal < bestFitness) {
        attacker[0] = newPos[0];
        attacker[1] = newPos[1];
        bestFitness = newFitnessVal;
      }
    }
  }
  
  // Update best solutions for barrier, chaser, driver
  const finalSortedIndices = newFitness.map((_, i) => i).sort((a, b) => newFitness[a] - newFitness[b]);
  const finalPopulation = finalSortedIndices.map(i => newPopulation[i]);
  const finalFitness = finalSortedIndices.map(i => newFitness[i]);
  
  // Find best solution
  let bestIdx = 0;
  let bestFit = finalFitness[0];
  for (let i = 1; i < finalPopulation.length; i++) {
    if (finalFitness[i] < bestFit) {
      bestFit = finalFitness[i];
      bestIdx = i;
    }
  }
  
  return { newPop: finalPopulation, newBest: finalPopulation[bestIdx], bestFit };
} 