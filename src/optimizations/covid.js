// Coronavirus Optimization Algorithm (COVIDOA) logic for node localization

const FC = 2;
const MR = 0.1; // Mutation Rate
const NUM_OF_PROTEINS = 2; // Number of proteins generated
const SHIFTING_NO = 1; // +1 frameshifting

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

export function covidStep(population, best, anchors, estimatedDistances, t, maxIter, areaSize) {
  const popSize = population.length;
  const minVal = 0, maxVal = areaSize;
  
  // Calculate fitness for current population
  const fitness = population.map(p => fitnessFunction(p, anchors, estimatedDistances));
  
  // Sort population by fitness (ascending)
  const sortedIndices = fitness.map((_, i) => i).sort((a, b) => fitness[a] - fitness[b]);
  const sortedPopulation = sortedIndices.map(i => population[i]);
  const sortedFitness = sortedIndices.map(i => fitness[i]);
  
  const newPopulation = [];
  const newFitness = [];
  
  // Virus replication phase
  for (let i = 0; i < popSize; i++) {
    // Roulette wheel selection for parent
    const fitnessSum = sortedFitness.reduce((sum, f) => sum + 1 / (1 + f), 0);
    const probabilities = sortedFitness.map(f => (1 / (1 + f)) / fitnessSum);
    const parentIdx = Math.floor(Math.random() * popSize);
    const parent = sortedPopulation[parentIdx].slice();
    
    // Frameshifting to produce proteins
    const proteins = [];
    for (let p = 0; p < NUM_OF_PROTEINS; p++) {
      const protein = [0, 0];
      if (SHIFTING_NO === 1) {
        // +1 frameshifting
        protein[0] = Math.random() * (maxVal - minVal) + minVal;
        protein[1] = parent[0];
      } else {
        // -1 frameshifting
        protein[0] = parent[1];
        protein[1] = Math.random() * (maxVal - minVal) + minVal;
      }
      // Boundary check
      protein[0] = Math.min(Math.max(protein[0], minVal), maxVal);
      protein[1] = Math.min(Math.max(protein[1], minVal), maxVal);
      proteins.push(protein);
    }
    
    // Uniform crossover to form new virion
    const newVirion = [0, 0];
    for (let j = 0; j < 2; j++) {
      const proteinIdx = Math.floor(Math.random() * NUM_OF_PROTEINS);
      newVirion[j] = proteins[proteinIdx][j];
    }
    
    // Mutation
    for (let j = 0; j < 2; j++) {
      if (Math.random() < MR) {
        newVirion[j] = Math.random() * (maxVal - minVal) + minVal;
      }
    }
    
    // Boundary check
    newVirion[0] = Math.min(Math.max(newVirion[0], minVal), maxVal);
    newVirion[1] = Math.min(Math.max(newVirion[1], minVal), maxVal);
    
    // Evaluate new solution
    const newFitnessVal = fitnessFunction(newVirion, anchors, estimatedDistances);
    newPopulation.push(newVirion);
    newFitness.push(newFitnessVal);
  }
  
  // Combine old and new populations and select top solutions
  const combinedPopulation = [...sortedPopulation, ...newPopulation];
  const combinedFitness = [...sortedFitness, ...newFitness];
  
  // Sort by fitness and select top PopNo solutions
  const finalIndices = combinedFitness.map((_, i) => i).sort((a, b) => combinedFitness[a] - combinedFitness[b]);
  const finalPopulation = finalIndices.slice(0, popSize).map(i => combinedPopulation[i]);
  const finalFitness = finalIndices.slice(0, popSize).map(i => combinedFitness[i]);
  
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