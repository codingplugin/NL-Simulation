import React, { useState, useRef } from 'react';
import { soaStep, fitnessFunction as soaFitness, getCentroid as soaCentroid, estimateDistance as soaEstimate, euclideanDistance as soaEuclidean } from './optimizations/soa';
import { aquilaStep, fitnessFunction as aqFitness, getCentroid as aqCentroid, estimateDistance as aqEstimate, euclideanDistance as aqEuclidean } from './optimizations/aq';
import { dboStep, fitnessFunction as dboFitness, getCentroid as dboCentroid, estimateDistance as dboEstimate, euclideanDistance as dboEuclidean } from './optimizations/dbo';
import { covidStep, fitnessFunction as covidFitness, getCentroid as covidCentroid, estimateDistance as covidEstimate, euclideanDistance as covidEuclidean } from './optimizations/covid';
import { chimpStep, fitnessFunction as chimpFitness, getCentroid as chimpCentroid, estimateDistance as chimpEstimate, euclideanDistance as chimpEuclidean } from './optimizations/chimp';
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, LineChart, Line, Legend } from 'recharts';
import Papa from 'papaparse';
import './index.css'; // Ensure global styles are loaded
import * as XLSX from 'xlsx';
import html2canvas from 'html2canvas';

const AREA_SIZE = 200;
const VISUAL_SCALE = 2; // Make figures smaller for side-by-side
const TN_OPTIONS = [25, 50, 75, 100, 125, 150];
const AN_OPTIONS = [15, 30, 45, 60, 75, 90];
const OPTIMIZATION_TECHNIQUES = [
  { value: 'soa', label: 'Seagull Optimization Algorithm (SOA)' },
  { value: 'aquila', label: 'Aquila Optimizer (AO)' },
  { value: 'dbo', label: 'Dung Beetle Optimizer (DBO)' },
  { value: 'covid', label: 'Coronavirus Optimization Algorithm (COVIDOA)' },
  { value: 'chimp', label: 'Chimp Optimization Algorithm (ChOA)' },
];
const POP_SIZE = 20;
const MAX_ITER = 50;
const FC = 2;
const TRANSMISSION_RANGE = 30;
const NOISE_FACTOR = 0.1;
const LOCALIZATION_THRESHOLD = 2.5; // Mark as localized if best fitness < threshold

function getRandomNodes(count) {
  return Array.from({ length: count }, () => [
    Math.random() * AREA_SIZE,
    Math.random() * AREA_SIZE,
  ]);
}

export default function App() {
  const [tn, setTn] = useState(TN_OPTIONS[0]);
  const [an, setAn] = useState(AN_OPTIONS[0]);
  const [technique, setTechnique] = useState(OPTIMIZATION_TECHNIQUES[0].value);
  // Set initial state for nodes to empty arrays
  const [nodes, setNodes] = useState({ tns: [], ans: [] });
  const [simStarted, setSimStarted] = useState(false);
  const [swarm, setSwarm] = useState([]); // Array of {pop, pBest, t, localized, phase}
  const [localized, setLocalized] = useState([]); // Array of localized TN indices
  const [iteration, setIteration] = useState(0);
  const [phase, setPhase] = useState('exploration');
  const [metrics, setMetrics] = useState({ nla: 0, nle: 0, nlt: 0 });
  const animRef = useRef();
  const simStartTime = useRef(0);
  const [showLocalizable, setShowLocalizable] = useState(false);
  const [localizableTNs, setLocalizableTNs] = useState([]);
  const [popSize, setPopSize] = useState(20);
  const [maxIter, setMaxIter] = useState(50);
  const [localizationThreshold, setLocalizationThreshold] = useState(2.5);
  const [noiseFactor, setNoiseFactor] = useState(0.1);
  const [deployed, setDeployed] = useState(false);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [iterationResults, setIterationResults] = useState([]);
  const isMobile = typeof window !== 'undefined' && window.innerWidth <= 700;
  // Add state for expanded graph modal
  const [expandedGraph, setExpandedGraph] = useState(null); // 'nla', 'nle', or null
  // Add refs for chart containers
  const nlaChartRef = useRef();
  const nleChartRef = useRef();

  // Remove or comment out the useEffect that auto-generates nodes on TN/AN change
  // React.useEffect(() => {
  //   const tns = getRandomNodes(tn);
  //   const ans = getRandomNodes(an);
  //   setNodes({ tns, ans });
  //   setSimStarted(false);
  //   setSwarm([]);
  //   setLocalized([]);
  //   setIteration(0);
  //   setPhase('exploration');
  //   setMetrics({ nla: 0, nle: 0, nlt: 0 });
  //   setShowLocalizable(false);
  //   // Compute localizable TNs
  //   const localizable = tns.map((target, idx) => {
  //     const dists = ans.map(a => euclideanDistance(target, a));
  //     const inRange = dists.filter(d => d <= TRANSMISSION_RANGE).length;
  //     return inRange >= 3 ? idx : null;
  //   }).filter(idx => idx !== null);
  //   setLocalizableTNs(localizable);
  // }, [tn, an]);

  const handleShowLocalizable = () => {
    setShowLocalizable(true);
  };

  // Helper functions to select algorithm
  const getAlgoFns = () => {
    if (technique === 'soa') {
      return {
        step: soaStep,
        fitness: soaFitness,
        centroid: soaCentroid,
        estimate: soaEstimate,
        euclidean: soaEuclidean,
        areaSize: 200,
      };
    } else if (technique === 'aquila') {
      return {
        step: aquilaStep,
        fitness: aqFitness,
        centroid: aqCentroid,
        estimate: aqEstimate,
        euclidean: aqEuclidean,
        areaSize: 200,
      };
    } else if (technique === 'dbo') {
      return {
        step: dboStep,
        fitness: dboFitness,
        centroid: dboCentroid,
        estimate: dboEstimate,
        euclidean: dboEuclidean,
        areaSize: 200,
      };
    } else if (technique === 'covid') {
      return {
        step: covidStep,
        fitness: covidFitness,
        centroid: covidCentroid,
        estimate: covidEstimate,
        euclidean: covidEuclidean,
        areaSize: 200,
      };
    } else if (technique === 'chimp') {
      return {
        step: chimpStep,
        fitness: chimpFitness,
        centroid: chimpCentroid,
        estimate: chimpEstimate,
        euclidean: chimpEuclidean,
        areaSize: 200,
      };
    }
  };

  const handleRunSimulation = () => {
    setSimStarted(true);
    setIteration(0);
    setPhase('exploration');
    setMetrics({ nla: 0, nle: 0, nlt: 0 });
    setIterationResults([]); // Reset per-iteration results at the start
    simStartTime.current = performance.now();
    const algo = getAlgoFns();
    // For each TN, find anchors in range
    const newSwarm = nodes.tns.map((target, idx) => {
      const dists = nodes.ans.map(a => algo.euclidean(target, a));
      const validAnchors = nodes.ans.filter((a, i) => dists[i] <= TRANSMISSION_RANGE);
      if (validAnchors.length < 3) {
        return null; // Not localizable
      }
      const actualDists = validAnchors.map(a => algo.euclidean(target, a));
      const estDists = actualDists.map(d => algo.estimate(d, noiseFactor));
      // Init population around centroid
      const centroid = algo.centroid(validAnchors);
      const pop = Array.from({ length: popSize }, () => [
        centroid[0] + (Math.random() - 0.5) * 10,
        centroid[1] + (Math.random() - 0.5) * 10,
      ]);
      let pBest = pop[0];
      let bestFit = algo.fitness(pBest, validAnchors, estDists);
      for (let i = 1; i < pop.length; i++) {
        const fit = algo.fitness(pop[i], validAnchors, estDists);
        if (fit < bestFit) {
          pBest = pop[i];
          bestFit = fit;
        }
      }
      return {
        pop,
        pBest,
        bestFit,
        anchors: validAnchors,
        estDists,
        t: 0,
        localized: false,
        idx,
        target,
        phase: 'exploration',
      };
    });
    setSwarm(newSwarm);
    setLocalized([]);
    if (animRef.current) cancelAnimationFrame(animRef.current);
    setTimeout(() => animateSOA(newSwarm, 0), 300);
  };

  // Animate SOA
  function animateSOA(currentSwarm, iter) {
    const algo = getAlgoFns();
    let updatedSwarm = currentSwarm.map((s) => {
      if (!s || s.localized) return s;
      // Determine phase
      const phase = iter < maxIter / 2 ? 'exploration' : 'exploitation';
      // SOA step
      let stepResult;
      if (technique === 'soa') {
        stepResult = algo.step(
          s.pop,
          s.pBest,
          s.anchors,
          s.estDists,
          iter,
          maxIter
        );
      } else if (technique === 'aquila') {
        stepResult = algo.step(
          s.pop,
          s.pBest,
          s.anchors,
          s.estDists,
          iter,
          maxIter,
          algo.areaSize
        );
        // aquilaStep returns { newPop, newBest, bestFit }
        stepResult = {
          newPop: stepResult.newPop,
          newPBest: stepResult.newBest,
          bestFit: stepResult.bestFit,
        };
      } else if (technique === 'dbo') {
        stepResult = algo.step(
          s.pop,
          s.pBest,
          s.anchors,
          s.estDists,
          iter,
          maxIter,
          algo.areaSize
        );
        // dboStep returns { newPop, newBest, bestFit }
        stepResult = {
          newPop: stepResult.newPop,
          newPBest: stepResult.newBest,
          bestFit: stepResult.bestFit,
        };
      } else if (technique === 'covid') {
        stepResult = algo.step(
          s.pop,
          s.pBest,
          s.anchors,
          s.estDists,
          iter,
          maxIter,
          algo.areaSize
        );
        // covidStep returns { newPop, newBest, bestFit }
        stepResult = {
          newPop: stepResult.newPop,
          newPBest: stepResult.newBest,
          bestFit: stepResult.bestFit,
        };
      } else if (technique === 'chimp') {
        stepResult = algo.step(
          s.pop,
          s.pBest,
          s.anchors,
          s.estDists,
          iter,
          maxIter,
          algo.areaSize
        );
        // chimpStep returns { newPop, newBest, bestFit }
        stepResult = {
          newPop: stepResult.newPop,
          newPBest: stepResult.newBest,
          bestFit: stepResult.bestFit,
        };
      }
      const { newPop, newPBest, bestFit } = stepResult;
      // Localize if bestFit < threshold
      const localized = bestFit < localizationThreshold;
      return {
        ...s,
        pop: newPop,
        pBest: newPBest,
        bestFit,
        t: iter,
        localized,
        phase,
      };
    });
    // Mark localized TNs
    const localizedIndices = updatedSwarm
      .map((s, i) => (s && s.localized ? s.idx : null))
      .filter((v) => v !== null);
    setSwarm(updatedSwarm);
    setLocalized(localizedIndices);
    setIteration(iter);
    setPhase(iter < maxIter / 2 ? 'exploration' : 'exploitation');
    // Per-iteration metrics
    const nla = localizedIndices.length;
    let nle = 0;
    let nlt = 0;
    if (nla > 0) {
      nle =
        updatedSwarm
          .filter((s) => s && s.localized)
          .reduce((acc, s) => acc + getAlgoFns().euclidean(s.pBest, s.target), 0) / nla;
      nlt = (performance.now() - simStartTime.current) / 1000 / nla;
    }
    setIterationResults(prev => [...prev, { iter, nla, nle: nle.toFixed(3), nlt: nlt.toFixed(3) }]);
    // If done, calculate metrics
    if (
      (iter >= maxIter - 1 || updatedSwarm.every((s) => !s || s.localized)) &&
      updatedSwarm.length > 0
    ) {
      setMetrics({ nla, nle: nle.toFixed(3), nlt: nlt.toFixed(3) });
    } else if (iter < maxIter - 1 && updatedSwarm.some((s) => s && !s.localized)) {
      animRef.current = requestAnimationFrame(() => animateSOA(updatedSwarm, iter + 1));
    }
  }

  // Helper to scale positions and sizes
  const scale = (v) => v * VISUAL_SCALE;

  // Add deployNodes function
  const deployNodes = () => {
    const tns = getRandomNodes(tn);
    const ans = getRandomNodes(an);
    setNodes({ tns, ans });
    setSimStarted(false);
    setSwarm([]);
    setLocalized([]);
    setIteration(0);
    setPhase('exploration');
    setMetrics({ nla: 0, nle: 0, nlt: 0 });
    setShowLocalizable(false);
    setDeployed(true);
    // Compute localizable TNs
    const localizable = tns.map((target, idx) => {
      const dists = ans.map(a => getAlgoFns().euclidean(target, a));
      const inRange = dists.filter(d => d <= TRANSMISSION_RANGE).length;
      return inRange >= 3 ? idx : null;
    }).filter(idx => idx !== null);
    setLocalizableTNs(localizable);
  };

  const downloadExcel = () => {
    // Prepare data: add technique as a header row
    const techniqueLabel = OPTIMIZATION_TECHNIQUES.find(opt => opt.value === technique)?.label || technique;
    const header = [[`Optimization Technique: ${techniqueLabel}`]];
    const columns = [['Iteration', 'NLA', 'NLE (m)', 'NLT (s)']];
    const data = iterationResults.map(r => [r.iter + 1, r.nla, r.nle, r.nlt]);
    const wsData = [...header, ...columns, ...data];
    const ws = XLSX.utils.aoa_to_sheet(wsData);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'Summary');
    XLSX.writeFile(wb, `localization_summary_${technique}.xlsx`);
  };

  // Download handler for NLA
  const handleDownloadNLA = async () => {
    if (nlaChartRef.current) {
      const canvas = await html2canvas(nlaChartRef.current, { backgroundColor: '#fff', useCORS: true });
      const link = document.createElement('a');
      link.download = 'NLA_vs_Iteration.png';
      link.href = canvas.toDataURL('image/png');
      link.click();
    }
  };
  // Download handler for NLE
  const handleDownloadNLE = async () => {
    if (nleChartRef.current) {
      const canvas = await html2canvas(nleChartRef.current, { backgroundColor: '#fff', useCORS: true });
      const link = document.createElement('a');
      link.download = 'NLE_vs_Iteration.png';
      link.href = canvas.toDataURL('image/png');
      link.click();
    }
  };

  return (
    <div className="container">
      <h1>Swarm Node Localization Simulation</h1>
      {/* Optimization Technique Selector */}
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', marginTop: 12, marginBottom: 8 }}>
        <label style={{ fontWeight: 600, marginRight: 8 }}>Optimization Technique:</label>
        <select value={technique} onChange={e => setTechnique(e.target.value)} style={{ fontSize: 15, padding: '4px 10px', borderRadius: 6, border: '1.5px solid #b0bec5', background: '#f8fafc', fontWeight: 500 }}>
          {OPTIMIZATION_TECHNIQUES.map(opt => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </div>
      {/* Main Layout: Settings and Figures */}
      <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', justifyContent: 'center', alignItems: isMobile ? 'stretch' : 'center', gap: 32, width: '100%', marginTop: 32 }}>
        {/* Settings Panels */}
        {isMobile ? (
          <div className="mobile-settings-row">
            <div className="mobile-settings-panel" style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 8, background: '#f8fafc', border: '1.5px solid #b0bec5', borderRadius: 10, padding: '16px 10px', boxShadow: '0 1px 4px 0 rgba(0,0,0,0.04)' }}>
              <span style={{ fontWeight: 600, marginBottom: 2 }}>Network Settings</span>
              <label>Target Nodes (TNs):
                <select value={tn} onChange={e => { setTn(Number(e.target.value)); setDeployed(false); }} style={{ marginLeft: 4 }}>
                  {TN_OPTIONS.map(opt => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              </label>
              <label>Anchor Nodes (ANs):
                <select value={an} onChange={e => { setAn(Number(e.target.value)); setDeployed(false); }} style={{ marginLeft: 4 }}>
                  {AN_OPTIONS.map(opt => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              </label>
              <button onClick={deployNodes} style={{ fontWeight: 600, background: '#388e3c', marginTop: 6 }}>
                Deploy
              </button>
            </div>
            <div className="mobile-settings-panel" style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 8, background: '#f8fafc', border: '1.5px solid #b0bec5', borderRadius: 10, padding: '16px 10px', boxShadow: '0 1px 4px 0 rgba(0,0,0,0.04)' }}>
              <span style={{ fontWeight: 600, marginBottom: 2 }}>SOA Settings</span>
              <label>Population Size:
                <input type="number" min={5} max={200} value={popSize} onChange={e => setPopSize(Number(e.target.value))} style={{ width: 60, marginLeft: 4 }} />
              </label>
              <label>Iterations:
                <input type="number" min={10} max={500} value={maxIter} onChange={e => setMaxIter(Number(e.target.value))} style={{ width: 60, marginLeft: 4 }} />
              </label>
              <label>Threshold:
                <input type="number" min={0.1} max={20} step={0.1} value={localizationThreshold} onChange={e => setLocalizationThreshold(Number(e.target.value))} style={{ width: 60, marginLeft: 4 }} />
              </label>
              <label>Noise:
                <input type="number" min={0} max={1} step={0.01} value={noiseFactor} onChange={e => setNoiseFactor(Number(e.target.value))} style={{ width: 60, marginLeft: 4 }} />
              </label>
            </div>
          </div>
        ) : (
          <>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 8, minWidth: 180, background: '#f8fafc', border: '1.5px solid #b0bec5', borderRadius: 10, padding: '16px 18px', boxShadow: '0 1px 4px 0 rgba(0,0,0,0.04)' }}>
              <span style={{ fontWeight: 600, marginBottom: 2 }}>Network Settings</span>
              <label>Target Nodes (TNs):
                <select value={tn} onChange={e => { setTn(Number(e.target.value)); setDeployed(false); }} style={{ marginLeft: 4 }}>
                  {TN_OPTIONS.map(opt => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              </label>
              <label>Anchor Nodes (ANs):
                <select value={an} onChange={e => { setAn(Number(e.target.value)); setDeployed(false); }} style={{ marginLeft: 4 }}>
                  {AN_OPTIONS.map(opt => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              </label>
              <button onClick={deployNodes} style={{ fontWeight: 600, background: '#388e3c', marginTop: 6 }}>
                Deploy
              </button>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 8, minWidth: 220, background: '#f8fafc', border: '1.5px solid #b0bec5', borderRadius: 10, padding: '16px 18px', boxShadow: '0 1px 4px 0 rgba(0,0,0,0.04)' }}>
              <span style={{ fontWeight: 600, marginBottom: 2 }}>SOA Settings</span>
              <label>Population Size:
                <input type="number" min={5} max={200} value={popSize} onChange={e => setPopSize(Number(e.target.value))} style={{ width: 60, marginLeft: 4 }} />
              </label>
              <label>Iterations:
                <input type="number" min={10} max={500} value={maxIter} onChange={e => setMaxIter(Number(e.target.value))} style={{ width: 60, marginLeft: 4 }} />
              </label>
              <label>Threshold:
                <input type="number" min={0.1} max={20} step={0.1} value={localizationThreshold} onChange={e => setLocalizationThreshold(Number(e.target.value))} style={{ width: 60, marginLeft: 4 }} />
              </label>
              <label>Noise:
                <input type="number" min={0} max={1} step={0.01} value={noiseFactor} onChange={e => setNoiseFactor(Number(e.target.value))} style={{ width: 60, marginLeft: 4 }} />
              </label>
            </div>
          </>
        )}
        {/* Figures - Center */}
        <div className="sim-figures" style={{ flexDirection: isMobile ? 'column' : 'row', gap: 32, justifyContent: 'center', alignItems: isMobile ? 'stretch' : 'flex-start', marginLeft: isMobile ? 0 : -16 }}>
          {/* Deployment View */}
          <div>
            <div className="sim-figures-title">Deployment & Localizable TNs</div>
            <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 18, marginBottom: 8 }}>
              <button onClick={handleShowLocalizable} disabled={!deployed}>
                Show Localizable TNs
              </button>
              {showLocalizable && (
                <div style={{ fontWeight: 400, fontSize: 15, color: '#43a047' }}>
                  Localizable TNs: <b>{localizableTNs.length}</b>
                </div>
              )}
            </div>
            <div style={{
              width: scale(AREA_SIZE), height: scale(AREA_SIZE), border: '2px solid #333', position: 'relative', background: '#f9f9f9',
            }}>
              {/* Draw ANs */}
              {nodes.ans.length > 0 && nodes.ans.map(([x, y], i) => (
                <div key={`an-${i}`} style={{
                  position: 'absolute', left: scale(x) - 4, top: scale(y) - 4, width: 8, height: 8, background: '#1976d2', borderRadius: '50%', border: '2px solid #fff', zIndex: 2,
                }} title={`Anchor Node ${i+1}`}></div>
              ))}
              {/* Draw TNs, highlight localizable if showLocalizable */}
              {nodes.tns.length > 0 && nodes.tns.map(([x, y], i) => (
                <div key={`tn-${i}`} style={{
                  position: 'absolute', left: scale(x) - 3, top: scale(y) - 3, width: 6, height: 6, background: showLocalizable && localizableTNs.includes(i) ? '#43a047' : '#e53935', borderRadius: '50%', border: '1px solid #fff', zIndex: 1,
                }} title={`Target Node ${i+1}`}></div>
              ))}
            </div>
          </div>
          {/* Simulation View */}
          <div>
            <div className="sim-figures-title">Simulation</div>
            <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 18, marginBottom: 8 }}>
              <button onClick={handleRunSimulation} disabled={!deployed}>
                Run Simulation
              </button>
              {(metrics.nla > 0 || metrics.nle > 0) && (
                <div className="metrics-card" style={{ margin: 0 }}>
                  <span style={{ marginRight: 16 }}>NLA: <b>{metrics.nla}</b></span>
                  <span style={{ marginRight: 16 }}>NLE: <b>{metrics.nle} m</b></span>
                  <span>NLT: <b>{metrics.nlt} s</b></span>
                </div>
              )}
            </div>
            <div style={{
              width: scale(AREA_SIZE), height: scale(AREA_SIZE), border: '2px solid #333', position: 'relative', background: '#f9f9f9',
            }}>
              {/* Draw ANs */}
              {nodes.ans.length > 0 && nodes.ans.map(([x, y], i) => (
                <div key={`an-${i}`} style={{
                  position: 'absolute', left: scale(x) - 4, top: scale(y) - 4, width: 8, height: 8, background: '#1976d2', borderRadius: '50%', border: '2px solid #fff', zIndex: 2,
                }} title={`Anchor Node ${i+1}`}></div>
              ))}
              {/* Draw TNs, localized turn green after simulation */}
              {nodes.tns.length > 0 && nodes.tns.map(([x, y], i) => (
                <div key={`tn-sim-${i}`} style={{
                  position: 'absolute', left: scale(x) - 3, top: scale(y) - 3, width: 6, height: 6, background: localized.includes(i) ? '#43a047' : '#e53935', borderRadius: '50%', border: '1px solid #fff', zIndex: 1,
                }} title={`Target Node ${i+1}`}></div>
              ))}
              {/* Draw SOA population for each localizable TN, all pink */}
              {swarm.map((s, idx) => s && !s.localized && s.pop.map(([x, y], j) => (
                <div key={`swarm-${idx}-${j}`} style={{
                  position: 'absolute', left: scale(x) - 2, top: scale(y) - 2, width: 4, height: 4, background: '#e91e63', borderRadius: '50%', opacity: 0.7, zIndex: 3,
                }}></div>
              )))}
            </div>
          </div>
        </div>
      </div>
      {/* Legend */}
      <div style={{ margin: '24px auto 0 auto', maxWidth: scale(AREA_SIZE) * 2 + 64, display: 'flex', gap: 24, fontSize: 15, justifyContent: 'center' }}>
        <span><span style={{ display: 'inline-block', width: 16, height: 16, background: '#1976d2', borderRadius: '50%', border: '2px solid #fff', verticalAlign: 'middle', marginRight: 4 }}></span> Anchor Node (Blue)</span>
        <span><span style={{ display: 'inline-block', width: 14, height: 14, background: '#e53935', borderRadius: '50%', border: '1px solid #fff', verticalAlign: 'middle', marginRight: 4 }}></span> Target Node (Red)</span>
        <span><span style={{ display: 'inline-block', width: 14, height: 14, background: '#43a047', borderRadius: '50%', border: '1px solid #fff', verticalAlign: 'middle', marginRight: 4 }}></span> Localizable/Localized TN (Green)</span>
        <span><span style={{ display: 'inline-block', width: 14, height: 14, background: '#e91e63', borderRadius: '50%', opacity: 0.7, marginRight: 4, border: '1px solid #fff', verticalAlign: 'middle' }}></span> Swarm (Pink)</span>
      </div>
      {/* After simulation, show Analyse button below the simulation figure */}
      {/* Modal popup for analysis */}
      {(simStarted || metrics.nla > 0 || metrics.nle > 0) && !showAnalysis && (
        <div style={{ display: 'flex', justifyContent: 'center', marginTop: 16 }}>
          <button onClick={() => setShowAnalysis(true)} style={{ background: '#1976d2', color: '#fff', fontWeight: 600 }}>
            Analyse
          </button>
        </div>
      )}
      {showAnalysis && (
        <div style={{
          position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', background: 'rgba(0,0,0,0.25)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <div
            style={{
              background: '#fff',
              borderRadius: 12,
              boxShadow: '0 4px 24px 0 rgba(0,0,0,0.18)',
              padding: window.innerWidth <= 700 ? 10 : 32,
              minWidth: window.innerWidth <= 700 ? 0 : 420,
              minHeight: window.innerWidth <= 700 ? 0 : 320,
              maxWidth: window.innerWidth <= 700 ? '98vw' : '90vw',
              maxHeight: window.innerWidth <= 700 ? '98vh' : '90vh',
              overflow: 'auto',
              position: 'relative',
              display: 'flex',
              flexDirection: 'column',
              fontSize: window.innerWidth <= 700 ? '0.98rem' : undefined,
            }}
          >
            <button onClick={() => setShowAnalysis(false)} style={{ position: 'absolute', top: 12, right: 16, background: '#e53935', color: '#fff', borderRadius: 6, border: 'none', fontWeight: 700, fontSize: 18, width: 32, height: 32, cursor: 'pointer' }}>×</button>
            <h2 style={{ marginTop: 0, marginBottom: 18, textAlign: 'center', fontSize: window.innerWidth <= 700 ? '1.1rem' : undefined }}>Simulation Analysis</h2>
            {/* Scrollable content box for charts and table */}
            <div style={{ flex: 1, minHeight: 0, maxHeight: window.innerWidth <= 700 ? '60vh' : '60vh', overflowY: 'auto', border: '1.5px solid #b0bec5', borderRadius: 8, padding: 16, background: '#f8fafc' }}>
              <div style={{ display: 'flex', flexDirection: 'row', gap: 32, justifyContent: 'center', alignItems: 'flex-start', flexWrap: 'wrap' }}>
                {/* Line Chart: NLA vs Iteration */}
                <div style={{ minWidth: 260, height: 220, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <div style={{ textAlign: 'center', fontWeight: 500, marginBottom: 6 }}>Localized TNs vs Iteration (NLA)</div>
                  <div ref={nlaChartRef} style={{ width: '100%' }}>
                    <ResponsiveContainer width="100%" height={180}>
                      <LineChart data={iterationResults.map(r => ({ iter: r.iter + 1, nla: Number(r.nla) }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="iter" label={{ value: 'Iteration', position: 'insideBottomRight', offset: -5 }} allowDecimals={false} />
                        <YAxis label={{ value: 'NLA', angle: -90, position: 'insideLeft' }} allowDecimals={false} />
                        <Tooltip />
                        <Legend />
                        <Line dataKey="nla" stroke="#43a047" dot={false} strokeWidth={2} connectNulls={true} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
                    <button style={{ fontSize: 13, padding: '2px 10px', borderRadius: 5, background: '#1976d2', color: '#fff', border: 'none', cursor: 'pointer' }} onClick={() => setExpandedGraph('nla')}>Expand</button>
                    <button style={{ fontSize: 13, padding: '2px 10px', borderRadius: 5, background: '#43a047', color: '#fff', border: 'none', cursor: 'pointer' }} onClick={handleDownloadNLA}>Download</button>
                  </div>
                </div>
                {/* Line Chart: NLE vs Iteration */}
                <div style={{ minWidth: 260, height: 220, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                  <div style={{ textAlign: 'center', fontWeight: 500, marginBottom: 6 }}>Localization Error vs Iteration (NLE)</div>
                  <div ref={nleChartRef} style={{ width: '100%' }}>
                    <ResponsiveContainer width="100%" height={180}>
                      <LineChart data={iterationResults.map(r => ({ iter: r.iter + 1, nle: Number(r.nle) }))}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="iter" label={{ value: 'Iteration', position: 'insideBottomRight', offset: -5 }} allowDecimals={false} />
                        <YAxis label={{ value: 'NLE (m)', angle: -90, position: 'insideLeft' }} />
                        <Tooltip />
                        <Legend />
                        <Line dataKey="nle" stroke="#1976d2" dot={false} strokeWidth={2} connectNulls={true} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
                    <button style={{ fontSize: 13, padding: '2px 10px', borderRadius: 5, background: '#1976d2', color: '#fff', border: 'none', cursor: 'pointer' }} onClick={() => setExpandedGraph('nle')}>Expand</button>
                    <button style={{ fontSize: 13, padding: '2px 10px', borderRadius: 5, background: '#43a047', color: '#fff', border: 'none', cursor: 'pointer' }} onClick={handleDownloadNLE}>Download</button>
                  </div>
                </div>
              </div>
              {/* Results Table */}
              <div style={{ marginTop: 24 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <div style={{ fontWeight: 500 }}>Summary Table (per iteration)</div>
                  <button onClick={downloadExcel} style={{ background: '#43a047', color: '#fff', fontWeight: 500, borderRadius: 4, padding: '1px 7px', border: 'none', cursor: 'pointer', fontSize: 11, height: 22, minWidth: 0 }}>
                    Download as Excel
                  </button>
                </div>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 15 }}>
                  <thead>
                    <tr style={{ background: '#f5f5f5' }}>
                      <th style={{ border: '1px solid #b0bec5', padding: '6px 12px' }}>Iteration</th>
                      <th style={{ border: '1px solid #b0bec5', padding: '6px 12px' }}>NLA</th>
                      <th style={{ border: '1px solid #b0bec5', padding: '6px 12px' }}>NLE (m)</th>
                      <th style={{ border: '1px solid #b0bec5', padding: '6px 12px' }}>NLT (s)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {iterationResults.map((r, i) => (
                      <tr key={i}>
                        <td style={{ border: '1px solid #b0bec5', padding: '6px 12px' }}>{r.iter + 1}</td>
                        <td style={{ border: '1px solid #b0bec5', padding: '6px 12px' }}>{r.nla}</td>
                        <td style={{ border: '1px solid #b0bec5', padding: '6px 12px' }}>{r.nle}</td>
                        <td style={{ border: '1px solid #b0bec5', padding: '6px 12px' }}>{r.nlt}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
      {/* Expanded Graph Modal */}
      {expandedGraph && (
        <div style={{
          position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', background: 'rgba(0,0,0,0.35)', zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <div style={{ background: '#fff', borderRadius: 12, boxShadow: '0 4px 24px 0 rgba(0,0,0,0.18)', padding: 24, minWidth: 400, minHeight: 320, maxWidth: '90vw', maxHeight: '90vh', overflow: 'auto', position: 'relative', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <button onClick={() => setExpandedGraph(null)} style={{ position: 'absolute', top: 12, right: 16, background: '#e53935', color: '#fff', borderRadius: 6, border: 'none', fontWeight: 700, fontSize: 18, width: 32, height: 32, cursor: 'pointer' }}>×</button>
            <div style={{ fontWeight: 600, fontSize: 18, marginBottom: 12 }}>
              {expandedGraph === 'nla' ? 'Localized TNs vs Iteration (NLA)' : 'Localization Error vs Iteration (NLE)'}
            </div>
            <ResponsiveContainer width={window.innerWidth > 700 ? 700 : '98vw'} height={window.innerWidth > 700 ? 420 : 320}>
              <LineChart data={expandedGraph === 'nla' ? iterationResults.map(r => ({ iter: r.iter + 1, nla: Number(r.nla) })) : iterationResults.map(r => ({ iter: r.iter + 1, nle: Number(r.nle) }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="iter" label={{ value: 'Iteration', position: 'insideBottomRight', offset: -5 }} allowDecimals={false} />
                <YAxis label={{ value: expandedGraph === 'nla' ? 'NLA' : 'NLE (m)', angle: -90, position: 'insideLeft' }} allowDecimals={expandedGraph === 'nla'} />
                <Tooltip />
                <Legend />
                {expandedGraph === 'nla' ? (
                  <Line dataKey="nla" stroke="#43a047" dot={false} strokeWidth={2} connectNulls={true} />
                ) : (
                  <Line dataKey="nle" stroke="#1976d2" dot={false} strokeWidth={2} connectNulls={true} />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
} 