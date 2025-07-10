# Node Localization Simulation Platform

A React-based simulation platform for wireless sensor network node localization using multiple bio-inspired optimization algorithms.

## 🚀 Features

- **Multiple Optimization Algorithms**: Supports 5 different bio-inspired optimization techniques
- **Interactive Visualization**: Real-time simulation with animated swarm behavior
- **Performance Analysis**: Comprehensive metrics and iteration-based analysis
- **Responsive Design**: Works on both desktop and mobile devices

## 🧬 Supported Optimization Algorithms

1. **Seagull Optimization Algorithm (SOA)**
   - Inspired by seagull behavior and migration patterns
   - Efficient exploration and exploitation phases

2. **Aquila Optimizer (AO)**
   - Based on Aquila hunting behavior
   - Four different hunting strategies

3. **Dung Beetle Optimizer (DBO)**
   - Inspired by dung beetle behavior
   - Ball rolling, dancing, and breeding phases

4. **Coronavirus Optimization Algorithm (COVIDOA)**
   - Based on coronavirus spread patterns
   - Infection and recovery mechanisms

5. **Chimp Optimization Algorithm (ChOA)**
   - Inspired by chimpanzee social behavior
   - Multiple hunting strategies and social dynamics

## 📊 Metrics Tracked

- **NLA (Number of Localized Anchors)**: Count of successfully localized target nodes
- **NLE (Network Localization Error)**: Average localization error in meters
- **NLT (Network Localization Time)**: Time taken for localization in seconds

## 🛠️ Technology Stack

- **Frontend**: React 18 with Vite
- **Charts**: Recharts for data visualization
- **Styling**: CSS3 with responsive design
- **Data Processing**: Papa Parse for CSV handling

## 🚀 Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/codingplugin/NL-Simulation.git
cd NL-Simulation
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser and navigate to `http://localhost:5173`

## 📖 Usage

### Basic Workflow

1. **Deploy Nodes**: Set the number of Target Nodes (TNs) and Anchor Nodes (ANs), then click "Deploy"
2. **Configure Parameters**: Adjust population size, iterations, threshold, and noise factor
3. **Select Algorithm**: Choose from the 5 available optimization techniques
4. **Run Simulation**: Click "Run Simulation" to start the optimization process
5. **Analyze Results**: Use the "Analyse" button to view detailed performance metrics

### Parameters

- **Population Size**: Number of particles in the swarm (default: 20)
- **Iterations**: Maximum number of optimization iterations (default: 50)
- **Threshold**: Fitness threshold for considering a node localized (default: 2.5)
- **Noise Factor**: Noise level in distance measurements (default: 0.1)

### Visualization

- **Blue dots**: Anchor Nodes (ANs)
- **Red dots**: Target Nodes (TNs)
- **Green dots**: Localized/Localizable TNs
- **Pink dots**: Swarm particles during optimization

## 📈 Analysis Features

The platform provides comprehensive analysis including:

- **NLA vs Iteration**: Shows how the number of localized nodes progresses over time
- **NLE vs Iteration**: Displays localization error reduction over iterations
- **Summary Table**: Detailed per-iteration metrics
- **Performance Comparison**: Compare different algorithms' effectiveness

## 🔬 Algorithm Details

### Seagull Optimization Algorithm (SOA)
- **Exploration Phase**: Global search using spiral motion
- **Exploitation Phase**: Local search with attack behavior
- **Parameters**: FC (frequency control), population size, iterations

### Aquila Optimizer (AO)
- **High Soar**: Global exploration
- **Contour Flight**: Local search
- **Low Flight**: Exploitation
- **Descent Attack**: Fine-tuning

### Dung Beetle Optimizer (DBO)
- **Ball Rolling**: Global exploration
- **Dancing**: Local search
- **Breeding**: Exploitation
- **Foraging**: Fine-tuning

### Coronavirus Optimization Algorithm (COVIDOA)
- **Infection Phase**: Global spread simulation
- **Recovery Phase**: Local optimization
- **Immunity**: Convergence mechanism

### Chimp Optimization Algorithm (ChOA)
- **Chaser**: Global exploration
- **Barrier**: Local search
- **Driver**: Exploitation
- **Attacker**: Fine-tuning

## 📁 Project Structure

```
src/
├── App.jsx                 # Main application component
├── main.jsx               # Application entry point
├── index.css              # Global styles
└── optimizations/         # Optimization algorithms
    ├── soa.js            # Seagull Optimization Algorithm
    ├── aq.js             # Aquila Optimizer
    ├── dbo.js            # Dung Beetle Optimizer
    ├── covid.js          # Coronavirus Optimization Algorithm
    └── chimp.js          # Chimp Optimization Algorithm
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Research papers and original implementations of the bio-inspired algorithms
- React and Vite communities for the excellent development tools
- Recharts library for beautiful data visualizations

## 📞 Contact

- GitHub: [@codingplugin](https://github.com/codingplugin)
- Project Link: [https://github.com/codingplugin/NL-Simulation](https://github.com/codingplugin/NL-Simulation)

---

**Note**: This simulation platform is designed for educational and research purposes in wireless sensor network node localization using bio-inspired optimization algorithms. 