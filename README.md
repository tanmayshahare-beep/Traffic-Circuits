# Traffic Circuits: Electrical Analogy for Traffic Flow Prediction

Traffic Circuits implements a novel approach to traffic flow prediction using electrical circuit analogies. The system models traffic networks as electrical circuits where traffic flow corresponds to electrical current, road resistance corresponds to electrical resistance, and travel time corresponds to voltage drop.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

Traditional traffic flow models often struggle with complex network effects and real-time prediction accuracy. This project explores an innovative approach by leveraging the mathematical similarities between electrical circuits and traffic networks:

- **Traffic Flow** ↔ **Electrical Current (I)**
- **Road Congestion** ↔ **Electrical Resistance (R)**
- **Travel Time** ↔ **Voltage Drop (V)**
- **Intersections** ↔ **Circuit Nodes**
- **Trip Demand** ↔ **Current Source**

This analogy allows us to apply well-established circuit analysis techniques to traffic flow problems, enabling efficient computation of network-wide traffic patterns.

## Project Structure

```
Traffic Circuits/
├── Iteration1_Basic_Model/
│   └── traffic_circuit.py      # Basic deterministic traffic circuit model
├── Iteration2_Enhanced_Model/
│   └── enhanced_traffic_circuit.py # Enhanced PINN-based traffic circuit model
├── Iteration3_Improved_Model/
│   └── traffic_circuit_enhanced.py  # Improved traffic circuit model
└── README.md                   # This file
```

## Iterative Development

This project evolved through three iterations, each building upon the previous one:

### Iteration 1: Basic Traffic Circuit Model (`traffic_circuit.py`)
A simplified but functional implementation demonstrating the core electrical analogy:
- Simple 4-node, 4-link linear topology
- Deterministic multipliers based on time, weather, and incidents
- Clear visualization of circuit concepts
- Suitable for educational purposes and proof-of-concept

### Iteration 2: Enhanced Traffic Circuit Model (`enhanced_traffic_circuit.py`)
An advanced implementation using Physics-Informed Neural Networks (PINN):
- Synthetic data generation with realistic traffic patterns
- Integration of Bureau of Public Roads (BPR) function
- Scenario-based simulations (morning/evening peaks, weather conditions)
- Comprehensive visualization suite
- Initial attempt at incorporating physics constraints

### Iteration 3: Improved Traffic Circuit Model (`traffic_circuit_enhanced.py`)
A refined version addressing limitations in the enhanced model:
- Proper parameter constraints (α=0.15, β=4.0 as per transportation theory)
- Better physics enforcement and regularization
- Positive R² scores indicating improved performance
- More realistic travel time predictions
- Better generalization and reduced overfitting

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Dependencies
```bash
pip install numpy pandas torch matplotlib scikit-learn seaborn scipy
```

### Setup
1. Clone or download the repository
2. Install the required dependencies
3. Run any of the models as described in the Usage section

## Usage

### Basic Model
```bash
python Iteration1_Basic_Model/traffic_circuit.py
```

This will run the basic traffic circuit demonstration, showing:
- Electrical analogy explanation
- Travel time predictions for various scenarios
- Visualizations of circuit behavior

### Enhanced Model
```bash
python Iteration2_Enhanced_Model/enhanced_traffic_circuit.py
```

This will:
- Generate synthetic traffic data
- Train a Physics-Informed Neural Network
- Run traffic circuit simulations
- Save visualizations to disk

### Improved Model
```bash
python Iteration3_Improved_Model/traffic_circuit_enhanced.py
```

This will:
- Generate realistic traffic data with proper BPR implementation
- Train an improved PINN with physics constraints
- Evaluate model performance
- Create comprehensive analysis plots

## Methodology

### Electrical Circuit Analogy

The core concept relies on the mathematical similarity between Ohm's Law and traffic flow equations:

**Ohm's Law:** `V = I × R` (Voltage = Current × Resistance)
**Traffic Analogy:** `Travel_Time = Flow × Resistance × Scale_Factor`

Where:
- **V (Voltage)** represents the travel time differential between nodes
- **I (Current)** represents the traffic flow rate
- **R (Resistance)** represents the road's resistance to flow (affected by congestion, weather, incidents)

### Network Topology

Traffic networks are represented as directed graphs where:
- Nodes represent intersections or significant points
- Links represent road segments connecting nodes
- An incidence matrix describes the network connectivity

### Physics Constraints

The enhanced models incorporate the Bureau of Public Roads (BPR) function:
```
Travel_Time = Free_Flow_Time × (1 + α × (Flow/Capacity)^β)
```

Where α and β are typically 0.15 and 4.0 respectively, based on transportation engineering literature.

## Results

### Basic Model
- Demonstrates the validity of the electrical analogy
- Shows realistic travel time predictions for different scenarios
- Provides clear visualizations of the circuit behavior

### Enhanced Model
- Successfully integrates physics laws with neural networks
- Achieves reasonable predictions across multiple scenarios
- Demonstrates the potential for scalable traffic modeling

### Improved Model
- Achieves positive R² scores (>0.5) indicating good fit
- Properly enforces physics constraints
- Generates realistic traffic patterns consistent with transportation theory

## Applications

This traffic circuit model can be applied to:

1. **Travel Time Prediction**: Estimate travel times for route planning
2. **Traffic Management**: Optimize signal timing and traffic flow
3. **Infrastructure Planning**: Evaluate the impact of new roads or changes
4. **Emergency Response**: Plan optimal evacuation routes
5. **Congestion Analysis**: Identify bottlenecks and network vulnerabilities

## Future Enhancements

Potential improvements include:

1. **Real-World Data Integration**: Connect to live traffic APIs and weather services
2. **Advanced Network Topologies**: Support for complex urban road networks
3. **Dynamic Modeling**: Incorporate temporal dynamics and predictive capabilities
4. **Multi-Modal Transportation**: Include public transit, cycling, and pedestrian flows
5. **Edge Deployment**: Optimize models for real-time applications

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The electrical circuit analogy for traffic flow is inspired by classical transportation network theory
- Physics-Informed Neural Networks methodology follows recent advances in scientific machine learning
- The Bureau of Public Roads (BPR) function is a standard in transportation engineering