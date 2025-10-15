# 🚛 Vehicle Routing Problem with Time Windows (VRPTW) - Genetic Algorithm

A beautiful and comprehensive implementation of a genetic algorithm solving the Vehicle Routing Problem with Time Windows (VRPTW), featuring interactive visualizations and detailed analysis.

![VRPTW Visualization](https://img.shields.io/badge/Python-3.7+-blue.svg)
![Genetic Algorithm](https://img.shields.io/badge/Algorithm-Genetic-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [What is VRPTW?](#-what-is-vrptw)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Algorithm Overview](#-algorithm-overview)
- [Visualization](#-visualization)
- [Examples](#-examples)
- [Testing](#-testing)
- [Contributing](#-contributing)

## 🎯 What is VRPTW?

The **Vehicle Routing Problem with Time Windows (VRPTW)** is a complex combinatorial optimization problem that extends the classic Vehicle Routing Problem (VRP) by adding time constraints. In VRPTW:

- **Multiple vehicles** must visit a set of customers
- Each **customer has a time window** during which they must be served
- **Vehicles have limited capacity** and must return to depot
- The goal is to **minimize total distance** while respecting all constraints

VRPTW is **NP-hard**, meaning optimal solutions are computationally expensive for large instances. Genetic algorithms provide excellent approximate solutions.

## ✨ Features

- 🎯 **Complete VRPTW Implementation**: Full problem modeling with time windows, capacities, and service times
- 🧬 **Advanced Genetic Algorithm**: Tournament selection, order crossover, multiple mutation operators
- 📊 **Beautiful Visualizations**: Interactive Plotly charts showing routes, time windows, and fitness evolution
- 🔬 **Interactive Jupyter Notebook**: Step-by-step exploration with parameter experimentation
- 🧪 **Comprehensive Testing**: Unit tests covering edge cases and algorithm components
- 📈 **Performance Analysis**: Detailed metrics and convergence tracking
- 🎨 **Clean Architecture**: Modular design with separation of concerns

## 🛠 Installation

### Prerequisites
- Python 3.7+
- pip

### Install Dependencies
```bash
# Clone or download the repository
cd vrptw-genetic-algorithm

# Install requirements
pip install -r requirements.txt
```

## 🚀 Quick Start

### Run the Demo
```bash
cd examples
python demo.py
```

This will:
- Create a sample problem with 20 customers and 3 vehicles
- Run the genetic algorithm for 150 generations
- Generate interactive visualizations
- Save results as HTML and PNG files

### Interactive Exploration
```bash
cd examples
jupyter notebook vrptw_interactive.ipynb
```

## 📁 Project Structure

```
vrptw-genetic-algorithm/
├── src/
│   └── vrptw.py                 # Core VRPTW and GA implementation
├── examples/
│   ├── demo.py                  # Command-line demonstration
│   └── vrptw_interactive.ipynb  # Jupyter notebook exploration
├── tests/
│   └── test_vrptw.py           # Unit tests
├── docs/                       # Documentation
├── data/                       # Sample datasets
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🧬 Algorithm Overview

### Genetic Representation
- **Chromosome**: List of routes, one per vehicle
- **Gene**: Customer ID within a route
- **Fitness**: Total distance + penalties for constraint violations

### Genetic Operators

#### Selection
- **Tournament Selection**: Selects best individuals from random subsets
- Maintains population diversity while favoring fit solutions

#### Crossover
- **Order Crossover**: Preserves relative ordering of customers
- Handles variable-length routes appropriately

#### Mutation
- **Swap**: Exchange customers between routes
- **Move**: Relocate customer to different route
- **Reverse**: Reverse segments within routes

### Fitness Function
```
Fitness = Total_Distance + Penalty_Capacity + Penalty_Time_Windows + Penalty_Depot_Return
```

Penalties ensure constraint satisfaction while allowing infeasible solutions to guide search.

## 📊 Visualization

### Solution Visualization
- Interactive routes colored by vehicle
- Customer time windows displayed on hover
- Depot marked with star symbol
- Distance and feasibility indicators

### Fitness Evolution
- Best and average fitness over generations
- Convergence analysis
- Performance comparison across runs

## 📚 Examples

### Basic Usage
```python
from vrptw import Customer, Vehicle, VRPTWProblem, VRPTWGeneticAlgorithm

# Create problem
depot = Customer(id=0, x=0, y=0, demand=0, ready_time=0, due_time=480, service_time=0)
customers = [...]  # Your customers
vehicles = [...]   # Your vehicles

problem = VRPTWProblem(customers, vehicles, depot)

# Solve with genetic algorithm
ga = VRPTWGeneticAlgorithm(problem, population_size=100, generations=200)
best_solution, routes, fitness = ga.evolve()

# Visualize results
fig = ga.visualize_solution(routes)
fig.show()
```

### Parameter Tuning
```python
# Experiment with different parameters
ga_configs = [
    {'mutation_rate': 0.1, 'crossover_rate': 0.8},
    {'mutation_rate': 0.2, 'crossover_rate': 0.9},
    {'mutation_rate': 0.3, 'crossover_rate': 0.7}
]

for config in ga_configs:
    ga = VRPTWGeneticAlgorithm(problem, **config)
    # Compare performance...
```

## 🧪 Testing

Run the test suite:
```bash
cd tests
python -m unittest test_vrptw.py
```

Tests cover:
- Problem creation and validation
- Genetic operators (crossover, mutation)
- Fitness evaluation
- Edge cases (empty routes, single customer)
- Constraint handling

## 📈 Performance Tips

1. **Population Size**: 50-200 individuals (larger = better solutions, slower)
2. **Generations**: 100-500 (more generations = better convergence)
3. **Mutation Rate**: 0.1-0.3 (higher = more exploration)
4. **Crossover Rate**: 0.7-0.9 (higher = more exploitation)

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional crossover operators
- Local search improvement heuristics
- Parallel processing for large populations
- More benchmark problem instances
- Alternative metaheuristics (ACO, SA, etc.)

### Development Setup
```bash
git clone <repository-url>
cd vrptw-genetic-algorithm
pip install -r requirements.txt
pip install -e .  # For development
```

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by classical VRPTW literature and genetic algorithm research
- Built with modern Python data science stack
- Visualization powered by Plotly for interactive exploration

## 🔗 Related Resources

- [Vehicle Routing Problem on Wikipedia](https://en.wikipedia.org/wiki/Vehicle_routing_problem)
- [Genetic Algorithms: A Survey](https://arxiv.org/abs/1802.03877)
- [VRPTW Benchmarks](http://www.sintef.no/projectweb/top/vrptw/)

---

**Happy routing!** 🎯✨

*This project demonstrates the power of evolutionary computation in solving complex real-world optimization problems.*
