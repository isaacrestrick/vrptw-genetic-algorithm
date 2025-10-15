# VRPTW Genetic Algorithm Time-lapse Visualizations

## Overview

This directory contains scripts that generate beautiful time-lapse animations and visualizations showing how the genetic algorithm improves vehicle routing solutions over generations. You can see the algorithm discovering better and better routes in real-time!

## 📊 Available Visualizations

### 1. **Basic Time-lapse Animation** (`timelapse.py`)
The foundational time-lapse visualization with dual-panel view:
- **Left Panel**: Current best solution with routes color-coded by vehicle
- **Right Panel**: Fitness evolution graph showing convergence

**Run:**
```bash
python3 timelapse.py
```

**Outputs:**
- `vrptw_timelapse.gif` - 26.7 second animation at 3 fps (2.8 MB)
- `vrptw_solution_comparison.png` - Side-by-side snapshots of evolution

### 2. **Enhanced Time-lapse Animation** (`enhanced_timelapse.py`)
A more comprehensive visualization with detailed analytics:
- **Solution Panel** (top-left): Current best routing solution
- **Fitness Evolution** (top-right): Best fitness over generations
- **Improvement Rate** (bottom-left): Fitness gains per generation
- **Population Diversity** (bottom-center): Population variance over time
- **Progress Statistics** (bottom-right): Real-time optimization metrics

**Run:**
```bash
python3 enhanced_timelapse.py
```

**Outputs:**
- `vrptw_enhanced_timelapse.gif` - 20 second enhanced animation (3.0 MB)
- `vrptw_detailed_analysis.png` - 6-panel detailed analysis report

## 🎯 What You're Seeing

### Solution Evolution
As you watch the animation, notice:
- **Early generations**: Routes appear chaotic with many crossings
- **Mid generations**: Routes begin to optimize, reducing crossings
- **Late generations**: Clean, efficient routes emerge
- **Final solution**: Well-organized vehicle routes with minimal backtracking

### Fitness Metrics
The right panel shows three key trends:
- **Fitness Score** (y-axis): Lower is better
  - Measures total distance + penalties for constraint violations
- **Generation** (x-axis): 0-150 iterations
- **Current Generation** (red dot): Marks progress through evolution

### Convergence Pattern
- **Rapid improvement**: Generations 0-40 show steep fitness gains
- **Plateau phase**: Generations 40+ show diminishing returns (convergence)
- **Final state**: Algorithm stabilizes on a near-optimal solution

## 📈 Key Metrics Explained

### Fitness Score
```
Fitness = Total_Distance + Penalties
```
- **Total Distance**: Sum of all vehicle routes (minimized by GA)
- **Penalties**: Constraints like time windows and capacity violations
- **Lower fitness = Better solution**

### Population Diversity
Measures how different solutions are from each other:
- **High diversity**: Population exploring many solution patterns
- **Low diversity**: Population converging to similar solutions
- **Healthy pattern**: Starts high, gradually decreases as best solutions emerge

### Improvement Rate
Shows fitness gains between generations:
- **Large bars**: Significant improvements found
- **Small bars**: Marginal gains (convergence phase)
- **Zero bars**: No improvement (plateau)

## 🚀 Quick Start

1. **Install dependencies** (if not already done):
   ```bash
   pip install -r ../requirements.txt
   ```

2. **Generate visualizations**:
   ```bash
   # Basic timelapse
   python3 timelapse.py
   
   # Or enhanced version with detailed metrics
   python3 enhanced_timelapse.py
   ```

3. **View animations**:
   - Open `.gif` files in any image viewer or browser
   - Or use command: `open vrptw_timelapse.gif` (macOS) or `xdg-open` (Linux)

4. **View static analysis**:
   - Open `.png` files to see detailed snapshots and metrics

## 🎨 Customization

Want to modify the visualizations? Edit the Python files:

### Adjust animation speed (fps):
```python
create_timelapse_animation(fps=2)  # Slower (2 frames/second)
create_timelapse_animation(fps=5)  # Faster (5 frames/second)
```

### Change problem size:
```python
# In create_sample_problem():
for i in range(1, 31):  # 30 customers instead of 20
    # ...
```

### Modify GA parameters:
```python
ga = VRPTWGeneticAlgorithmWithHistory(
    problem=problem,
    population_size=200,      # Larger population
    generations=300,          # More generations
    mutation_rate=0.20,       # More exploration
    crossover_rate=0.90       # More exploitation
)
```

### Change colors:
```python
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', ...]  # RGB hex colors
```

## 📊 Sample Results

From the latest run:

```
Initial Fitness:  925,238.54
Final Fitness:    503,078.45
Total Improvement: 45.6%
Generations:      150
Population Size:  100
Convergence:      Generation 5 (fastest improvement)
```

### Detailed Analysis Metrics:
- **Best Average Fitness**: 504,769
- **Worst Average Fitness**: 2,000,550
- **Population Diversity**: High initially (5.1×10¹¹), stabilizes to ~1.6×10⁹

## 🔬 How to Interpret the Results

### Early Phase (Gen 0-50)
- Routes are poorly organized
- Fitness score drops rapidly
- Population diversity remains high as algorithm searches solution space

### Middle Phase (Gen 50-100)
- Routes show emerging structure
- Fitness improvement slows (diminishing returns)
- Population diversity decreases as good solutions emerge

### Late Phase (Gen 100-150)
- Routes are highly optimized
- Minimal fitness improvement
- Population converges to similar solutions
- Algorithm may be stuck in local optima

## 💡 Tips for Better Understanding

1. **Watch at different speeds**: Slow playback (1-2 fps) shows evolution clearly
2. **Compare early vs late**: Look at generation 0 vs generation 149
3. **Study the metrics**: Fitness graph shows when major improvements occur
4. **Experiment with parameters**: Run with different mutation rates to see effects
5. **Large problems**: Try with 50+ customers to see more dramatic improvements

## 🐛 Troubleshooting

**Animation won't generate:**
- Ensure all dependencies installed: `pip install -r ../requirements.txt`
- Check you have write permissions in the examples directory
- Verify Python 3.7+ installed: `python3 --version`

**Animation quality issues:**
- Increase DPI: Change `dpi=100` to `dpi=150` in script
- Reduce frame skip: Lower the sampling rate in `frame_generations`

**Slow generation:**
- Reduce generations: Change from 150 to 75
- Reduce population size: Change from 100 to 50
- Run on faster machine: Large animations can take 5-10 minutes

## 📚 Related Files

- `../src/vrptw.py` - Core GA implementation
- `demo.py` - Non-animated demonstration
- `vrptw_interactive.ipynb` - Interactive Jupyter notebook

## 🎓 Learning Resources

To understand the concepts shown:
1. **Genetic Algorithms**: How crossover, mutation, and selection work
2. **Vehicle Routing**: Constraints, time windows, capacity limits
3. **Optimization**: Local optima vs global optima, convergence
4. **Data Visualization**: Time-series analysis, multi-panel plots

## 📝 License

Same as parent project (MIT License)

---

**Enjoy watching your genetic algorithm optimize in real-time!** 🚀
