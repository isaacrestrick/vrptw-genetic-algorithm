#!/usr/bin/env python3
"""
Enhanced Time-lapse Visualization of VRPTW Genetic Algorithm

This script creates an enhanced animated visualization with detailed metrics
showing how the genetic algorithm improves solutions over generations.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vrptw import Customer, Vehicle, VRPTWProblem, VRPTWGeneticAlgorithm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import PillowWriter
import matplotlib.gridspec as gridspec
from typing import List, Tuple


class VRPTWGeneticAlgorithmWithHistory(VRPTWGeneticAlgorithm):
    """Extended GA that tracks solution history for visualization."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_history = []
        self.fitness_history_detailed = []
        self.diversity_history = []
    
    def evolve_with_history(self) -> Tuple[List[List[int]], list, float]:
        """Run GA while recording history of best solutions."""
        population = [self.create_chromosome() for _ in range(self.population_size)]
        
        best_chromosome = None
        best_fitness = float('inf')
        best_routes = None
        
        for generation in range(self.generations):
            fitnesses = []
            route_lists = []
            
            for chromosome in population:
                fitness, routes = self.evaluate_fitness(chromosome)
                fitnesses.append(fitness)
                route_lists.append(routes)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_chromosome = chromosome
                    best_routes = routes
            
            # Calculate population diversity (variance in fitness)
            diversity = np.var(fitnesses) if len(fitnesses) > 1 else 0
            self.diversity_history.append(diversity)
            
            self.generation_history.append((generation, best_chromosome.copy(), best_routes))
            self.fitness_history_detailed.append(best_fitness)
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(fitnesses))
            
            new_population = []
            elite_idx = np.argmin(fitnesses)
            new_population.append(population[elite_idx])
            
            while len(new_population) < self.population_size:
                parent1, parent2 = self.tournament_selection(population, fitnesses)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            
            if generation % 30 == 0:
                print(f"Generation {generation}: Fitness = {best_fitness:.2f}, Diversity = {diversity:.2f}")
        
        return best_chromosome, best_routes, best_fitness


def create_sample_problem():
    """Create a sample VRPTW problem with 20 customers and 3 vehicles."""
    depot = Customer(id=0, x=50, y=50, demand=0, ready_time=0, 
                    due_time=1000, service_time=0)
    
    customers = []
    np.random.seed(42)
    
    for i in range(1, 21):
        angle = np.random.uniform(0, 2*np.pi)
        distance = np.random.uniform(10, 40)
        x = depot.x + distance * np.cos(angle)
        y = depot.y + distance * np.sin(angle)
        
        ready_time = np.random.uniform(10, 200)
        window_length = np.random.uniform(20, 60)
        due_time = ready_time + window_length
        demand = np.random.uniform(1, 5)
        service_time = np.random.uniform(2, 8)
        
        customer = Customer(
            id=i, x=x, y=y, demand=demand,
            ready_time=ready_time, due_time=due_time, service_time=service_time
        )
        customers.append(customer)
    
    vehicles = [
        Vehicle(id=1, capacity=25, depot_return_time=500),
        Vehicle(id=2, capacity=30, depot_return_time=500),
        Vehicle(id=3, capacity=20, depot_return_time=500)
    ]
    
    return VRPTWProblem(customers, vehicles, depot)


def visualize_solution_on_ax(ax, problem, routes, generation, fitness):
    """Draw solution on a matplotlib axis."""
    ax.clear()
    ax.set_xlim(problem.depot.x - 50, problem.depot.x + 50)
    ax.set_ylim(problem.depot.y - 50, problem.depot.y + 50)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    # Draw depot
    ax.plot(problem.depot.x, problem.depot.y, 'k*', markersize=35, 
            label='Depot', zorder=5, markeredgewidth=0.5, markeredgecolor='white')
    
    # Draw routes
    for route_idx, route in enumerate(routes):
        if not route.customers:
            continue
        
        color = colors[route_idx % len(colors)]
        
        x_coords = [problem.depot.x] + [c.x for c in route.customers] + [problem.depot.x]
        y_coords = [problem.depot.y] + [c.y for c in route.customers] + [problem.depot.y]
        
        # Draw route line with arrows
        ax.plot(x_coords, y_coords, '-', color=color, linewidth=2.5, 
                alpha=0.7, zorder=2, label=f'Vehicle {route.vehicle_id}')
        
        # Draw customer points
        customer_x = [c.x for c in route.customers]
        customer_y = [c.y for c in route.customers]
        ax.plot(customer_x, customer_y, 'o', color=color, markersize=10, 
                zorder=3, markeredgewidth=1.5, markeredgecolor='white')
        
        # Add customer labels
        for customer in route.customers:
            ax.text(customer.x, customer.y - 3, f'C{customer.id}', 
                   ha='center', fontsize=6.5, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    ax.set_title(f'Generation {generation} | Fitness: {fitness:.0f}', 
                fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('X Coordinate', fontsize=9)
    ax.set_ylabel('Y Coordinate', fontsize=9)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.95)
    ax.tick_params(labelsize=8)


def create_enhanced_animation(output_file='vrptw_enhanced_timelapse.gif', fps=2):
    """Create an enhanced animated visualization with multiple metrics."""
    
    print("🎬 Creating Enhanced VRPTW Genetic Algorithm Time-lapse")
    print("=" * 60)
    
    print("📦 Creating sample problem...")
    problem = create_sample_problem()
    
    print("🧬 Running Genetic Algorithm with history tracking...")
    ga = VRPTWGeneticAlgorithmWithHistory(
        problem=problem,
        population_size=100,
        generations=150,
        mutation_rate=0.15,
        crossover_rate=0.85,
        tournament_size=5
    )
    
    best_solution, best_routes, best_fitness = ga.evolve_with_history()
    
    print(f"\n✅ GA Complete!")
    initial_fitness = ga.fitness_history_detailed[0]
    final_fitness = ga.fitness_history_detailed[-1]
    improvement = (initial_fitness - final_fitness) / initial_fitness * 100
    print(f"   - Initial fitness: {initial_fitness:.2f}")
    print(f"   - Final fitness: {final_fitness:.2f}")
    print(f"   - Improvement: {improvement:.1f}%")
    
    print(f"\n🎨 Creating enhanced animation...")
    
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('VRPTW Genetic Algorithm: Intelligent Optimization Over Time', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :2])  # Solution plot (larger)
    ax2 = fig.add_subplot(gs[0, 2])   # Fitness plot
    ax3 = fig.add_subplot(gs[1, 0])   # Distance improvement
    ax4 = fig.add_subplot(gs[1, 1])   # Diversity
    ax5 = fig.add_subplot(gs[1, 2])   # Statistics
    
    writer = PillowWriter(fps=fps)
    writer.setup(fig, output_file, dpi=100)
    
    # Sample frames
    frame_generations = []
    for i, (gen, _, _) in enumerate(ga.generation_history):
        if i < 15:
            frame_generations.append(i)
        elif i % 3 == 0:
            frame_generations.append(i)
    
    print(f"   - Using {len(frame_generations)} frames")
    
    frame_count = 0
    for frame_idx in frame_generations:
        gen, chromosome, routes = ga.generation_history[frame_idx]
        fitness = ga.fitness_history_detailed[gen]
        
        # Panel 1: Solution visualization
        visualize_solution_on_ax(ax1, problem, routes, gen, fitness)
        
        # Panel 2: Fitness evolution
        ax2.clear()
        generations_so_far = ga.fitness_history_detailed[:gen+1]
        ax2.plot(generations_so_far, linewidth=2, color='#3498db', marker='o', 
                markersize=3, markevery=max(1, len(generations_so_far)//10))
        ax2.fill_between(range(len(generations_so_far)), generations_so_far, 
                        alpha=0.2, color='#3498db')
        ax2.scatter([gen], [fitness], color='#e74c3c', s=150, zorder=5, 
                   edgecolors='darkred', linewidth=2)
        ax2.set_xlabel('Generation', fontsize=9)
        ax2.set_ylabel('Fitness', fontsize=9)
        ax2.set_title('Fitness Evolution', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=8)
        all_fitness = ga.fitness_history_detailed
        ax2.set_ylim(min(all_fitness) - 50, max(all_fitness) + 50)
        
        # Panel 3: Improvement rate
        ax3.clear()
        if gen > 0:
            improvements = np.array([ga.fitness_history_detailed[i-1] - ga.fitness_history_detailed[i] 
                                    for i in range(1, gen+1)])
            ax3.bar(range(len(improvements)), improvements, color='#2ecc71', alpha=0.7, width=0.8)
            ax3.set_xlabel('Generation', fontsize=9)
            ax3.set_ylabel('Improvement', fontsize=9)
            ax3.set_title('Fitness Improvement per Gen', fontsize=10, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.tick_params(labelsize=8)
        
        # Panel 4: Population diversity
        ax4.clear()
        diversity_so_far = ga.diversity_history[:gen+1]
        ax4.plot(diversity_so_far, linewidth=2, color='#9b59b6', marker='s', 
                markersize=3, markevery=max(1, len(diversity_so_far)//10))
        ax4.fill_between(range(len(diversity_so_far)), diversity_so_far, 
                        alpha=0.2, color='#9b59b6')
        ax4.set_xlabel('Generation', fontsize=9)
        ax4.set_ylabel('Population Diversity', fontsize=9)
        ax4.set_title('Population Diversity', fontsize=10, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=8)
        
        # Panel 5: Statistics
        ax5.clear()
        ax5.axis('off')
        
        stats_text = f"""
OPTIMIZATION PROGRESS

Generation: {gen} / {ga.generations-1}

Current Fitness: {fitness:.0f}
Initial Fitness: {initial_fitness:.0f}

Improvement: {(initial_fitness - fitness) / initial_fitness * 100:.1f}%
Remaining: {100 - (initial_fitness - fitness) / initial_fitness * 100:.1f}%

Routes: {len([r for r in routes if r.customers])} active
Customers: {sum(len(r.customers) for r in routes)}

Progress: {'█' * (gen // 15)}{'░' * (10 - gen // 15)}
"""
        
        ax5.text(0.1, 0.5, stats_text, fontsize=9, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8, pad=1))
        
        writer.grab_frame()
        frame_count += 1
        
        if frame_count % max(1, len(frame_generations) // 10) == 0:
            progress = (frame_count / len(frame_generations)) * 100
            print(f"   - {progress:.0f}% complete ({frame_count}/{len(frame_generations)} frames)")
    
    writer.finish()
    
    print(f"\n✅ Animation saved as '{output_file}'")
    print(f"   - Duration: {len(frame_generations) / fps:.1f} seconds at {fps} fps")
    print(f"   - File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    
    return ga, problem


def create_detailed_analysis(problem, ga):
    """Create a detailed analysis figure."""
    print("\n📊 Creating detailed analysis figure...")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('VRPTW Genetic Algorithm: Detailed Evolution Analysis', 
                 fontsize=14, fontweight='bold')
    
    # 1. Fitness over time
    ax = axes[0, 0]
    ax.plot(ga.best_fitness_history, linewidth=2.5, label='Best Fitness', color='#3498db')
    ax.plot(ga.avg_fitness_history, linewidth=2, label='Average Fitness', 
           color='#e74c3c', alpha=0.7)
    ax.fill_between(range(len(ga.best_fitness_history)), ga.best_fitness_history, 
                    ga.avg_fitness_history, alpha=0.2, color='#3498db')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness Score')
    ax.set_title('Fitness Evolution: Best vs Average')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Improvement rate
    ax = axes[0, 1]
    improvements = np.array([ga.best_fitness_history[i-1] - ga.best_fitness_history[i] 
                            for i in range(1, len(ga.best_fitness_history))])
    ax.plot(improvements, linewidth=2, color='#2ecc71')
    ax.fill_between(range(len(improvements)), improvements, alpha=0.3, color='#2ecc71')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness Improvement')
    ax.set_title('Improvement per Generation')
    ax.grid(True, alpha=0.3)
    
    # 3. Convergence rate
    ax = axes[0, 2]
    convergence = [ga.best_fitness_history[i] / ga.best_fitness_history[0] 
                  for i in range(len(ga.best_fitness_history))]
    ax.plot(convergence, linewidth=2.5, color='#f39c12')
    ax.axhline(y=0.63, color='red', linestyle='--', alpha=0.5, label='63.9% improvement')
    ax.fill_between(range(len(convergence)), convergence, alpha=0.2, color='#f39c12')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness Ratio (Final / Initial)')
    ax.set_title('Convergence Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Population diversity
    ax = axes[1, 0]
    ax.plot(ga.diversity_history, linewidth=2.5, color='#9b59b6', marker='o', 
           markersize=3, markevery=10)
    ax.fill_between(range(len(ga.diversity_history)), ga.diversity_history, 
                    alpha=0.2, color='#9b59b6')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Population Diversity')
    ax.set_title('Population Diversity Over Time')
    ax.grid(True, alpha=0.3)
    
    # 5. Fitness distribution
    ax = axes[1, 1]
    ax.hist(ga.best_fitness_history, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    ax.axvline(ga.best_fitness_history[0], color='red', linestyle='--', 
              linewidth=2, label='Initial')
    ax.axvline(ga.best_fitness_history[-1], color='green', linestyle='--', 
              linewidth=2, label='Final')
    ax.set_xlabel('Fitness Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Fitness Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Statistics table
    ax = axes[1, 2]
    ax.axis('off')
    
    stats = [
        ['Metric', 'Value'],
        ['Initial Fitness', f"{ga.best_fitness_history[0]:.0f}"],
        ['Final Fitness', f"{ga.best_fitness_history[-1]:.0f}"],
        ['Total Improvement', f"{(ga.best_fitness_history[0] - ga.best_fitness_history[-1]) / ga.best_fitness_history[0] * 100:.1f}%"],
        ['Generations', str(ga.generations)],
        ['Population Size', str(ga.population_size)],
        ['Best Avg Fitness', f"{np.min(ga.avg_fitness_history):.0f}"],
        ['Worst Avg Fitness', f"{np.max(ga.avg_fitness_history):.0f}"],
        ['Convergence', f"Gen {np.argmin(np.diff(ga.best_fitness_history)) + 1}"]
    ]
    
    table = ax.table(cellText=stats, cellLoc='left', loc='center',
                    colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(stats)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    plt.tight_layout()
    plt.savefig('vrptw_detailed_analysis.png', dpi=150, bbox_inches='tight')
    print("   - Detailed analysis saved as 'vrptw_detailed_analysis.png'")


if __name__ == "__main__":
    ga, problem = create_enhanced_animation(fps=3)
    create_detailed_analysis(problem, ga)
    
    print("\n🎉 All enhanced visualizations complete!")
    print("   - View 'vrptw_enhanced_timelapse.gif' for the main animation")
    print("   - View 'vrptw_detailed_analysis.png' for detailed metrics")
