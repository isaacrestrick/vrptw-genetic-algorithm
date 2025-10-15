#!/usr/bin/env python3
"""
Time-lapse Visualization of VRPTW Genetic Algorithm

This script creates an animated visualization showing how the genetic algorithm
improves solutions over generations, creating a beautiful time-lapse video.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vrptw import Customer, Vehicle, VRPTWProblem, VRPTWGeneticAlgorithm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import PillowWriter
import random
from typing import List, Tuple


class VRPTWGeneticAlgorithmWithHistory(VRPTWGeneticAlgorithm):
    """Extended GA that tracks solution history for visualization."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_history = []  # Store best solution at each generation
        self.fitness_history_detailed = []  # Fitness at each generation
    
    def evolve_with_history(self) -> Tuple[List[List[int]], list, float]:
        """Run GA while recording history of best solutions."""
        population = [self.create_chromosome() for _ in range(self.population_size)]
        
        best_chromosome = None
        best_fitness = float('inf')
        best_routes = None
        
        for generation in range(self.generations):
            # Evaluate fitness
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
            
            # Record history (every few generations or key milestones)
            self.generation_history.append((generation, best_chromosome.copy(), best_routes))
            self.fitness_history_detailed.append(best_fitness)
            
            # Track fitness history
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(fitnesses))
            
            # Create new population
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
            
            if generation % 20 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.2f}")
        
        return best_chromosome, best_routes, best_fitness


def visualize_solution_on_ax(ax, problem, routes, generation, fitness, title_suffix=""):
    """Draw solution on a matplotlib axis."""
    ax.clear()
    ax.set_xlim(problem.depot.x - 50, problem.depot.x + 50)
    ax.set_ylim(problem.depot.y - 50, problem.depot.y + 50)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
              '#F7DC6F', '#BB8FCE', '#85C1E2']
    
    # Draw depot
    ax.plot(problem.depot.x, problem.depot.y, 'k*', markersize=30, 
            label='Depot', zorder=5)
    
    # Draw all customers (unvisited ones as light dots)
    for customer in problem.customers:
        ax.plot(customer.x, customer.y, 'o', color='lightgray', 
                markersize=6, alpha=0.4, zorder=1)
    
    # Draw routes
    for route_idx, route in enumerate(routes):
        if not route.customers:
            continue
        
        color = colors[route_idx % len(colors)]
        
        # Create route path
        x_coords = [problem.depot.x] + [c.x for c in route.customers] + [problem.depot.x]
        y_coords = [problem.depot.y] + [c.y for c in route.customers] + [problem.depot.y]
        
        # Draw route line
        ax.plot(x_coords, y_coords, '-', color=color, linewidth=2, 
                alpha=0.8, zorder=2, label=f'Vehicle {route.vehicle_id}')
        
        # Draw customer points
        customer_x = [c.x for c in route.customers]
        customer_y = [c.y for c in route.customers]
        ax.plot(customer_x, customer_y, 'o', color=color, markersize=8, 
                zorder=3)
        
        # Add customer labels
        for i, customer in enumerate(route.customers):
            ax.text(customer.x, customer.y - 2, f'C{customer.id}', 
                   ha='center', fontsize=7, fontweight='bold')
    
    # Add title with generation and fitness info
    improvement = ""
    if len(problem.customers) > 0:
        ax.set_title(f'Generation {generation} - Fitness: {fitness:.1f}\n{title_suffix}', 
                    fontsize=12, fontweight='bold')
    
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend(loc='upper right', fontsize=9)


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


def create_timelapse_animation(output_file='vrptw_timelapse.gif', fps=2):
    """Create an animated time-lapse of the GA improving solutions."""
    
    print("🎬 Creating VRPTW Genetic Algorithm Time-lapse Visualization")
    print("=" * 60)
    
    # Create problem
    print("📦 Creating sample problem...")
    problem = create_sample_problem()
    
    # Run GA with history tracking
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
    print(f"   - Initial fitness: {ga.fitness_history_detailed[0]:.2f}")
    print(f"   - Final fitness: {ga.fitness_history_detailed[-1]:.2f}")
    improvement = (ga.fitness_history_detailed[0] - ga.fitness_history_detailed[-1]) / ga.fitness_history_detailed[0] * 100
    print(f"   - Improvement: {improvement:.1f}%")
    
    # Create animation
    print(f"\n🎨 Creating animation with {len(ga.generation_history)} frames...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('VRPTW Genetic Algorithm: Solutions Getting Better Over Time ✨', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    writer = PillowWriter(fps=fps)
    writer.setup(fig, output_file, dpi=100)
    
    # Sample frames for smoother animation (every Nth generation, more frequent at start)
    frame_generations = []
    for i, (gen, _, _) in enumerate(ga.generation_history):
        if i < 10:  # First 10 generations every step
            frame_generations.append(i)
        elif i % 2 == 0:  # Then every other generation
            frame_generations.append(i)
    
    print(f"   - Using {len(frame_generations)} frames for animation")
    
    frame_count = 0
    for frame_idx in frame_generations:
        gen, chromosome, routes = ga.generation_history[frame_idx]
        fitness = ga.fitness_history_detailed[gen]
        
        # Left plot: Current solution
        visualize_solution_on_ax(
            ax1, problem, routes, gen, fitness,
            title_suffix=f"Routes optimized for distance & time windows"
        )
        
        # Right plot: Fitness evolution
        ax2.clear()
        generations_so_far = ga.fitness_history_detailed[:gen+1]
        ax2.plot(generations_so_far, linewidth=2.5, color='#3498db')
        ax2.fill_between(range(len(generations_so_far)), generations_so_far, 
                        alpha=0.3, color='#3498db')
        ax2.scatter([gen], [fitness], color='#e74c3c', s=100, zorder=5, 
                   label='Current Generation')
        ax2.set_xlabel('Generation', fontsize=11)
        ax2.set_ylabel('Best Fitness Score', fontsize=11)
        ax2.set_title(f'Fitness Evolution (Gen {gen})', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Set consistent y-axis limits
        all_fitness = ga.fitness_history_detailed
        ax2.set_ylim(min(all_fitness) - 50, max(all_fitness) + 50)
        
        writer.grab_frame()
        frame_count += 1
        
        if frame_count % max(1, len(frame_generations) // 10) == 0:
            progress = (frame_count / len(frame_generations)) * 100
            print(f"   - {progress:.0f}% complete ({frame_count}/{len(frame_generations)} frames)")
    
    writer.finish()
    
    print(f"\n✨ Time-lapse animation saved as '{output_file}'")
    print(f"   - Duration: {len(frame_generations) / fps:.1f} seconds at {fps} fps")
    
    return ga, problem


def create_static_comparison(problem, ga):
    """Create a static comparison figure with key snapshots."""
    print("\n📊 Creating static comparison figure...")
    
    # Select key generations to show progression
    key_gens = [
        0,  # Start
        ga.generations // 4,
        ga.generations // 2,
        3 * ga.generations // 4,
        ga.generations - 1  # End
    ]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('VRPTW Solution Evolution: From Random to Optimized', 
                 fontsize=14, fontweight='bold')
    
    for idx, ax in enumerate(axes):
        if idx < len(key_gens):
            gen_idx = key_gens[idx]
            if gen_idx < len(ga.generation_history):
                gen, chromosome, routes = ga.generation_history[gen_idx]
                fitness = ga.fitness_history_detailed[gen]
                visualize_solution_on_ax(ax, problem, routes, gen, fitness)
    
    plt.tight_layout()
    plt.savefig('vrptw_solution_comparison.png', dpi=150, bbox_inches='tight')
    print("   - Comparison figure saved as 'vrptw_solution_comparison.png'")


if __name__ == "__main__":
    # Create time-lapse animation
    ga, problem = create_timelapse_animation(fps=3)
    
    # Create static comparison
    create_static_comparison(problem, ga)
    
    print("\n🎉 All visualizations complete!")
    print("   - Open 'vrptw_timelapse.gif' to see the animated time-lapse")
    print("   - View 'vrptw_solution_comparison.png' for side-by-side snapshots")
