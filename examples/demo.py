#!/usr/bin/env python3
"""
Demonstration of VRPTW Genetic Algorithm Solver

This script creates a sample VRPTW problem and solves it using the genetic algorithm.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vrptw import Customer, Vehicle, VRPTWProblem, VRPTWGeneticAlgorithm
import numpy as np


def create_sample_problem():
    """Create a sample VRPTW problem with 20 customers and 3 vehicles."""

    # Depot at center
    depot = Customer(id=0, x=50, y=50, demand=0, ready_time=0, due_time=1000, service_time=0)

    # Create 20 customers with time windows
    customers = []
    np.random.seed(42)  # For reproducible results

    for i in range(1, 21):
        # Random positions around depot
        angle = np.random.uniform(0, 2*np.pi)
        distance = np.random.uniform(10, 40)
        x = depot.x + distance * np.cos(angle)
        y = depot.y + distance * np.sin(angle)

        # Random time windows
        ready_time = np.random.uniform(10, 200)
        window_length = np.random.uniform(20, 60)
        due_time = ready_time + window_length

        # Random demand (1-5 units)
        demand = np.random.uniform(1, 5)

        # Service time (2-8 time units)
        service_time = np.random.uniform(2, 8)

        customer = Customer(
            id=i, x=x, y=y, demand=demand,
            ready_time=ready_time, due_time=due_time, service_time=service_time
        )
        customers.append(customer)

    # Create 3 vehicles with different capacities
    vehicles = [
        Vehicle(id=1, capacity=25, depot_return_time=500),
        Vehicle(id=2, capacity=30, depot_return_time=500),
        Vehicle(id=3, capacity=20, depot_return_time=500)
    ]

    return VRPTWProblem(customers, vehicles, depot)


def main():
    """Run the VRPTW genetic algorithm demonstration."""
    print("🚛 VRPTW Genetic Algorithm Demonstration")
    print("=" * 50)

    # Create problem instance
    print("📦 Creating sample problem with 20 customers and 3 vehicles...")
    problem = create_sample_problem()

    print(f"   - Customers: {len(problem.customers)}")
    print(f"   - Vehicles: {len(problem.vehicles)}")
    print(f"   - Total demand: {sum(c.demand for c in problem.customers):.1f}")
    print(f"   - Vehicle capacities: {[v.capacity for v in problem.vehicles]}")

    # Create and run genetic algorithm
    print("\n🧬 Running Genetic Algorithm...")
    ga = VRPTWGeneticAlgorithm(
        problem=problem,
        population_size=100,
        generations=150,
        mutation_rate=0.15,
        crossover_rate=0.85,
        tournament_size=5
    )

    best_solution, best_routes, best_fitness = ga.evolve()

    print("\n✅ Optimization Complete!")
    print(f"   - Best fitness: {best_fitness:.2f}")
    print(f"   - Final best fitness: {ga.best_fitness_history[-1]:.2f}")
    print(f"   - Improvement: {(ga.best_fitness_history[0] - ga.best_fitness_history[-1])/ga.best_fitness_history[0]*100:.1f}%")

    # Analyze routes
    print("\n📊 Route Analysis:")
    total_distance = 0
    total_customers = 0
    feasible_routes = 0

    for i, route in enumerate(best_routes):
        if route.customers:
            total_distance += route.total_distance
            total_customers += len(route.customers)
            feasibility = "✅ Feasible" if route.feasible else "❌ Infeasible"
            load = sum(c.demand for c in route.customers)

            print(f"   Vehicle {route.vehicle_id}: {len(route.customers)} customers, "
                  f"distance={route.total_distance:.1f}, load={load:.1f}/{problem.vehicles[i].capacity}, "
                  f"time={route.total_time:.1f} {feasibility}")
            if route.feasible:
                feasible_routes += 1

    print(f"\n📈 Summary:")
    print(f"   - Total distance: {total_distance:.2f}")
    print(f"   - Customers served: {total_customers}/{len(problem.customers)}")
    print(f"   - Feasible routes: {feasible_routes}/{len(best_routes)}")

    # Create visualizations
    print("\n🎨 Creating visualizations...")

    # Solution visualization
    fig_solution = ga.visualize_solution(best_routes)
    fig_solution.write_html("vrptw_solution.html")
    print("   - Solution visualization saved as 'vrptw_solution.html'")

    # Fitness evolution plot
    fig_fitness = ga.plot_fitness_evolution()
    fig_fitness.savefig("fitness_evolution.png", dpi=300, bbox_inches='tight')
    print("   - Fitness evolution plot saved as 'fitness_evolution.png'")

    print("\n🎉 Demonstration complete! Open the HTML file to see the interactive solution.")


if __name__ == "__main__":
    main()
