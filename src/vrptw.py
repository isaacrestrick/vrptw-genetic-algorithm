"""
Vehicle Routing Problem with Time Windows (VRPTW) Solver using Genetic Algorithm

This module implements a genetic algorithm to solve the VRPTW, where vehicles must
visit customers within their specified time windows while minimizing total distance.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import plotly.graph_objects as go


@dataclass
class Customer:
    """Represents a customer with location, demand, and time window constraints."""
    id: int
    x: float
    y: float
    demand: float
    ready_time: float  # Earliest service time
    due_time: float    # Latest service time
    service_time: float  # Time required to serve this customer

    @property
    def location(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Vehicle:
    """Represents a vehicle with capacity and depot return time."""
    id: int
    capacity: float
    depot_return_time: float = 0.0


@dataclass
class Route:
    """Represents a single vehicle route."""
    vehicle_id: int
    customers: List[Customer]
    total_distance: float = 0.0
    total_time: float = 0.0
    feasible: bool = True


class VRPTWProblem:
    """Vehicle Routing Problem with Time Windows definition."""

    def __init__(self, customers: List[Customer], vehicles: List[Vehicle],
                 depot: Customer, distance_matrix: Optional[np.ndarray] = None,
                 time_matrix: Optional[np.ndarray] = None):
        self.customers = customers
        self.vehicles = vehicles
        self.depot = depot
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix

        if distance_matrix is None or time_matrix is None:
            self._build_matrices()

    def _build_matrices(self):
        """Build distance and time matrices from customer locations."""
        n = len(self.customers) + 1  # +1 for depot
        self.distance_matrix = np.zeros((n, n))
        self.time_matrix = np.zeros((n, n))

        all_points = [self.depot] + self.customers

        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = all_points[i].x - all_points[j].x
                    dy = all_points[i].y - all_points[j].y
                    distance = np.sqrt(dx*dx + dy*dy)
                    self.distance_matrix[i, j] = distance
                    # Assume travel time equals distance for simplicity
                    self.time_matrix[i, j] = distance

    def get_customer_index(self, customer: Customer) -> int:
        """Get the index of a customer in the distance/time matrices."""
        if customer.id == self.depot.id:
            return 0
        return [c.id for c in self.customers].index(customer.id) + 1


class VRPTWGeneticAlgorithm:
    """Genetic Algorithm solver for VRPTW."""

    def __init__(self, problem: VRPTWProblem, population_size: int = 100,
                 generations: int = 200, mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8, tournament_size: int = 5):
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size

        # Fitness tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []

    def create_chromosome(self) -> List[List[int]]:
        """
        Create a chromosome representing vehicle routes.
        Each sublist represents customers assigned to one vehicle.
        """
        # Random assignment of customers to vehicles
        customer_ids = [c.id for c in self.problem.customers]
        random.shuffle(customer_ids)

        routes = [[] for _ in self.problem.vehicles]

        for customer_id in customer_ids:
            # Find vehicle with most remaining capacity
            vehicle_idx = random.randint(0, len(self.problem.vehicles) - 1)
            routes[vehicle_idx].append(customer_id)

        return routes

    def evaluate_fitness(self, chromosome: List[List[int]]) -> Tuple[float, List[Route]]:
        """
        Evaluate the fitness of a chromosome.
        Returns (fitness_score, routes) where lower fitness is better.
        """
        routes = []
        total_penalty = 0.0
        total_distance = 0.0

        for vehicle_idx, customer_ids in enumerate(chromosome):
            vehicle = self.problem.vehicles[vehicle_idx]
            route_customers = [c for c in self.problem.customers if c.id in customer_ids]

            if not route_customers:
                # Empty route
                route = Route(vehicle.id, [], 0.0, 0.0, True)
                routes.append(route)
                continue

            # Sort customers in a reasonable order (nearest neighbor heuristic)
            route_customers = self._optimize_route_order(route_customers)

            # Calculate route metrics
            route_distance, route_time, penalty, feasible = self._calculate_route_metrics(
                route_customers, vehicle
            )

            route = Route(vehicle.id, route_customers, route_distance, route_time, feasible)
            routes.append(route)

            total_distance += route_distance
            total_penalty += penalty

        # Fitness is total distance + penalties for violations
        fitness = total_distance + total_penalty * 1000  # Heavy penalty for violations

        return fitness, routes

    def _optimize_route_order(self, customers: List[Customer]) -> List[Customer]:
        """Use nearest neighbor heuristic to order customers in a route."""
        if not customers:
            return customers

        ordered = [customers[0]]
        remaining = customers[1:]

        while remaining:
            current = ordered[-1]
            current_idx = self.problem.get_customer_index(current)

            # Find nearest remaining customer
            nearest_idx = None
            nearest_distance = float('inf')

            for cust in remaining:
                cust_idx = self.problem.get_customer_index(cust)
                distance = self.problem.distance_matrix[current_idx, cust_idx]
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_idx = remaining.index(cust)

            ordered.append(remaining.pop(nearest_idx))

        return ordered

    def _calculate_route_metrics(self, customers: List[Customer], vehicle: Vehicle) \
            -> Tuple[float, float, float, bool]:
        """Calculate distance, time, and penalty for a route."""
        if not customers:
            return 0.0, 0.0, 0.0, True

        distance = 0.0
        time = 0.0
        penalty = 0.0
        feasible = True

        # Start from depot
        current_time = 0.0
        current_load = 0.0

        prev_idx = 0  # depot index

        for customer in customers:
            cust_idx = self.problem.get_customer_index(customer)

            # Travel time from previous location
            travel_time = self.problem.time_matrix[prev_idx, cust_idx]
            arrival_time = current_time + travel_time

            # Time window constraints
            if arrival_time < customer.ready_time:
                # Wait until ready time
                wait_time = customer.ready_time - arrival_time
                service_start = customer.ready_time
            elif arrival_time > customer.due_time:
                # Late arrival - penalty
                penalty += arrival_time - customer.due_time
                feasible = False
                service_start = arrival_time
            else:
                service_start = arrival_time

            # Service time
            service_start += customer.service_time

            # Capacity constraint
            current_load += customer.demand
            if current_load > vehicle.capacity:
                penalty += (current_load - vehicle.capacity) * 10
                feasible = False

            # Update for next customer
            distance += self.problem.distance_matrix[prev_idx, cust_idx]
            current_time = service_start
            prev_idx = cust_idx

        # Return to depot
        depot_idx = 0
        return_travel = self.problem.time_matrix[prev_idx, depot_idx]
        distance += self.problem.distance_matrix[prev_idx, depot_idx]

        # Check depot return time constraint
        final_time = current_time + return_travel
        if final_time > vehicle.depot_return_time and vehicle.depot_return_time > 0:
            penalty += final_time - vehicle.depot_return_time
            feasible = False

        return distance, final_time, penalty, feasible

    def tournament_selection(self, population: List[List[List[int]]],
                           fitnesses: List[float]) -> List[List[int]]:
        """Tournament selection for choosing parents."""
        selected = []

        for _ in range(2):  # Select 2 parents
            tournament = random.sample(list(zip(population, fitnesses)), self.tournament_size)
            winner = min(tournament, key=lambda x: x[1])  # Minimize fitness
            selected.append(winner[0])

        return selected

    def crossover(self, parent1: List[List[int]], parent2: List[List[int]]) \
            -> Tuple[List[List[int]], List[List[int]]]:
        """Order crossover for vehicle routing."""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        child1, child2 = [], []

        for route_idx in range(len(self.problem.vehicles)):
            route1 = parent1[route_idx]
            route2 = parent2[route_idx]

            if len(route1) <= 1 or len(route2) <= 1:
                child1.append(route1.copy())
                child2.append(route2.copy())
                continue

            # Order crossover
            start = random.randint(0, min(len(route1), len(route2)) - 1)
            end = random.randint(start + 1, min(len(route1), len(route2)))

            # Child 1 gets segment from parent 1
            child1_segment = route1[start:end]
            child2_segment = route2[start:end]

            # Fill remaining positions with customers from other parent
            remaining1 = [c for c in route2 if c not in child1_segment]
            remaining2 = [c for c in route1 if c not in child2_segment]

            child1_route = remaining1[:start] + child1_segment + remaining1[start:]
            child2_route = remaining2[:start] + child2_segment + remaining2[start:]

            child1.append(child1_route)
            child2.append(child2_route)

        return child1, child2

    def mutate(self, chromosome: List[List[int]]) -> List[List[int]]:
        """Mutation operations for the chromosome."""
        if random.random() > self.mutation_rate:
            return chromosome

        mutated = [route.copy() for route in chromosome]

        # Choose mutation type
        mutation_type = random.choice(['swap', 'move', 'reverse'])

        if mutation_type == 'swap':
            # Swap two customers between routes
            route1_idx = random.randint(0, len(mutated) - 1)
            route2_idx = random.randint(0, len(mutated) - 1)

            if route1_idx != route2_idx and mutated[route1_idx] and mutated[route2_idx]:
                pos1 = random.randint(0, len(mutated[route1_idx]) - 1)
                pos2 = random.randint(0, len(mutated[route2_idx]) - 1)

                mutated[route1_idx][pos1], mutated[route2_idx][pos2] = \
                    mutated[route2_idx][pos2], mutated[route1_idx][pos1]

        elif mutation_type == 'move':
            # Move a customer to a different route
            from_route = random.randint(0, len(mutated) - 1)
            if mutated[from_route]:
                to_route = random.randint(0, len(mutated) - 1)
                customer_idx = random.randint(0, len(mutated[from_route]) - 1)
                customer = mutated[from_route].pop(customer_idx)
                insert_pos = random.randint(0, len(mutated[to_route]))
                mutated[to_route].insert(insert_pos, customer)

        elif mutation_type == 'reverse':
            # Reverse a segment within a route
            route_idx = random.randint(0, len(mutated) - 1)
            if len(mutated[route_idx]) > 2:
                start = random.randint(0, len(mutated[route_idx]) - 2)
                end = random.randint(start + 1, len(mutated[route_idx]))
                mutated[route_idx][start:end] = reversed(mutated[route_idx][start:end])

        return mutated

    def evolve(self) -> Tuple[List[List[int]], List[Route], float]:
        """Run the genetic algorithm and return the best solution."""
        # Initialize population
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

            # Track fitness history
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(fitnesses))

            # Create new population
            new_population = []

            # Elitism: keep best solution
            elite_idx = np.argmin(fitnesses)
            new_population.append(population[elite_idx])

            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1, parent2 = self.tournament_selection(population, fitnesses)

                # Crossover
                child1, child2 = self.crossover(parent1, parent2)

                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

            if generation % 20 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.2f}")

        return best_chromosome, best_routes, best_fitness

    def visualize_solution(self, routes: List[Route], save_path: Optional[str] = None):
        """Create a beautiful visualization of the solution."""
        fig = go.Figure()

        # Colors for different vehicles
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

        # Plot depot
        fig.add_trace(go.Scatter(
            x=[self.problem.depot.x], y=[self.problem.depot.y],
            mode='markers+text',
            marker=dict(size=20, color='black', symbol='star'),
            text=['Depot'],
            textposition="top center",
            name='Depot',
            showlegend=True
        ))

        # Plot routes
        for i, route in enumerate(routes):
            if not route.customers:
                continue

            color = colors[i % len(colors)]

            # Route points
            x_coords = [self.problem.depot.x] + [c.x for c in route.customers] + [self.problem.depot.x]
            y_coords = [self.problem.depot.y] + [c.y for c in route.customers] + [self.problem.depot.y]

            # Add route line
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                mode='lines+markers',
                line=dict(color=color, width=3),
                marker=dict(size=10, color=color),
                name=f'Vehicle {route.vehicle_id}',
                text=[f'Depot'] + [f'C{c.id}' for c in route.customers] + [f'Depot'],
                hovertemplate='%{text}<br>Time: %{customdata}',
                customdata=[0] + [f'{c.ready_time}-{c.due_time}' for c in route.customers] + [0]
            ))

            # Add customer markers with time windows
            for j, customer in enumerate(route.customers):
                fig.add_trace(go.Scatter(
                    x=[customer.x], y=[customer.y],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=color,
                        symbol='circle',
                        line=dict(width=2, color='white')
                    ),
                    text=[f'C{customer.id}<br>Time: {customer.ready_time:.1f}-{customer.due_time:.1f}'],
                    hoverinfo='text',
                    showlegend=False
                ))

        # Update layout
        fig.update_layout(
            title='Vehicle Routing Problem with Time Windows - Genetic Algorithm Solution',
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            showlegend=True,
            width=1000,
            height=800,
            template='plotly_white'
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_fitness_evolution(self, save_path: Optional[str] = None):
        """Plot the evolution of fitness over generations."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.best_fitness_history, label='Best Fitness', linewidth=2, color='blue')
        ax.plot(self.avg_fitness_history, label='Average Fitness', linewidth=2, color='red', alpha=0.7)

        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness (Total Distance + Penalties)')
        ax.set_title('Genetic Algorithm Fitness Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig
