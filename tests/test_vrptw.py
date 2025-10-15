"""
Unit tests for VRPTW Genetic Algorithm implementation.
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.vrptw import Customer, Vehicle, VRPTWProblem, VRPTWGeneticAlgorithm


class TestVRPTW(unittest.TestCase):
    """Test cases for VRPTW implementation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create simple test problem
        depot = Customer(id=0, x=0, y=0, demand=0, ready_time=0, due_time=100, service_time=0)

        customers = [
            Customer(id=1, x=1, y=1, demand=5, ready_time=10, due_time=50, service_time=5),
            Customer(id=2, x=-1, y=1, demand=3, ready_time=20, due_time=60, service_time=3),
            Customer(id=3, x=1, y=-1, demand=4, ready_time=30, due_time=70, service_time=4),
        ]

        vehicles = [
            Vehicle(id=1, capacity=10),
            Vehicle(id=2, capacity=8),
        ]

        self.problem = VRPTWProblem(customers, vehicles, depot)
        self.ga = VRPTWGeneticAlgorithm(self.problem, population_size=20, generations=10)

    def test_problem_creation(self):
        """Test that problem is created correctly."""
        self.assertEqual(len(self.problem.customers), 3)
        self.assertEqual(len(self.problem.vehicles), 2)
        self.assertEqual(self.problem.depot.id, 0)

        # Check distance matrix dimensions
        self.assertEqual(self.problem.distance_matrix.shape, (4, 4))  # 3 customers + 1 depot

    def test_customer_index_mapping(self):
        """Test customer index mapping."""
        self.assertEqual(self.problem.get_customer_index(self.problem.depot), 0)
        self.assertEqual(self.problem.get_customer_index(self.problem.customers[0]), 1)
        self.assertEqual(self.problem.get_customer_index(self.problem.customers[1]), 2)

    def test_chromosome_creation(self):
        """Test chromosome creation."""
        chromosome = self.ga.create_chromosome()
        self.assertEqual(len(chromosome), 2)  # 2 vehicles

        # Check that all customers are assigned
        all_assigned = []
        for route in chromosome:
            all_assigned.extend(route)

        self.assertEqual(set(all_assigned), {1, 2, 3})

    def test_fitness_evaluation(self):
        """Test fitness evaluation."""
        chromosome = [[1, 2], [3]]  # Customer 1,2 to vehicle 1; customer 3 to vehicle 2

        fitness, routes = self.ga.evaluate_fitness(chromosome)

        self.assertIsInstance(fitness, float)
        self.assertGreaterEqual(fitness, 0)
        self.assertEqual(len(routes), 2)

        # Check routes
        self.assertEqual(len(routes[0].customers), 2)  # Vehicle 1 has 2 customers
        self.assertEqual(len(routes[1].customers), 1)  # Vehicle 2 has 1 customer

    def test_route_optimization(self):
        """Test route optimization heuristic."""
        customers = self.problem.customers.copy()
        optimized = self.ga._optimize_route_order(customers)

        self.assertEqual(len(optimized), len(customers))
        self.assertEqual(set(c.id for c in optimized), set(c.id for c in customers))

    def test_crossover(self):
        """Test crossover operation."""
        parent1 = [[1, 2], [3]]
        parent2 = [[1], [2, 3]]

        child1, child2 = self.ga.crossover(parent1, parent2)

        # Children should have same structure
        self.assertEqual(len(child1), 2)
        self.assertEqual(len(child2), 2)

        # All customers should be present in children
        for child in [child1, child2]:
            all_customers = []
            for route in child:
                all_customers.extend(route)
            self.assertEqual(set(all_customers), {1, 2, 3})

    def test_mutation(self):
        """Test mutation operation."""
        chromosome = [[1, 2], [3]]
        mutated = self.ga.mutate(chromosome)

        # Structure should be preserved
        self.assertEqual(len(mutated), 2)

        # All customers should still be present
        all_customers = []
        for route in mutated:
            all_customers.extend(route)
        self.assertEqual(set(all_customers), {1, 2, 3})

    def test_feasible_solution(self):
        """Test with a feasible solution."""
        # Create a feasible chromosome
        chromosome = [[1], [2, 3]]  # Balanced assignment

        fitness, routes = self.ga.evaluate_fitness(chromosome)

        # Check that capacities are not exceeded
        for i, route in enumerate(routes):
            load = sum(c.demand for c in route.customers)
            self.assertLessEqual(load, self.problem.vehicles[i].capacity)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""

    def test_empty_routes(self):
        """Test handling of empty routes."""
        depot = Customer(id=0, x=0, y=0, demand=0, ready_time=0, due_time=100, service_time=0)
        customers = [Customer(id=1, x=1, y=1, demand=1, ready_time=10, due_time=50, service_time=1)]
        vehicles = [Vehicle(id=1, capacity=10), Vehicle(id=2, capacity=10)]

        problem = VRPTWProblem(customers, vehicles, depot)
        ga = VRPTWGeneticAlgorithm(problem, population_size=10, generations=5)

        # Test chromosome with empty route
        chromosome = [[1], []]  # One vehicle has customer, other is empty
        fitness, routes = ga.evaluate_fitness(chromosome)

        self.assertIsInstance(fitness, float)
        self.assertEqual(len(routes), 2)
        self.assertEqual(len(routes[0].customers), 1)  # First route has customer
        self.assertEqual(len(routes[1].customers), 0)  # Second route is empty

    def test_single_customer(self):
        """Test problem with single customer."""
        depot = Customer(id=0, x=0, y=0, demand=0, ready_time=0, due_time=100, service_time=0)
        customers = [Customer(id=1, x=1, y=1, demand=1, ready_time=10, due_time=50, service_time=1)]
        vehicles = [Vehicle(id=1, capacity=10)]

        problem = VRPTWProblem(customers, vehicles, depot)
        ga = VRPTWGeneticAlgorithm(problem, population_size=10, generations=5)

        chromosome = [[1]]
        fitness, routes = ga.evaluate_fitness(chromosome)

        self.assertIsInstance(fitness, float)
        self.assertEqual(len(routes), 1)
        self.assertEqual(len(routes[0].customers), 1)


if __name__ == '__main__':
    unittest.main()
