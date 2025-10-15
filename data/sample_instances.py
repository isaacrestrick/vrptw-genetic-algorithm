"""
Sample VRPTW problem instances for testing and benchmarking.
"""

import numpy as np
from src.vrptw import Customer, Vehicle, VRPTWProblem


def create_small_instance():
    """Create a small VRPTW instance with 8 customers and 2 vehicles."""
    depot = Customer(id=0, x=25, y=25, demand=0, ready_time=0, due_time=240, service_time=0)

    customers = [
        Customer(id=1, x=20, y=20, demand=5, ready_time=30, due_time=80, service_time=10),
        Customer(id=2, x=30, y=20, demand=4, ready_time=40, due_time=90, service_time=8),
        Customer(id=3, x=20, y=30, demand=3, ready_time=50, due_time=100, service_time=6),
        Customer(id=4, x=30, y=30, demand=6, ready_time=60, due_time=110, service_time=12),
        Customer(id=5, x=15, y=25, demand=4, ready_time=70, due_time=120, service_time=8),
        Customer(id=6, x=35, y=25, demand=5, ready_time=80, due_time=130, service_time=10),
        Customer(id=7, x=25, y=15, demand=3, ready_time=90, due_time=140, service_time=6),
        Customer(id=8, x=25, y=35, demand=4, ready_time=100, due_time=150, service_time=8),
    ]

    vehicles = [
        Vehicle(id=1, capacity=15, depot_return_time=240),
        Vehicle(id=2, capacity=12, depot_return_time=240),
    ]

    return VRPTWProblem(customers, vehicles, depot)


def create_medium_instance():
    """Create a medium VRPTW instance with 15 customers and 3 vehicles."""
    depot = Customer(id=0, x=50, y=50, demand=0, ready_time=0, due_time=480, service_time=0)

    # Generate customers in clusters
    customers = []
    np.random.seed(123)

    # Morning cluster
    for i in range(1, 6):
        x = 30 + np.random.uniform(-10, 10)
        y = 30 + np.random.uniform(-10, 10)
        customers.append(Customer(
            id=i, x=x, y=y, demand=np.random.uniform(2, 6),
            ready_time=np.random.uniform(60, 120),
            due_time=np.random.uniform(120, 180),
            service_time=np.random.uniform(5, 15)
        ))

    # Afternoon cluster
    for i in range(6, 11):
        x = 70 + np.random.uniform(-10, 10)
        y = 70 + np.random.uniform(-10, 10)
        customers.append(Customer(
            id=i, x=x, y=y, demand=np.random.uniform(2, 6),
            ready_time=np.random.uniform(180, 240),
            due_time=np.random.uniform(240, 300),
            service_time=np.random.uniform(5, 15)
        ))

    # Evening cluster
    for i in range(11, 16):
        x = 20 + np.random.uniform(-8, 8)
        y = 80 + np.random.uniform(-8, 8)
        customers.append(Customer(
            id=i, x=x, y=y, demand=np.random.uniform(2, 6),
            ready_time=np.random.uniform(300, 360),
            due_time=np.random.uniform(360, 420),
            service_time=np.random.uniform(5, 15)
        ))

    vehicles = [
        Vehicle(id=1, capacity=18, depot_return_time=480),
        Vehicle(id=2, capacity=20, depot_return_time=480),
        Vehicle(id=3, capacity=16, depot_return_time=480),
    ]

    return VRPTWProblem(customers, vehicles, depot)


def create_clustered_instance():
    """Create a clustered VRPTW instance that encourages efficient routing."""
    depot = Customer(id=0, x=0, y=0, demand=0, ready_time=0, due_time=400, service_time=0)

    customers = []
    np.random.seed(456)

    # Four clusters around the depot
    cluster_centers = [(20, 20), (-20, 20), (20, -20), (-20, -20)]
    cluster_times = [
        (50, 120),   # Morning cluster 1
        (80, 150),   # Morning cluster 2
        (200, 270),  # Afternoon cluster 1
        (230, 300),  # Afternoon cluster 2
    ]

    customer_id = 1
    for center_idx, (cx, cy) in enumerate(cluster_centers):
        ready_time, due_time = cluster_times[center_idx]

        # 4 customers per cluster
        for _ in range(4):
            x = cx + np.random.uniform(-8, 8)
            y = cy + np.random.uniform(-8, 8)
            customers.append(Customer(
                id=customer_id, x=x, y=y,
                demand=np.random.uniform(2, 5),
                ready_time=np.random.uniform(ready_time, ready_time + 20),
                due_time=np.random.uniform(due_time - 20, due_time),
                service_time=np.random.uniform(3, 10)
            ))
            customer_id += 1

    vehicles = [
        Vehicle(id=1, capacity=22, depot_return_time=400),
        Vehicle(id=2, capacity=20, depot_return_time=400),
        Vehicle(id=3, capacity=18, depot_return_time=400),
    ]

    return VRPTWProblem(customers, vehicles, depot)


def create_random_instance(n_customers=25, n_vehicles=4, seed=42):
    """Create a random VRPTW instance."""
    np.random.seed(seed)

    depot = Customer(id=0, x=0, y=0, demand=0, ready_time=0, due_time=480, service_time=0)

    customers = []
    for i in range(1, n_customers + 1):
        # Random position in [-50, 50] x [-50, 50]
        x = np.random.uniform(-50, 50)
        y = np.random.uniform(-50, 50)

        # Random time window
        ready_time = np.random.uniform(10, 400)
        window_length = np.random.uniform(30, 90)
        due_time = ready_time + window_length

        # Random demand and service time
        demand = np.random.uniform(1, 8)
        service_time = np.random.uniform(2, 15)

        customers.append(Customer(
            id=i, x=x, y=y, demand=demand,
            ready_time=ready_time, due_time=due_time, service_time=service_time
        ))

    # Create vehicles with appropriate capacities
    total_demand = sum(c.demand for c in customers)
    avg_capacity = total_demand / n_vehicles * 1.5  # Some buffer

    vehicles = []
    for i in range(1, n_vehicles + 1):
        capacity = np.random.uniform(avg_capacity * 0.8, avg_capacity * 1.2)
        vehicles.append(Vehicle(id=i, capacity=capacity, depot_return_time=480))

    return VRPTWProblem(customers, vehicles, depot)


# Dictionary of available instances
INSTANCES = {
    'small': create_small_instance,
    'medium': create_medium_instance,
    'clustered': create_clustered_instance,
    'random_25': lambda: create_random_instance(25, 4, 42),
    'random_50': lambda: create_random_instance(50, 5, 123),
}


def get_instance(name):
    """Get a problem instance by name."""
    if name not in INSTANCES:
        available = ', '.join(INSTANCES.keys())
        raise ValueError(f"Unknown instance '{name}'. Available: {available}")

    return INSTANCES[name]()


if __name__ == "__main__":
    # Quick test of all instances
    for name in INSTANCES.keys():
        try:
            instance = get_instance(name)
            print(f"✅ {name}: {len(instance.customers)} customers, {len(instance.vehicles)} vehicles")
        except Exception as e:
            print(f"❌ {name}: {e}")
