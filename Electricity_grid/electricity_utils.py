import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import itertools
import math

def generate_nodes(size=30):
    # Set random seed for reproducibility
    np.random.seed(2)

    #### Generate (X, Y) Coordinates ####
    # Generate random 2D coordinates for each node (X, Y) between 0 and 100
    coordinates = np.random.rand(size, 2) * 100  # Scale coordinates to be between 0 and 100

    #### Calculate Distance Matrix ####
    # Initialize the distance matrix
    distance_matrix = np.zeros((size, size))

    # Calculate Euclidean distance between each pair of nodes based on their coordinates
    for i in range(size):
        for j in range(i + 1, size):
            distance = np.linalg.norm(coordinates[i] - coordinates[j])  # Euclidean distance
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetric matrix

    #### Set some distances to infinity to represent disconnected nodes ####
    num_infinity = int(0.4 * distance_matrix.size)  # 10% of the entries

    # Randomly select indices to set as infinity, excluding the diagonal
    indices = np.tril_indices(size, -1)
    random_indices = np.random.choice(len(indices[0]), size=num_infinity, replace=False)

    for idx in random_indices:
        i, j = indices[0][idx], indices[1][idx]
        distance_matrix[i, j] = np.inf
        distance_matrix[j, i] = np.inf  # Ensure symmetry

    #### Get generator nodes ####
    generators = determine_generators(size)  # Returns (hydro, wind, fossil)
    all_generators = list(itertools.chain.from_iterable(generators))  # Flatten generator list

    #### Remove edges connected to generators ####
    for g in all_generators:
        # Find all nodes connected to the generator
        connected_edges = [(g, j) for j in range(size) if distance_matrix[g, j] != np.inf and g != j]
        num_edges_to_remove = math.floor(len(connected_edges) * 0.9)
        
        # Randomly select edges to remove
        edges_to_remove = np.random.choice(len(connected_edges), size=num_edges_to_remove, replace=False)
        
        # Set the distance for selected edges to infinity
        for idx in edges_to_remove:
            i, j = connected_edges[idx]
            distance_matrix[i, j] = np.inf
            distance_matrix[j, i] = np.inf  # Ensure symmetry

    return distance_matrix, coordinates, generators

def determine_generators(size):
    generator = []
    # Hydropump is assigned to suburb 1
    hydropump = []

    # Wind Turbine is assigned to suburbs 6 and 11
    wind_turbine = []

    # Fossil Fuel is assigned to suburbs 9 and 19
    fossil_fuel = []

    while True:
        idx = np.random.randint(0, size)

        if idx not in generator:
            if len(hydropump) != 1:
                hydropump.append(idx)
            elif len(wind_turbine) != 1:
                wind_turbine.append(idx)
            elif len(fossil_fuel) != 1:
                fossil_fuel.append(idx)
            else:
                break
        
    generators = [hydropump, wind_turbine, fossil_fuel]
    
    return generators

def generate_demand(size):
    # Generate a random 20x20 distance matrix with values between 0 and 100
    demand = np.random.randint(30, 100, size=size)

    return demand

def plot_electric_grid(coordinates, distance_matrix, generators, X, W):
    plt.figure(figsize=(12, 12))
    
    # Plot nodes
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', s=100, label='Nodes')
    
    # Highlight generator nodes with red color
    generator_nodes = list(itertools.chain.from_iterable(generators))
    plt.scatter(coordinates[generator_nodes, 0], coordinates[generator_nodes, 1], c='red', s=100, label='Generators')

    # Annotate nodes with their indices
    for i, (x, y) in enumerate(coordinates):
        plt.text(x, y, str(i), fontsize=12, ha='right', va='bottom')

    # Plot edges with electricity flow
    for (i, j), flow in W.items():
        if flow.X > 0:  # Only plot edges with positive flow
            plt.plot([coordinates[i, 0], coordinates[j, 0]], [coordinates[i, 1], coordinates[j, 1]], 'k-', lw=0.5)

            # Add text showing the flow of electricity on the edge
            mid_x, mid_y = (coordinates[i, 0] + coordinates[j, 0]) / 2, (coordinates[i, 1] + coordinates[j, 1]) / 2
            plt.text(mid_x, mid_y, f'{flow.X:.2f}', color='green', fontsize=10, ha='center', va='center')

    # Annotate generator nodes with the amount of electricity they generate
    for g in generator_nodes:
        plt.text(coordinates[g, 0], coordinates[g, 1], f'Gen: {X[g].X:.2f}', color='red', fontsize=12, ha='left', va='top')

    plt.title("Electric Grid - Generation and Flow")
    plt.legend()
    plt.show()