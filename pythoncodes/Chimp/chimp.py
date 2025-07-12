import numpy as np
import time
import matplotlib.pyplot as plt
import csv

# Set a fixed random seed for reproducibility (important for research and paper-like results)
np.random.seed(42)

# Simulation Parameters (Table 4, Section 5.1)
AREA_SIZE = 200  # 200x200 m^2
TRANSMISSION_RANGE = 30  # meters
FC = 2  # Frequency-controlled parameter (not used in ChOA)
MAX_ITERATIONS = [20, 40, 60, 80, 100]
TARGET_NODES = [25, 50, 75, 100, 125, 150]
ANCHOR_NODES = [15, 30, 45, 60, 75, 90]  # Per Table 4
NOISE_FACTOR = 0.01  # Noise in distance estimation (Section 5.2)
POPULATION_SIZE = 50  # Number of chimps (Table 8)
THRESHOLD = 1.5  # Only count TN as localized if error <= 1.5

def initialize_nodes(num_tn, num_an, area_size):
    """Randomly deploy target and anchor nodes in the area (Section 5.1)."""
    target_nodes = np.random.uniform(0, area_size, (num_tn, 2))
    anchor_nodes = np.random.uniform(0, area_size, (num_an, 2))
    return target_nodes, anchor_nodes

def initialize_nodes_all_localizable(num_tn, num_an, area_size, transmission_range):
    """Deploy anchor nodes randomly, then deploy TNs so all are localizable."""
    anchor_nodes = np.random.uniform(0, area_size, (num_an, 2))
    target_nodes = []
    max_attempts = 10000
    for _ in range(num_tn):
        for attempt in range(max_attempts):
            candidate = np.random.uniform(0, area_size, 2)
            dists = np.linalg.norm(anchor_nodes - candidate, axis=1)
            if np.sum(dists <= transmission_range) >= 3:
                target_nodes.append(candidate)
                break
        else:
            raise RuntimeError("Failed to place a localizable TN after many attempts. Try increasing ANs or area size.")
    return np.array(target_nodes), anchor_nodes

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def estimate_distance(actual_dist, noise_factor):
    """Add Gaussian noise to distance estimation (RSSI-based, Section 5.2)."""
    noise = np.random.normal(0, actual_dist * noise_factor)
    return actual_dist + noise

def get_centroid(anchor_nodes):
    """Calculate centroid of anchor nodes within communication range (Section 5.4)."""
    return np.mean(anchor_nodes, axis=0)

def fitness_function(position, anchor_nodes, estimated_distances):
    """Objective function: Mean square distance error (Section 5.5)."""
    errors = []
    for i, anchor in enumerate(anchor_nodes):
        dist = euclidean_distance(position, anchor)
        error = (dist - estimated_distances[i])**2
        errors.append(error)
    return np.mean(errors) if len(errors) >= 3 else float('inf')

def gauss_mouse_map(x, iterations):
    """Gauss/mouse chaotic map (Table 2, ChOA12)."""
    chaotic_values = []
    for _ in range(iterations):
        if x == 0:
            x = 1
        else:
            x = 1 / (x % 1)
        chaotic_values.append(x)
    return chaotic_values

def choa_optimization(target_pos, anchor_nodes, estimated_distances, max_iter, area_size, population_size):
    """Chimp Optimization Algorithm (ChOA) for node localization (Section 2.2)."""
    # Initialize population around centroid
    centroid = get_centroid(anchor_nodes)
    min_val, max_val = 0, area_size
    population = np.array([centroid] * population_size) + 10 * (np.random.rand(population_size, 2) - 0.5)
    population = np.clip(population, min_val, max_val)
    fitness = np.array([fitness_function(p, anchor_nodes, estimated_distances) for p in population])
    
    # Initialize best solutions for attacker, barrier, chaser, driver
    sorted_indices = np.argsort(fitness)
    attacker = population[sorted_indices[0]].copy()
    barrier = population[sorted_indices[1]].copy() if population_size > 1 else attacker
    chaser = population[sorted_indices[2]].copy() if population_size > 2 else attacker
    driver = population[sorted_indices[3]].copy() if population_size > 3 else attacker
    best_fitness = fitness[sorted_indices[0]]
    
    # Initialize chaotic map (Gauss/mouse map for ChOA12)
    chaotic_value = np.random.uniform(0, 1)
    chaotic_values = gauss_mouse_map(chaotic_value, max_iter)
    
    t = 0
    while t < max_iter:
        # Dynamic coefficient f for four groups (ChOA1, Table 1)
        f_values = [
            1.95 - 2 * (t ** (1/4)) / (max_iter ** (1/3)),  # Group 1
            1.95 - 2 * (t ** (1/3)) / (max_iter ** (1/4)),  # Group 2
            (-3 * (t ** 3) / (max_iter ** 3)) + 1.5,      # Group 3
            (-2 * (t ** 3) / (max_iter ** 3)) + 1.5       # Group 4
        ]
        
        group_size = population_size // 4
        new_population = np.zeros_like(population)
        
        for i in range(population_size):
            # Assign chimp to one of the four groups
            group_idx = min(i // group_size, 3)
            f = f_values[group_idx]
            
            # Calculate coefficients (Equations 3, 4, 5)
            r1 = np.random.rand(2)
            r2 = np.random.rand(2)
            a = 2 * f * r1 - f  # Equation (3)
            c = 2 * r2         # Equation (4)
            m = chaotic_values[t]  # Equation (5)
            
            # Select one of the best solutions (attacker, barrier, chaser, driver)
            best_positions = [attacker, barrier, chaser, driver]
            best_idx = np.random.randint(0, 4)
            x_prey = best_positions[best_idx]
            
            # Driving and chasing (Equations 1 and 2)
            d = np.abs(c * x_prey - m * population[i])  # Equation (1)
            new_pos = x_prey - a * d                    # Equation (2)
            new_pos = np.clip(new_pos, min_val, max_val)
            
            # Evaluate new position
            new_fitness = fitness_function(new_pos, anchor_nodes, estimated_distances)
            
            # Update population if better
            if new_fitness < fitness[i]:
                new_population[i] = new_pos
                fitness[i] = new_fitness
            else:
                new_population[i] = population[i]
            
            # Update best solutions
            if new_fitness < best_fitness:
                attacker = new_pos.copy()
                best_fitness = new_fitness
        
        # Attacking phase (Equations 6, 7, 8)
        for i in range(population_size):
            group_idx = min(i // group_size, 3)
            f = f_values[group_idx]
            r1 = np.random.rand(2)
            r2 = np.random.rand(2)
            a1, a2, a3, a4 = [2 * f * np.random.rand(2) - f for _ in range(4)]
            c1, c2, c3, c4 = [2 * np.random.rand(2) for _ in range(4)]
            m1, m2, m3, m4 = [chaotic_values[t] for _ in range(4)]
            
            d_attacker = np.abs(c1 * attacker - m1 * new_population[i])  # Equation (6)
            d_barrier = np.abs(c2 * barrier - m2 * new_population[i])
            d_chaser = np.abs(c3 * chaser - m3 * new_population[i])
            d_driver = np.abs(c4 * driver - m4 * new_population[i])
            
            x1 = attacker - a1 * d_attacker  # Equation (7)
            x2 = barrier - a2 * d_barrier
            x3 = chaser - a3 * d_chaser
            x4 = driver - a4 * d_driver
            
            new_pos = (x1 + x2 + x3 + x4) / 4  # Equation (8)
            new_pos = np.clip(new_pos, min_val, max_val)
            
            new_fitness = fitness_function(new_pos, anchor_nodes, estimated_distances)
            
            if new_fitness < fitness[i]:
                new_population[i] = new_pos
                fitness[i] = new_fitness
                
                if new_fitness < best_fitness:
                    attacker = new_pos.copy()
                    best_fitness = new_fitness
        
        # Update best solutions for barrier, chaser, driver
        sorted_indices = np.argsort(fitness)
        barrier = new_population[sorted_indices[1]].copy() if population_size > 1 else attacker
        chaser = new_population[sorted_indices[2]].copy() if population_size > 2 else attacker
        driver = new_population[sorted_indices[3]].copy() if population_size > 3 else attacker
        
        population = new_population
        t += 1
    
    return attacker, best_fitness

def node_localization(num_tn, num_an, max_iter, area_size, noise_factor, transmission_range, population_size, fc, collect_positions=False):
    """Perform node localization and collect positions for plotting (Section 5)."""
    target_nodes, anchor_nodes = initialize_nodes_all_localizable(
        num_tn, num_an, area_size, transmission_range
    )
    localized_nodes = 0
    total_error = 0
    start_time = time.time()
    
    if collect_positions:
        unlocalized_positions = []
        actual_positions = []  # Localized target nodes
        est_positions = []  # Estimated positions
    
    for tn in target_nodes:
        distances = [euclidean_distance(tn, an) for an in anchor_nodes]
        valid_anchors = [(an, dist) for an, dist in zip(anchor_nodes, distances) if dist <= transmission_range]
        
        if len(valid_anchors) >= 3:  # Localizable node (Section 5.3)
            anchors, actual_dists = zip(*valid_anchors)
            anchors = np.array(anchors)
            estimated_dists = [estimate_distance(d, noise_factor) for d in actual_dists]
            
            estimated_pos, _ = choa_optimization(tn, anchors, estimated_dists, max_iter, area_size, population_size)
            
            error = euclidean_distance(tn, estimated_pos)
            if error <= THRESHOLD:
                total_error += error
                localized_nodes += 1
                
                if collect_positions:
                    actual_positions.append(tn)
                    est_positions.append(estimated_pos)
        elif collect_positions:
            unlocalized_positions.append(tn)
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    avg_error = total_error / localized_nodes if localized_nodes > 0 else float('inf')
    
    positions = (unlocalized_positions, actual_positions, est_positions, anchor_nodes) if collect_positions else None
    
    return avg_error, computation_time, localized_nodes, positions

def plot_average_deployment(positions_list, area_size, tn, an, anl_a):
    """Plot deployment with unlocalized TN and ANL_A localized nodes (Table 6)."""
    # Use unlocalized TN and anchor nodes from the first iteration
    unlocalized_positions = positions_list[0][0]  # unlocalized TN from first iteration
    anchor_nodes = positions_list[0][3]  # anchors from first iteration
    
    # Aggregate localized nodes across all iterations
    actual_positions_all = []
    est_positions_all = []
    for _, actual_pos, est_pos, _ in positions_list:
        actual_positions_all.extend(actual_pos)
        est_positions_all.extend(est_pos)
    
    # Select ANL_A localized nodes
    if len(actual_positions_all) > anl_a:
        indices = np.random.choice(len(actual_positions_all), size=anl_a, replace=False)
        actual_positions_all = [actual_positions_all[i] for i in indices]
        est_positions_all = [est_positions_all[i] for i in indices]
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(anchor_nodes[:, 0], anchor_nodes[:, 1], c='blue', marker='^', label=f'Anchor Nodes ({an})')
    if unlocalized_positions:
        plt.scatter([p[0] for p in unlocalized_positions], [p[1] for p in unlocalized_positions], 
                    c='red', marker='o', label=f'Unlocalized Target Nodes ({len(unlocalized_positions)})')
    if actual_positions_all:
        plt.scatter([p[0] for p in est_positions_all], [p[1] for p in est_positions_all], 
                    c='green', marker='x', label=f'Localized Nodes ({anl_a})')
        for i in range(len(actual_positions_all)):
            plt.plot([actual_positions_all[i][0], est_positions_all[i][0]], 
                     [actual_positions_all[i][1], est_positions_all[i][1]], 'k--', alpha=0.3)
    plt.xlim(0, area_size)
    plt.ylim(0, area_size)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'Average Node Deployment (TN={tn}, AN={an}, ANL_A={anl_a}, Table 6)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'average_deployment_TN{tn}_AN{an}.png')
    plt.close()

def plot_bar_metrics(table6_data, metric, ylabel, title, filename):
    """Plot bar charts for ANL_E or ANL_T (Figures 9 and 10)."""
    labels = [f'TN={r["TN"]}, AN={r["AN"]}' for r in table6_data]
    values = [r[metric] for r in table6_data]
    
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color='skyblue', edgecolor='black')
    plt.xlabel('Test Case (TN, AN)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_table5_to_csv(results):
    """Save Table 5 results to a CSV file."""
    with open('table5_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['TN', 'AN', 'Iterations', 'NL_E (m)', 'NL_T (s)', 'NL_A']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'TN': r['TN'],
                'AN': r['AN'],
                'Iterations': r['Iterations'],
                'NL_E (m)': r['NL_E'],
                'NL_T (s)': r['NL_T'],
                'NL_A': r['NL_A']
            })

def save_table6_to_csv(table6_data):
    """Save Table 6 results to a CSV file."""
    with open('table6_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['TN', 'AN', 'ANL_E (m)', 'ANL_T (s)', 'ANL_A']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in table6_data:
            writer.writerow({
                'TN': r['TN'],
                'AN': r['AN'],
                'ANL_E (m)': r['ANL_E'],
                'ANL_T (s)': r['ANL_T'],
                'ANL_A': r['ANL_A']
            })

def main():
    """Generate results for Tables 5 and 6, save to CSV, six deployment figures, and bar graphs."""
    results = []
    table6_data = []
    positions_data = { (tn, an): [] for tn, an in zip(TARGET_NODES, ANCHOR_NODES) }
    
    for tn, an in zip(TARGET_NODES, ANCHOR_NODES):
        for max_iter in MAX_ITERATIONS:
            avg_error, comp_time, localized, positions = node_localization(
                tn, an, max_iter, AREA_SIZE, NOISE_FACTOR, TRANSMISSION_RANGE, POPULATION_SIZE, FC, 
                collect_positions=True
            )
            results.append({
                'TN': tn,
                'AN': an,
                'Iterations': max_iter,
                'NL_E': round(avg_error, 3),
                'NL_T': round(comp_time, 2),
                'NL_A': localized
            })
            
            if positions is not None:
                positions_data[(tn, an)].append(positions)
        
        tn_results = [r for r in results if r['TN'] == tn and r['AN'] == an]
        if tn_results:
            avg_nle = np.mean([r['NL_E'] for r in tn_results])
            avg_nlt = np.mean([r['NL_T'] for r in tn_results])
            avg_nla = round(np.mean([r['NL_A'] for r in tn_results]))
            table6_data.append({
                'TN': tn,
                'AN': an,
                'ANL_E': round(avg_nle, 3),
                'ANL_T': round(avg_nlt, 2),
                'ANL_A': avg_nla
            })
    
    # Print Table 5
    print("\nTable 5: Performance Metrics of ChOA")
    print("| TN | AN | Iterations | NL_E (m) | NL_T (s) | NL_A |")
    print("|----|----|------------|----------|----------|------|")
    for r in results:
        print(f"| {r['TN']} | {r['AN']} | {r['Iterations']} | {r['NL_E']} | {r['NL_T']} | {r['NL_A']} |")
    
    # Print Table 6
    print("\nTable 6: Result Summary of ChOA")
    print("| TN | AN | ANL_E (m) | ANL_T (s) | ANL_A |")
    print("|----|----|-----------|-----------|-------|")
    for r in table6_data:
        print(f"| {r['TN']} | {r['AN']} | {r['ANL_E']} | {r['ANL_T']} | {r['ANL_A']} |")
    
    # Save results to CSV files
    save_table5_to_csv(results)
    save_table6_to_csv(table6_data)
    
    # Plot bar charts for Figures 9 and 10
    plot_bar_metrics(table6_data, 'ANL_E', 'Average Localization Error (m)', 
                     'Average Localization Error Across Test Cases (Figure 9)', 'figure9_anl_e.png')
    plot_bar_metrics(table6_data, 'ANL_T', 'Average Computation Time (s)', 
                     'Average Computation Time Across Test Cases (Figure 10)', 'figure10_anl_t.png')
    
    # Plot average deployment for all six test cases
    for tn, an in zip(TARGET_NODES, ANCHOR_NODES):
        if positions_data[(tn, an)]:
            anl_a = next((r['ANL_A'] for r in table6_data if r['TN'] == tn and r['AN'] == an), 0)
            plot_average_deployment(positions_data[(tn, an)], AREA_SIZE, tn, an, anl_a)

if __name__ == "__main__":
    main()