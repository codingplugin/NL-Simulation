import numpy as np
import time
import matplotlib.pyplot as plt
import csv

# Set a fixed random seed for reproducibility (important for research and paper-like results)
np.random.seed(42)

# Simulation Parameters (Table 4, Section 5.1)
AREA_SIZE = 200  # 200x200 m^2
TRANSMISSION_RANGE = 30  # meters
FC = 2  # Frequency-controlled parameter (kept for compatibility, not used in DBO)
MAX_ITERATIONS = [20, 40, 60, 80, 100]
TARGET_NODES = [25, 50, 75, 100, 125, 150]
ANCHOR_NODES = [15, 30, 45, 60, 75, 90]  # Per Table 4
NOISE_FACTOR = 0.1  # Noise in distance estimation (Section 5.2)
POPULATION_SIZE = 50  # Dung beetle population size (default)

def initialize_nodes(num_tn, num_an, area_size):
    """Randomly deploy target and anchor nodes in the area (Section 5.1)."""
    target_nodes = np.random.uniform(0, area_size, (num_tn, 2))
    anchor_nodes = np.random.uniform(0, area_size, (num_an, 2))
    return target_nodes, anchor_nodes

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

def dbo_optimization(target_pos, anchor_nodes, estimated_distances, max_iter, area_size, population_size, fc):
    """Dung Beetle Optimizer for node localization (Section 2.2, Algorithm 4)."""
    # Initialize parameters from Section 3.5
    k = 0.1  # Deflection coefficient
    b = 0.3  # Light intensity factor
    S = 0.5  # Stealing factor
    Tmax = max_iter
    
    # Initialize population around centroid
    centroid = get_centroid(anchor_nodes)
    population = np.array([centroid] * population_size) + 10 * (np.random.rand(population_size, 2) - 0.5)
    population = np.clip(population, 0, area_size)
    fitness = np.array([fitness_function(p, anchor_nodes, estimated_distances) for p in population])
    
    # Track previous positions for ball-rolling
    prev_population = population.copy()
    
    # Initialize best and worst positions
    best_idx = np.argmin(fitness)
    worst_idx = np.argmax(fitness)
    Xb = population[best_idx].copy()  # Global best position
    Xstar = Xb.copy()  # Local best position (assumed same initially)
    Xw = population[worst_idx].copy()  # Global worst position
    best_fitness = fitness[best_idx]
    
    # Population distribution (scaled from Figure 8: 6, 6, 7, 11 for N=30 to N=50)
    n_ball_rolling = 10  # Ball-rolling dung beetles
    n_brood = 10  # Brood balls
    n_small = 12  # Small dung beetles
    n_thief = 18  # Thieves
    assert n_ball_rolling + n_brood + n_small + n_thief == population_size, "Population sizes must sum to POPULATION_SIZE"
    
    t = 0
    while t < Tmax:
        R = 1 - t / Tmax  # Dynamic parameter (Equations 3, 5)
        
        # Update positions for each group
        for i in range(population_size):
            current_pos = population[i].copy()
            
            if i < n_ball_rolling:  # Ball-rolling dung beetles
                # Simulate obstacle with 10% probability
                if np.random.rand() < 0.1:  # Dancing behavior (Equation 2)
                    theta = np.random.uniform(0, np.pi)
                    if theta not in [0, np.pi/2, np.pi]:  # Avoid no-update cases (Remark 2)
                        delta = np.abs(current_pos - prev_population[i])
                        new_pos = current_pos + np.tan(theta) * delta
                else:  # Ball-rolling behavior (Equation 1)
                    alpha = np.random.choice([1, -1])  # Random deviation (Remark 1)
                    delta_x = np.abs(current_pos - Xw)
                    new_pos = current_pos + alpha * k * prev_population[i] + b * delta_x
            
            elif i < n_ball_rolling + n_brood:  # Brood balls (Equations 3, 4)
                Lb_star = np.maximum(Xstar * (1 - R), 0)
                Ub_star = np.minimum(Xstar * (1 + R), area_size)
                b1 = np.random.rand(1, 2)
                b2 = np.random.rand(1, 2)
                new_pos = Xstar + b1 * (current_pos - Lb_star) + b2 * (current_pos - Ub_star)
            
            elif i < n_ball_rolling + n_brood + n_small:  # Small dung beetles (Equations 5, 6)
                Lbb = np.maximum(Xb * (1 - R), 0)
                Ubb = np.minimum(Xb * (1 + R), area_size)
                C1 = np.random.normal(0, 1)
                C2 = np.random.rand(1, 2)
                new_pos = current_pos + C1 * (current_pos - Lbb) + C2 * (current_pos - Ubb)
            
            else:  # Thieves (Equation 7)
                g = np.random.normal(0, 1, 2)
                new_pos = Xb + S * g * (np.abs(current_pos - Xstar) + np.abs(current_pos - Xb))
            
            # Boundary check
            new_pos = np.clip(new_pos, 0, area_size)
            
            # Evaluate fitness
            new_fitness = fitness_function(new_pos, anchor_nodes, estimated_distances)
            
            # Update if better
            if new_fitness < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fitness
                
                # Update global best if necessary
                if new_fitness < best_fitness:
                    Xb = new_pos.copy()
                    best_fitness = new_fitness
                    Xstar = Xb.copy()  # Update local best (assumed to track global best for simplicity)
        
        # Update global worst
        worst_idx = np.argmax(fitness)
        Xw = population[worst_idx].copy()
        
        # Update previous positions
        prev_population = population.copy()
        
        t += 1
    
    return Xb, best_fitness

def node_localization(num_tn, num_an, max_iter, area_size, noise_factor, transmission_range, population_size, fc, collect_positions=False):
    """Perform node localization and collect positions for plotting (Section 5)."""
    target_nodes, anchor_nodes = initialize_nodes(num_tn, num_an, area_size)
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
            
            estimated_pos, _ = dbo_optimization(tn, anchors, estimated_dists, max_iter, area_size, population_size, fc)
            
            error = euclidean_distance(tn, estimated_pos)
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
    print("\nTable 5: Performance Metrics of DBO")
    print("| TN | AN | Iterations | NL_E (m) | NL_T (s) | NL_A |")
    print("|----|----|------------|----------|----------|------|")
    for r in results:
        print(f"| {r['TN']} | {r['AN']} | {r['Iterations']} | {r['NL_E']} | {r['NL_T']} | {r['NL_A']} |")
    
    # Print Table 6
    print("\nTable 6: Result Summary of DBO")
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