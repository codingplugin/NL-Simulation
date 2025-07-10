import numpy as np
import time
import matplotlib.pyplot as plt
import csv
import math

# Set a fixed random seed for reproducibility (important for research and paper-like results)
np.random.seed(42)

# Simulation Parameters (Table 4, Section 5.1)
AREA_SIZE = 200  # 200x200 m^2
TRANSMISSION_RANGE = 30  # meters
FC = 2  # Frequency-controlled parameter (kept as a placeholder, not used in AO)
MAX_ITERATIONS = [20, 40, 60, 80, 100]
TARGET_NODES = [25, 50, 75, 100, 125, 150]
ANCHOR_NODES = [15, 30, 45, 60, 75, 90]  # Per Table 4
NOISE_FACTOR = 0.1  # Noise in distance estimation (Section 5.2)
POPULATION_SIZE = 50  # Population size for AO (default)

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

def levy_flight(D, beta=1.5):
    """Calculate Levy flight distribution (Equation 6 in the paper)."""
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, D)
    v = np.random.normal(0, 1, D)
    step = u / (np.abs(v) ** (1 / beta))
    return step

def ao_optimization(target_pos, anchor_nodes, estimated_distances, max_iter, area_size, population_size, fc):
    """Aquila Optimizer for node localization (Section 3.3 of the paper)."""
    # Initialize population (Equation 2)
    population = np.random.uniform(0, area_size, (population_size, 2))
    fitness = np.array([fitness_function(p, anchor_nodes, estimated_distances) for p in population])
    
    best_idx = np.argmin(fitness)
    X_best = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    t = 1
    T = max_iter
    Dim = 2  # 2D space (x, y coordinates)
    
    while t <= T:
        # Calculate mean location of current solutions (Equation 4)
        XM = np.mean(population, axis=0)
        
        for i in range(population_size):
            if t <= (2/3) * T:  # Exploration phase
                # Step 1: Expanded exploration (X1, Equation 3)
                if np.random.rand() < 0.5:
                    X1 = X_best * (1 - t / T) + (XM - X_best * np.random.rand())
                    new_pos = np.clip(X1, 0, area_size)
                # Step 2: Narrowed exploration (X2, Equation 5)
                else:
                    XR = population[np.random.randint(0, population_size)]
                    y, x = np.random.uniform(-1, 1, 2)  # Random y, x in [-1, 1]
                    levy = levy_flight(Dim)
                    X2 = X_best * levy + XR + (y - x) * np.random.rand()
                    new_pos = np.clip(X2, 0, area_size)
            else:  # Exploitation phase
                # Step 3: Expanded exploitation (X3, Equation 8)
                if np.random.rand() < 0.5:
                    alpha, delta = 0.1, 0.1  # Parameters from Section 3.3.3
                    UB, LB = area_size, 0
                    X3 = (X_best - XM) * alpha - np.random.rand() * (UB - LB) * delta + LB
                    new_pos = np.clip(X3, 0, area_size)
                # Step 4: Narrowed exploitation (X4, Equation 10)
                else:
                    G1 = 2 * np.random.rand() - 1  # Equation 11
                    G2 = 2 * (1 - t / T)  # Equation 12
                    QF = t ** ((2 * np.random.rand() - 1) / (1 - T) ** 2)  # Equation 13
                    levy = levy_flight(Dim)
                    X4 = QF * X_best - (G2 * population[i] * np.random.rand()) - G2 * levy + np.random.rand() * G1
                    new_pos = np.clip(X4, 0, area_size)
            
            new_fitness = fitness_function(new_pos, anchor_nodes, estimated_distances)
            
            if new_fitness < fitness[i]:
                population[i] = new_pos
                fitness[i] = new_fitness
                
                if new_fitness < best_fitness:
                    X_best = new_pos.copy()
                    best_fitness = new_fitness
        
        t += 1
    
    return X_best, best_fitness

def node_localization(num_tn, num_an, max_iter, area_size, noise_factor, transmission_range, population_size, fc, collect_positions=False):
    """Perform node localization and collect positions for plotting (Section 5)."""
    target_nodes, anchor_nodes = initialize_nodes(num_tn, num_an, area_size)
    localized_nodes = 0
    total_error = 0
    start_time = time.time()
    
    if collect_positions:
        unlocalized_positions = []
        actual_positions = []
        est_positions = []
    
    for tn in target_nodes:
        distances = [euclidean_distance(tn, an) for an in anchor_nodes]
        valid_anchors = [(an, dist) for an, dist in zip(anchor_nodes, distances) if dist <= transmission_range]
        
        if len(valid_anchors) >= 3:  # Localizable node (Section 5.3)
            anchors, actual_dists = zip(*valid_anchors)
            anchors = np.array(anchors)
            estimated_dists = [estimate_distance(d, noise_factor) for d in actual_dists]
            
            estimated_pos, _ = ao_optimization(tn, anchors, estimated_dists, max_iter, area_size, population_size, fc)
            
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
    unlocalized_positions = positions_list[0][0]
    anchor_nodes = positions_list[0][3]
    
    actual_positions_all = []
    est_positions_all = []
    for _, actual_pos, est_pos, _ in positions_list:
        actual_positions_all.extend(actual_pos)
        est_positions_all.extend(est_pos)
    
    if len(actual_positions_all) > anl_a:
        indices = np.random.choice(len(actual_positions_all), size=anl_a, replace=False)
        actual_positions_all = [actual_positions_all[i] for i in indices]
        est_positions_all = [est_positions_all[i] for i in indices]
    
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
                'NL_E': round(r['NL_E'], 3),
                'NL_T': round(r['NL_T'], 2),
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
                'ANL_E': round(r['ANL_E'], 3),
                'ANL_T': round(r['ANL_T'], 2),
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
    print("\nTable 5: Performance Metrics of AO")
    print("| TN | AN | Iterations | NL_E (m) | NL_T (s) | NL_A |")
    print("|----|----|------------|----------|----------|------|")
    for r in results:
        print(f"| {r['TN']} | {r['AN']} | {r['Iterations']} | {r['NL_E']} | {r['NL_T']} | {r['NL_A']} |")
    
    # Print Table 6
    print("\nTable 6: Result Summary of AO")
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