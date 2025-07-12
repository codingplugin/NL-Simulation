import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the comparison data
df = pd.read_csv('algorithm_comparison_table6.csv')

# Filter out the average performance rows for individual network size analysis
df_networks = df[~df['Network_Size'].str.contains('Average')]

# Set up the plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'

# Colors for each algorithm
colors = {
    'Dung_Beetle': '#2E8B57',  # Sea Green
    'Chimp': '#4169E1',        # Royal Blue
    'Aquila': '#FF6347',       # Tomato
    'Covid': '#9370DB'         # Medium Purple
}

# Network sizes for x-axis
network_sizes = df_networks['Network_Size'].unique()
x = np.arange(len(network_sizes))
width = 0.2

# 1. NLE GRAPH (Normalized Localization Error)
fig1, ax1 = plt.subplots(figsize=(14, 8))

for i, algorithm in enumerate(['Dung_Beetle', 'Chimp', 'Aquila', 'Covid']):
    data = df_networks[df_networks['Algorithm'] == algorithm]['NLE_m'].values
    bars = ax1.bar(x + i*width, data, width, label=algorithm.replace('_', ' '), 
                    color=colors[algorithm], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, data):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                 f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_xlabel('Network Size (Target Nodes, Anchor Nodes)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Normalized Localization Error (NLE) in meters', fontsize=14, fontweight='bold')

ax1.set_xticks(x + width*1.5)
ax1.set_xticklabels([size.replace('_', '\n') for size in network_sizes], fontsize=12)
ax1.legend(loc='upper right', fontsize=12, framealpha=0.9)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(0, max(df_networks['NLE_m']) * 1.15)

# Add performance ranking text


plt.tight_layout()
plt.savefig('NLE_comparison_detailed.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. NLT GRAPH (Normalized Localization Time)
fig2, ax2 = plt.subplots(figsize=(14, 8))

for i, algorithm in enumerate(['Dung_Beetle', 'Chimp', 'Aquila', 'Covid']):
    data = df_networks[df_networks['Algorithm'] == algorithm]['NLT_s'].values
    bars = ax2.bar(x + i*width, data, width, label=algorithm.replace('_', ' '), 
                    color=colors[algorithm], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, data):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                 f'{value:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_xlabel('Network Size (Target Nodes, Anchor Nodes)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Normalized Localization Time (NLT) in seconds', fontsize=14, fontweight='bold')

ax2.set_xticks(x + width*1.5)
ax2.set_xticklabels([size.replace('_', '\n') for size in network_sizes], fontsize=12)
ax2.legend(loc='upper left', fontsize=12, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim(0, max(df_networks['NLT_s']) * 1.15)

# Add performance ranking text


plt.tight_layout()
plt.savefig('NLT_comparison_detailed.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. NLA GRAPH (Number of Localized Anchors)
fig3, ax3 = plt.subplots(figsize=(14, 8))

for i, algorithm in enumerate(['Dung_Beetle', 'Chimp', 'Aquila', 'Covid']):
    data = df_networks[df_networks['Algorithm'] == algorithm]['NLA'].values
    bars = ax3.bar(x + i*width, data, width, label=algorithm.replace('_', ' '), 
                    color=colors[algorithm], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, data):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'{int(value)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3.set_xlabel('Network Size (Target Nodes, Anchor Nodes)', fontsize=14, fontweight='bold')
ax3.set_ylabel('Number of Localized Anchors (NLA)', fontsize=14, fontweight='bold')

ax3.set_xticks(x + width*1.5)
ax3.set_xticklabels([size.replace('_', '\n') for size in network_sizes], fontsize=12)
ax3.legend(loc='upper left', fontsize=12, framealpha=0.9)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_ylim(0, max(df_networks['NLA']) * 1.15)

# Add performance ranking text


plt.tight_layout()
plt.savefig('NLA_comparison_detailed.png', dpi=300, bbox_inches='tight')
plt.show()

print("Three detailed comparison graphs have been created and saved!")
print("Files generated:")
print("1. NLE_comparison_detailed.png - Normalized Localization Error comparison")
print("2. NLT_comparison_detailed.png - Normalized Localization Time comparison")
print("3. NLA_comparison_detailed.png - Number of Localized Anchors comparison")
print("\nKey Features:")
print("- Value labels on each bar")
print("- Performance ranking boxes")
print("- Clear color coding for algorithms")
print("- Grid lines for easy reading")
print("- Professional formatting and titles") 