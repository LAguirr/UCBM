import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_tree_experiments():
    """
    Generate two graphs from tree experiment CSV files:
    1. min_samples_split vs train/test accuracy
    2. max_depth vs train/test accuracy
    """
    
    # Define file paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    min_samples_file = os.path.join(base_path, 'tree_experiments_min_samples_split.csv')
    max_depth_file = os.path.join(base_path, 'tree_experiments_max_depth.csv')
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: min_samples_split
    if os.path.exists(min_samples_file):
        df1 = pd.read_csv(min_samples_file)
        ax1.plot(df1['min_samples_split'], df1['train acc'], 'o-', label='Train Accuracy', linewidth=2, markersize=6)
        ax1.plot(df1['min_samples_split'], df1['test acc'], 's-', label='Test Accuracy', linewidth=2, markersize=6)
        ax1.set_xlabel('min_samples_split', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Decision Tree: min_samples_split vs Accuracy', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, f'File not found:\n{min_samples_file}', 
                ha='center', va='center', transform=ax1.transAxes)
    
    # Plot 2: max_depth
    if os.path.exists(max_depth_file):
        df2 = pd.read_csv(max_depth_file)
        ax2.plot(df2['max_depth'], df2['train acc'], 'o-', label='Train Accuracy', linewidth=2, markersize=6)
        ax2.plot(df2['max_depth'], df2['test acc'], 's-', label='Test Accuracy', linewidth=2, markersize=6)
        ax2.set_xlabel('max_depth', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Decision Tree: max_depth vs Accuracy', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, f'File not found:\n{max_depth_file}', 
                ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(base_path, 'tree_experiments_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graph saved to: {output_path}")
    
    plt.show()


if __name__ == '__main__':
    plot_tree_experiments()
