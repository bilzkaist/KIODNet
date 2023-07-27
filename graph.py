import numpy as np
import matplotlib.pyplot as plt

# Function to load results from text file
def load_results_from_file(results_file):
    with open(results_file, 'r') as f:
        data = f.readlines()
    return [float(x.strip()) for x in data]

# Load saved results for balanced dataset
balanced_accuracy_results = load_results_from_file('results_balanced_accuracy.txt')
balanced_f1_results = load_results_from_file('results_balanced_f1.txt')

# Load saved results for unbalanced dataset
unbalanced_accuracy_results = load_results_from_file('results_unbalanced_accuracy.txt')
unbalanced_f1_results = load_results_from_file('results_unbalanced_f1.txt')

# Calculate mean and median for both datasets
balanced_accuracy_mean = np.mean(balanced_accuracy_results)
balanced_accuracy_median = np.median(balanced_accuracy_results)
balanced_f1_mean = np.mean(balanced_f1_results)
balanced_f1_median = np.median(balanced_f1_results)

unbalanced_accuracy_mean = np.mean(unbalanced_accuracy_results)
unbalanced_accuracy_median = np.median(unbalanced_accuracy_results)
unbalanced_f1_mean = np.mean(unbalanced_f1_results)
unbalanced_f1_median = np.median(unbalanced_f1_results)

# Plot colorful bar graph for both datasets
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.2

# Bar positions for the mean and median
bar_positions_mean = np.arange(4)
bar_positions_median = bar_positions_mean + bar_width

# Plot the bars for balanced dataset
balanced_means = [balanced_accuracy_mean, balanced_f1_mean, unbalanced_accuracy_mean, unbalanced_f1_mean]
balanced_medians = [balanced_accuracy_median, balanced_f1_median, unbalanced_accuracy_median, unbalanced_f1_median]

ax.bar(bar_positions_mean, balanced_means, bar_width, label='Mean', color='dodgerblue')
ax.bar(bar_positions_median, balanced_medians, bar_width, label='Median', color='tomato')

# Set labels and titles
ax.set_xticks(bar_positions_mean + bar_width / 2)
ax.set_xticklabels(['Accuracy (Balanced)', 'F1-Score (Balanced)', 'Accuracy (Unbalanced)', 'F1-Score (Unbalanced)'])
ax.set_ylabel('Percentage (%)')
ax.set_title('Mean and Median of Accuracy and F1-Score for Balanced and Unbalanced Datasets')
ax.legend()

# Display the numerical values on the bars
for i in range(len(bar_positions_mean)):
    ax.text(bar_positions_mean[i], balanced_means[i], f'{balanced_means[i]:.4f}%', ha='center', va='bottom', fontsize=9)
    ax.text(bar_positions_median[i], balanced_medians[i], f'{balanced_medians[i]:.4f}%', ha='center', va='bottom', fontsize=9)

# Display the graph
plt.tight_layout()
plt.show()
