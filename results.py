import numpy as np

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

# Display numeric values for balanced dataset
print("Balanced Dataset - Accuracy:")
print("Mean Accuracy: {:.2f}%".format(np.mean(balanced_accuracy_results) * 100))
print("Median Accuracy: {:.2f}%".format(np.median(balanced_accuracy_results) * 100))
print("Standard Deviation Accuracy: {:.2f}%".format(np.std(balanced_accuracy_results) * 100))

print("\nBalanced Dataset - F1-Score:")
print("Mean F1-Score: {:.2f}%".format(np.mean(balanced_f1_results) * 100))
print("Median F1-Score: {:.2f}%".format(np.median(balanced_f1_results) * 100))
print("Standard Deviation F1-Score: {:.2f}%".format(np.std(balanced_f1_results) * 100))

# Display numeric values for unbalanced dataset
print("\nUnbalanced Dataset - Accuracy:")
print("Mean Accuracy: {:.2f}%".format(np.mean(unbalanced_accuracy_results) * 100))
print("Median Accuracy: {:.2f}%".format(np.median(unbalanced_accuracy_results) * 100))
print("Standard Deviation Accuracy: {:.2f}%".format(np.std(unbalanced_accuracy_results) * 100))

print("\nUnbalanced Dataset - F1-Score:")
print("Mean F1-Score: {:.2f}%".format(np.mean(unbalanced_f1_results) * 100))
print("Median F1-Score: {:.2f}%".format(np.median(unbalanced_f1_results) * 100))
print("Standard Deviation F1-Score: {:.2f}%".format(np.std(unbalanced_f1_results) * 100))
