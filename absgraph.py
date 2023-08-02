import matplotlib.pyplot as plt

# Data for Accuracy and F1-Score for six windows sizes and three learning rates
windows_sizes = [6, 12, 18, 24, 30, 36]
learning_rates = [0.01, 0.001, 0.0001]

accuracy_data = [
    [0.9960530823192982, 0.9956658375657199, 0.9953381689280767, 0.995725413681655, 0.9961573405221846, 0.9954126390729956],
    [0.9968871479423899, 0.9969169360003575, 0.9967531016815359, 0.996410539014909, 0.9966488434786495, 0.9971701344930817],
    [0.9969020419713737, 0.9969020419713737, 0.9966488434786495, 0.9967084195945846, 0.9967233136235684, 0.9969169360003575]
]

f1_score_data = [
    [0.9960530823087916, 0.9956658360658477, 0.9953381608286486, 0.9957254118875833, 0.9961573403176026, 0.9954126390607843],
    [0.9968871478664315, 0.996916934986784, 0.9967531003807345, 0.9964105320540105, 0.996648842644561, 0.9971701333593558],
    [0.9969020418641662, 0.9969020419218934, 0.9966488433433516, 0.996708419579981, 0.9967233135581494, 0.996916935996254]
]

# Convert accuracy and F1-Score data to percentage
accuracy_data_percent = [[val * 100 for val in window] for window in accuracy_data]
f1_score_data_percent = [[val * 100 for val in window] for window in f1_score_data]

# Create bar graph for Accuracy
plt.figure(figsize=(10, 6))
for i, learning_rate in enumerate(learning_rates):
    plt.bar([w + i * 0.2 for w in windows_sizes], accuracy_data_percent[i], width=0.2, label=f'Learning Rate: {learning_rate}')

plt.title('Accuracy for Different Window Sizes and Learning Rates')
plt.xlabel('Window Size')
plt.ylabel('Accuracy (%)')
plt.xticks([w + 0.2 for w in windows_sizes], windows_sizes)
plt.legend()
plt.ylim(99, 100)
plt.savefig('accuracy_graph.png')
plt.show()

# Create bar graph for F1-Score
plt.figure(figsize=(10, 6))
for i, learning_rate in enumerate(learning_rates):
    plt.bar([w + i * 0.2 for w in windows_sizes], f1_score_data_percent[i], width=0.2, label=f'Learning Rate: {learning_rate}')

plt.title('F1-Score for Different Window Sizes and Learning Rates')
plt.xlabel('Window Size')
plt.ylabel('F1-Score (%)')
plt.xticks([w + 0.2 for w in windows_sizes], windows_sizes)
plt.legend()
plt.ylim(99, 100)
plt.savefig('f1_score_graph.png')
plt.show()
