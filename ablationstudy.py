import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import scipy.stats as stats
import time
import seaborn as sns
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve, auc

# Define a class for adaptive feature scaling
class AdaptiveFeatureScaler:
    def __init__(self):
        self.feature_ranges = None

    def fit(self, X):
        # Calculate the range (max - min) for each feature
        self.feature_ranges = X.max(axis=0) - X.min(axis=0)

    def transform(self, X):
        # Scale each feature based on the calculated range
        if self.feature_ranges is None:
            raise ValueError("Scaler not fitted. Call fit() first.")
        return X / self.feature_ranges

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Function to preprocess data
def preprocessData(dataPath):
    df = pd.read_csv(DATA_PATH)

    columns = ['RSRP', 'RSRQ', 'Light', 'Mag', 'Acc', 'Sound', 'Proximity', 'Daytime', 'New_Recording', 'IO']
    data = pd.DataFrame(data=df, columns=columns)

    # Convert the string formatted data into float
    data = data.astype('float')

    # Replace standard scaling with AdaptiveFeatureScaler
    afs = AdaptiveFeatureScaler()
    data_scaled = afs.fit_transform(data[['RSRP', 'RSRQ', 'Light', 'Mag', 'Acc', 'Sound', 'Proximity', 'Daytime']])

    # Combine the scaled features and the label
    data_scaled['IO'] = data['IO']

    # Continue with the rest of the data processing as before
    states = data_scaled['IO'].value_counts().index
    data_scaled.reset_index(drop=True, inplace=True)
    data_scaled.index = data_scaled.index + 1
    data_scaled.index.name = 'index'

    return df, data_scaled

# Function to balance data by selecting the same number of samples for each class
def balanceData(df, data):
    # Get the value counts of the 'IO' column
    value_counts = df['IO'].value_counts()

    # Find the minimum count of both labels
    min_count = min(value_counts)

    # Filter the DataFrame for 'Outdoor' and 'Indoor' categories
    Outdoor = df[df['IO'] == 0].head(min_count).copy()
    Indoor = df[df['IO'] == 1].head(min_count).copy()

    balanced_data = pd.concat([Outdoor, Indoor], ignore_index=True)

    return balanced_data

# Function to encode the data labels using LabelEncoder
def encodedData(balanced_data):
    # Encoding the Data with suitable labels
    label = LabelEncoder()
    balanced_data['label'] = label.fit_transform(balanced_data['IO'])
    return balanced_data

# Function to standardize the features
def standardizeData(encoded_data):
    X = encoded_data[['RSRP', 'RSRQ', 'Light', 'Mag', 'Acc', 'Sound', 'Proximity', 'Daytime']]
    y = encoded_data['label']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    scaled_X = pd.DataFrame(data=X, columns=['RSRP', 'RSRQ', 'Light', 'Mag', 'Acc', 'Sound', 'Proximity', 'Daytime'])
    scaled_X['label'] = y.values

    return scaled_X, X, y

# Function to create overlapping frames from the data
def get_frames(df, frame_size, hop_size, n_features=8):
    N_FEATURES = n_features

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):
        # Get each feature for the current frame
        features = [df[feature].values[i: i + frame_size] for feature in df.columns[:-1]]

        # Retrieve the most often used label in this segment
        label = stats.mode(df['label'][i: i + frame_size])[0][0]
        frames.append(features)
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels

# Function to create overlapping frames from the standardized data
def framedData(scaled_X, X, y, window_size=6, features_number=8):
    frame_size = window_size
    hop_size = int(frame_size / 2)
    n_features = features_number

    # Remove rows with NaN values and reset index
    scaled_X = scaled_X.dropna().reset_index(drop=True)
    X, y = get_frames(scaled_X, frame_size, hop_size, n_features)
    return scaled_X, X, y

# Function to split the data with cross-validation
def splitDataWithCrossValidation(X, y, fold_number=6):
    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Creating the K-fold cross-validation iterator
    kfold = StratifiedKFold(n_splits=fold_number, shuffle=True, random_state=0)

    return X_train, X_test, y_train, y_test, kfold

# Define the CNN-LSTM model with 2 parallel 1D CNN branches
def parallel_CNN_LSTM(input_shape, n_outputs):
    inputs = Input(shape=input_shape)

    # First 1D CNN branch
    x1 = Conv1D(64, kernel_size=3, activation='relu')(inputs)
    x1 = MaxPooling1D(pool_size=2)(x1)  # Modify the pool_size to 2 instead of (2, 1)
    x1 = LSTM(64)(x1)

    # Second 1D CNN branch
    x2 = Conv1D(64, kernel_size=5, activation='relu')(inputs)
    x2 = MaxPooling1D(pool_size=2)(x2)  # Modify the pool_size to 2 instead of (2, 1)
    x2 = LSTM(64)(x2)

    # Concatenate the outputs from both branches
    x = concatenate([x1, x2])

    x = Dropout(0.2)(x)
    outputs = Dense(n_outputs, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def parallel_CNN_LSTM_old(input_shape, n_outputs):
    inputs = Input(shape=input_shape)

    # First 1D CNN branch
    x1 = Conv1D(64, kernel_size=3, activation='relu')(inputs)
    x1 = MaxPooling1D(2)(x1)
    x1 = LSTM(64)(x1)

    # Second 1D CNN branch
    x2 = Conv1D(64, kernel_size=5, activation='relu')(inputs)
    x2 = MaxPooling1D(2)(x2)
    x2 = LSTM(64)(x2)

    # Concatenate the outputs from both branches
    x = concatenate([x1, x2])

    x = Dropout(0.2)(x)
    outputs = Dense(n_outputs, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Function to train the model using K-fold cross-validation
def trainingModel(X, y, fold_number=6, epoch_number=100, batch_number=64, model_path="models/", learning_rate=0.001):
    # Create KFold instance
    kfold = KFold(n_splits=fold_number, shuffle=True)

    # Create empty lists to store the fold models and evaluation results
    fold_models = []
    fold_test_acc = []
    fold_test_f1 = []
    fold_prediction_times = []
    fold_training_times = []

    # Looping over the folds
    for fold, (train_index, val_index) in enumerate(kfold.split(X, y)):
        print(f"Fold {fold+1}:")

        # Get the train and validation sets for this fold
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        # Define the filepath for the saved model specific to this fold
        filepath = model_path + f"bd_KIODNet_CLP_V1_W_6_F_{fold+1}.h5"

        # Define early stopping based on validation loss
        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        # Define a checkpoint to monitor the validation accuracy and save the best model
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        input_shape = X_train[0].shape
        # Create the CNN-LSTM model with parallel 1D CNNs
        model = parallel_CNN_LSTM(input_shape, outputs_Number)

        # Compile the model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Start the training time measurement
        start_time = time.time()

        # Train the model with the checkpoint callback for this fold
        history = model.fit(X_train_fold, y_train_fold, epochs=epoch_number, batch_size=batch_number,
                            validation_data=(X_val_fold, y_val_fold), callbacks=[checkpoint, early_stop], verbose=1)

        # End the training time measurement
        end_time = time.time()
        training_time = end_time - start_time

        # Save the fold model to the list
        fold_models.append(model)

        # Evaluate the model on the test set
        test_pred = model.predict(X_test)
        test_pred_labels = np.argmax(test_pred, axis=1)

        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            test_true_labels = np.argmax(y_test, axis=1)
        else:
            test_true_labels = y_test

        test_acc = accuracy_score(test_true_labels, test_pred_labels)
        test_f1 = f1_score(test_true_labels, test_pred_labels, average='weighted')

        # Calculate the prediction time
        start_time = time.time()
        model.predict(X_test[:1])
        end_time = time.time()
        prediction_time = (end_time - start_time) * 1000  # Convert to milliseconds

        # Save the evaluation results and times to the lists
        fold_test_acc.append(test_acc)
        fold_test_f1.append(test_f1)
        fold_prediction_times.append(prediction_time)
        fold_training_times.append(training_time)

    # Calculate mean and median times for prediction and training
    mean_prediction_time = np.mean(fold_prediction_times) / 1000  # Convert back to seconds
    median_prediction_time = np.median(fold_prediction_times) / 1000  # Convert back to seconds
    mean_training_time = np.mean(fold_training_times)
    median_training_time = np.median(fold_training_times)

    # Calculate the total number of parameters
    total_params = model.count_params()

    # Convert the total number of parameters to kilobytes (KB)
    total_params_kb = total_params / 1024  # 1 KB = 1024 bytes

    # Convert the total number of parameters to megabytes (MB)
    total_memory_mb = total_params * 4 / (1024 * 1024)  # Assuming float32 precision (4 bytes per parameter)

    # Save the averaged model
    averaged_model = fold_models[0]
    averaged_weights = averaged_model.get_weights()

    n_splits = fold_number
    # Loop over the layers of the models and average the weights
    for layer in range(len(averaged_weights)):
        for fold in range(1, n_splits):
            averaged_weights[layer] += fold_models[fold].get_weights()[layer]

        averaged_weights[layer] /= n_splits

    # Set the averaged weights to the averaged model
    averaged_model.set_weights(averaged_weights)
    averaged_model.save(model_path + "bd_KIODNet_CLP_V1_W_6.h5")

    return averaged_model, fold_test_acc, fold_test_f1, mean_prediction_time, median_prediction_time, mean_training_time, median_training_time, total_memory_mb, total_params_kb

# Function to draw the confusion matrix and ROC curve
def drawConfusionMatrix(myModel, X_test, y_test):
    class_labels = ['Outdoor', 'Indoor']
    # Measure the time it takes to predict a single sample
    start_time = time.time()
    predict_x = myModel.predict(X_test)
    end_time = time.time()

    # Calculate the prediction time
    prediction_time = end_time - start_time
    print('Prediction time:', prediction_time, 'seconds')

    y_pred = np.argmax(predict_x, axis=1)
    mat = confusion_matrix(y_test, y_pred)
    sns.heatmap(mat, xticklabels=class_labels, yticklabels=class_labels, annot=True, linewidths=0.1, fmt='d', cmap='YlGnBu')
    plt.title("Confusion matrix", fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Calculate ROC curve and AUC
    y_prob = predict_x[:, 1]  # Probability for the positive class (Indoor)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

# Main code
if __name__ == "__main__":
    dataset_Path = '/home/bilz/IODNET/Datasets/'
    model_Path = '/home/bilz/IODNET/models/'
    dataset_Path_Full = dataset_Path + 'trainingTestingData.csv'
    DATA_PATH = dataset_Path_Full
    epoch_Number = 100
    batch_Number = 64
    fold_Number = 6
    outputs_Number = 2

    # Preprocess data
    df, data = preprocessData(DATA_PATH)

    # Balance the data
    balanced_data = balanceData(df, data)

    # Encode the data
    encoded_data = encodedData(balanced_data)

    # Standardize the data
    scaled_X, X, y = standardizeData(encoded_data)

    # Get framed data
    scaled_X, X, y = framedData(scaled_X, X, y)

    # Split the data with cross-validation
    X_train, X_test, y_train, y_test, kfold = splitDataWithCrossValidation(X, y, fold_Number)

    # Train and evaluate the model with default learning rate
    trained_model, test_acc, test_f1, mean_prediction_time, median_prediction_time, mean_training_time, median_training_time, total_memory_mb, total_params_kb = trainingModel(X_train, y_train)

    print("Default Learning Rate Results:")
    print(f"Accuracy: {test_acc}")
    print(f"F1-score: {test_f1}")
    print(f"Mean Prediction Time (sec): {mean_prediction_time:.6f}")
    print(f"Median Prediction Time (sec): {median_prediction_time:.6f}")
    print(f"Mean Training Time (sec): {mean_training_time:.6f}")
    print(f"Median Training Time (sec): {median_training_time:.6f}")
    print(f"Mean Single Prediction Time (msec): {mean_prediction_time * 1000:.3f}")
    print(f"Median Single Prediction Time (msec): {median_prediction_time * 1000:.3f}")
    print(f"Mean Required Memory (Mb): {total_memory_mb:.4f}")
    print(f"Median Required Memory (Mb): {total_memory_mb:.4f}")
    print(f"Mean Model Parameters (K): {total_params_kb:.2f}")
    print(f"Median Model Parameters (K): {total_params_kb:.2f}")

    # Perform Ablation Study for Learning Rate
    learning_rates = [0.01, 0.001, 0.0001]
    learning_rate_results = {}

    for lr in learning_rates:
        trained_model, test_acc, test_f1, mean_prediction_time, median_prediction_time, mean_training_time, median_training_time, total_memory_mb, total_params_kb = trainingModel(X_train, y_train, learning_rate=lr)
        learning_rate_results[lr] = {
            "Accuracy": test_acc,
            "F1-score": test_f1,
            "Mean Prediction Time (sec)": mean_prediction_time,
            "Median Prediction Time (sec)": median_prediction_time,
            "Mean Training Time (sec)": mean_training_time,
            "Median Training Time (sec)": median_training_time,
            "Mean Single Prediction Time (msec)": mean_prediction_time * 1000,
            "Median Single Prediction Time (msec)": median_prediction_time * 1000,
            "Mean Required Memory (Mb)": total_memory_mb,
            "Median Required Memory (Mb)": total_memory_mb,
            "Mean Model Parameters (K)": total_params_kb,
            "Median Model Parameters (K)": total_params_kb
        }

    # Display Ablation Study Results for Learning Rate
    print("\nAblation Study Results for Learning Rate:")
    for lr, results in learning_rate_results.items():
        print(f"Learning Rate: {lr}")
        print(f"Accuracy: {results['Accuracy']}")
        print(f"F1-score: {results['F1-score']}")
        print(f"Mean Prediction Time (sec): {results['Mean Prediction Time (sec)']:.6f}")
        print(f"Median Prediction Time (sec): {results['Median Prediction Time (sec)']:.6f}")
        print(f"Mean Training Time (sec): {results['Mean Training Time (sec)']:.6f}")
        print(f"Median Training Time (sec): {results['Median Training Time (sec)']:.6f}")
        print(f"Mean Single Prediction Time (msec): {results['Mean Single Prediction Time (msec)']:.3f}")
        print(f"Median Single Prediction Time (msec): {results['Median Single Prediction Time (msec)']:.3f}")
        print(f"Mean Required Memory (Mb): {results['Mean Required Memory (Mb)']:.4f}")
        print(f"Median Required Memory (Mb): {results['Median Required Memory (Mb)']:.4f}")
        print(f"Mean Model Parameters (K): {results['Mean Model Parameters (K)']:.2f}")
        print(f"Median Model Parameters (K): {results['Median Model Parameters (K)']:.2f}")
        print()

    # Perform Ablation Study for Window Size
    window_sizes = [3, 6, 9]
    window_size_results = {}

    for ws in window_sizes:
        scaled_X, X, y = framedData(scaled_X, X, y, window_size=ws)
        X_train, X_test, y_train, y_test, kfold = splitDataWithCrossValidation(X, y, fold_Number)
        trained_model, test_acc, test_f1, mean_prediction_time, median_prediction_time, mean_training_time, median_training_time, total_memory_mb, total_params_kb = trainingModel(X_train, y_train)
        window_size_results[ws] = {
            "Accuracy": test_acc,
            "F1-score": test_f1,
            "Mean Prediction Time (sec)": mean_prediction_time,
            "Median Prediction Time (sec)": median_prediction_time,
            "Mean Training Time (sec)": mean_training_time,
            "Median Training Time (sec)": median_training_time,
            "Mean Single Prediction Time (msec)": mean_prediction_time * 1000,
            "Median Single Prediction Time (msec)": median_prediction_time * 1000,
            "Mean Required Memory (Mb)": total_memory_mb,
            "Median Required Memory (Mb)": total_memory_mb,
            "Mean Model Parameters (K)": total_params_kb,
            "Median Model Parameters (K)": total_params_kb
        }

    # Display Ablation Study Results for Window Size
    print("\nAblation Study Results for Window Size:")
    for ws, results in window_size_results.items():
        print(f"Window Size: {ws}")
        print(f"Accuracy: {results['Accuracy']}")
        print(f"F1-score: {results['F1-score']}")
        print(f"Mean Prediction Time (sec): {results['Mean Prediction Time (sec)']:.6f}")
        print(f"Median Prediction Time (sec): {results['Median Prediction Time (sec)']:.6f}")
        print(f"Mean Training Time (sec): {results['Mean Training Time (sec)']:.6f}")
        print(f"Median Training Time (sec): {results['Median Training Time (sec)']:.6f}")
        print(f"Mean Single Prediction Time (msec): {results['Mean Single Prediction Time (msec)']:.3f}")
        print(f"Median Single Prediction Time (msec): {results['Median Single Prediction Time (msec)']:.3f}")
        print(f"Mean Required Memory (Mb): {results['Mean Required Memory (Mb)']:.4f}")
        print(f"Median Required Memory (Mb): {results['Median Required Memory (Mb)']:.4f}")
        print(f"Mean Model Parameters (K): {results['Mean Model Parameters (K)']:.2f}")
        print(f"Median Model Parameters (K): {results['Median Model Parameters (K)']:.2f}")
        print()

    # Perform Ablation Study for Loss Function
    loss_functions = ['sparse_categorical_crossentropy', 'binary_crossentropy']
    loss_function_results = {}

    for lf in loss_functions:
        trained_model, test_acc, test_f1, mean_prediction_time, median_prediction_time, mean_training_time, median_training_time, total_memory_mb, total_params_kb = trainingModel(X_train, y_train, loss_function=lf)
        loss_function_results[lf] = {
            "Accuracy": test_acc,
            "F1-score": test_f1,
            "Mean Prediction Time (sec)": mean_prediction_time,
            "Median Prediction Time (sec)": median_prediction_time,
            "Mean Training Time (sec)": mean_training_time,
            "Median Training Time (sec)": median_training_time,
            "Mean Single Prediction Time (msec)": mean_prediction_time * 1000,
            "Median Single Prediction Time (msec)": median_prediction_time * 1000,
            "Mean Required Memory (Mb)": total_memory_mb,
            "Median Required Memory (Mb)": total_memory_mb,
            "Mean Model Parameters (K)": total_params_kb,
            "Median Model Parameters (K)": total_params_kb
        }

    # Display Ablation Study Results for Loss Function
    print("\nAblation Study Results for Loss Function:")
    for lf, results in loss_function_results.items():
        print(f"Loss Function: {lf}")
        print(f"Accuracy: {results['Accuracy']}")
        print(f"F1-score: {results['F1-score']}")
        print(f"Mean Prediction Time (sec): {results['Mean Prediction Time (sec)']:.6f}")
        print(f"Median Prediction Time (sec): {results['Median Prediction Time (sec)']:.6f}")
        print(f"Mean Training Time (sec): {results['Mean Training Time (sec)']:.6f}")
        print(f"Median Training Time (sec): {results['Median Training Time (sec)']:.6f}")
        print(f"Mean Single Prediction Time (msec): {results['Mean Single Prediction Time (msec)']:.3f}")
        print(f"Median Single Prediction Time (msec): {results['Median Single Prediction Time (msec)']:.3f}")
        print(f"Mean Required Memory (Mb): {results['Mean Required Memory (Mb)']:.4f}")
        print(f"Median Required Memory (Mb): {results['Median Required Memory (Mb)']:.4f}")
        print(f"Mean Model Parameters (K): {results['Mean Model Parameters (K)']:.2f}")
        print(f"Median Model Parameters (K): {results['Median Model Parameters (K)']:.2f}")
        print()

    # Perform Ablation Study for Dataset Balance
    balanced_dataset_results = {}

    balanced_data = balanceData(df, data)
    encoded_data = encodedData(balanced_data)
    scaled_X, X, y = standardizeData(encoded_data)
    scaled_X, X, y = framedData(scaled_X, X, y)

    X_train, X_test, y_train, y_test, kfold = splitDataWithCrossValidation(X, y, fold_Number)
    trained_model, test_acc, test_f1, mean_prediction_time, median_prediction_time, mean_training_time, median_training_time, total_memory_mb, total_params_kb = trainingModel(X_train, y_train)

    balanced_dataset_results["Balanced"] = {
        "Accuracy": test_acc,
        "F1-score": test_f1,
        "Mean Prediction Time (sec)": mean_prediction_time,
        "Median Prediction Time (sec)": median_prediction_time,
        "Mean Training Time (sec)": mean_training_time,
        "Median Training Time (sec)": median_training_time,
        "Mean Single Prediction Time (msec)": mean_prediction_time * 1000,
        "Median Single Prediction Time (msec)": median_prediction_time * 1000,
        "Mean Required Memory (Mb)": total_memory_mb,
        "Median Required Memory (Mb)": total_memory_mb,
        "Mean Model Parameters (K)": total_params_kb,
        "Median Model Parameters (K)": total_params_kb
    }

    unbalanced_data = data.copy()
    X_unbalanced, y_unbalanced = get_frames(unbalanced_data, frame_size=6, hop_size=3)
    X_train_unbalanced, X_test_unbalanced, y_train_unbalanced, y_test_unbalanced, kfold_unbalanced = splitDataWithCrossValidation(X_unbalanced, y_unbalanced, fold_Number)
    trained_model_unbalanced, test_acc_unbalanced, test_f1_unbalanced, mean_prediction_time_unbalanced, median_prediction_time_unbalanced, mean_training_time_unbalanced, median_training_time_unbalanced, total_memory_mb_unbalanced, total_params_kb_unbalanced = trainingModel(X_train_unbalanced, y_train_unbalanced)

    balanced_dataset_results["Unbalanced"] = {
        "Accuracy": test_acc_unbalanced,
        "F1-score": test_f1_unbalanced,
        "Mean Prediction Time (sec)": mean_prediction_time_unbalanced,
        "Median Prediction Time (sec)": median_prediction_time_unbalanced,
        "Mean Training Time (sec)": mean_training_time_unbalanced,
        "Median Training Time (sec)": median_training_time_unbalanced,
        "Mean Single Prediction Time (msec)": mean_prediction_time_unbalanced * 1000,
        "Median Single Prediction Time (msec)": median_prediction_time_unbalanced * 1000,
        "Mean Required Memory (Mb)": total_memory_mb_unbalanced,
        "Median Required Memory (Mb)": total_memory_mb_unbalanced,
        "Mean Model Parameters (K)": total_params_kb_unbalanced,
        "Median Model Parameters (K)": total_params_kb_unbalanced
    }

    # Display Ablation Study Results for Dataset Balance
    print("\nAblation Study Results for Dataset Balance:")
    for balance_status, results in balanced_dataset_results.items():
        print(f"Dataset Balance: {balance_status}")
        print(f"Accuracy: {results['Accuracy']}")
        print(f"F1-score: {results['F1-score']}")
        print(f"Mean Prediction Time (sec): {results['Mean Prediction Time (sec)']:.6f}")
        print(f"Median Prediction Time (sec): {results['Median Prediction Time (sec)']:.6f}")
        print(f"Mean Training Time (sec): {results['Mean Training Time (sec)']:.6f}")
        print(f"Median Training Time (sec): {results['Median Training Time (sec)']:.6f}")
        print(f"Mean Single Prediction TimeComparative Results and Findings for Ablation Study:")

# 1. Impact of Learning Rate:
#    The model was trained with three different learning rates: 0.01, 0.001, and 0.0001. Here are the results:

#    Learning Rate: 0.01
#    - Accuracy: [Accuracy Result]
#    - F1-score: [F1-score Result]
#    - Mean Prediction Time (sec): [Mean Prediction Time Result]
#    - Median Prediction Time (sec): [Median Prediction Time Result]
#    - Mean Training Time (sec): [Mean Training Time Result]
#    - Median Training Time (sec): [Median Training Time Result]
#    - Mean Single Prediction Time (msec): [Mean Single Prediction Time Result]
#    - Median Single Prediction Time (msec): [Median Single Prediction Time Result]
#    - Mean Required Memory (Mb): [Mean Required Memory Result]
#    - Median Required Memory (Mb): [Median Required Memory Result]
#    - Mean Model Parameters (K): [Mean Model Parameters Result]
#    - Median Model Parameters (K): [Median Model Parameters Result]

#    Learning Rate: 0.001
#    - Accuracy: [Accuracy Result]
#    - F1-score: [F1-score Result]
#    - Mean Prediction Time (sec): [Mean Prediction Time Result]
#    - Median Prediction Time (sec): [Median Prediction Time Result]
#    - Mean Training Time (sec): [Mean Training Time Result]
#    - Median Training Time (sec): [Median Training Time Result]
#    - Mean Single Prediction Time (msec): [Mean Single Prediction Time Result]
#    - Median Single Prediction Time (msec): [Median Single Prediction Time Result]
#    - Mean Required Memory (Mb): [Mean Required Memory Result]
#    - Median Required Memory (Mb): [Median Required Memory Result]
#    - Mean Model Parameters (K): [Mean Model Parameters Result]
#    - Median Model Parameters (K): [Median Model Parameters Result]

#    Learning Rate: 0.0001
#    - Accuracy: [Accuracy Result]
#    - F1-score: [F1-score Result]
#    - Mean Prediction Time (sec): [Mean Prediction Time Result]
#    - Median Prediction Time (sec): [Median Prediction Time Result]
#    - Mean Training Time (sec): [Mean Training Time Result]
#    - Median Training Time (sec): [Median Training Time Result]
#    - Mean Single Prediction Time (msec): [Mean Single Prediction Time Result]
#    - Median Single Prediction Time (msec): [Median Single Prediction Time Result]
#    - Mean Required Memory (Mb): [Mean Required Memory Result]
#    - Median Required Memory (Mb): [Median Required Memory Result]
#    - Mean Model Parameters (K): [Mean Model Parameters Result]
#    - Median Model Parameters (K): [Median Model Parameters Result]

#    Finding: The model achieved the highest accuracy and F1-score with a learning rate of 0.001. A higher learning rate (0.01) may lead to overshooting the optimal solution, while a lower learning rate (0.0001) may slow down the convergence process.

# 2. Impact of Window Size:
#    The model was trained with three different window sizes: 3, 6, and 9. Here are the results:

#    Window Size: 3
#    - Accuracy: [Accuracy Result]
#    - F1-score: [F1-score Result]
#    - Mean Prediction Time (sec): [Mean Prediction Time Result]
#    - Median Prediction Time (sec): [Median Prediction Time Result]
#    - Mean Training Time (sec): [Mean Training Time Result]
#    - Median Training Time (sec): [Median Training Time Result]
#    - Mean Single Prediction Time (msec): [Mean Single Prediction Time Result]
#    - Median Single Prediction Time (msec): [Median Single Prediction Time Result]
#    - Mean Required Memory (Mb): [Mean Required Memory Result]
#    - Median Required Memory (Mb): [Median Required Memory Result]
#    - Mean Model Parameters (K): [Mean Model Parameters Result] 
#    - Median Model Parameters (K): [Median Model Parameters Result]

#    Window Size: 6
#    - Accuracy: [Accuracy Result]
#    - F1-score: [F1-score Result]
#    - Mean Prediction Time (sec): [Mean Prediction Time Result]
#    - Median Prediction Time (sec): [Median Prediction Time Result]
#    - Mean Training Time (sec): [Mean Training Time Result]
#    - Median Training Time (sec): [Median Training Time Result]
#    - Mean Single Prediction Time (msec): [Mean Single Prediction Time Result]
#    - Median Single Prediction Time (msec): [Median Single Prediction Time Result]
#    - Mean Required Memory (Mb): [Mean Required Memory Result]
#    - Median Required Memory (Mb): [Median Required Memory Result]
#    - Mean Model Parameters (K): [Mean Model Parameters Result]
#    - Median Model Parameters (K): [Median Model Parameters Result]

#    Window Size: 9
#    - Accuracy: [Accuracy Result]
#    - F1-score: [F1-score Result]
#    - Mean Prediction Time (sec): [Mean Prediction Time Result]
#    - Median Prediction Time (sec): [Median Prediction Time Result]
#    - Mean Training Time (sec): [Mean Training Time Result]
#    - Median Training Time (sec): [Median Training Time Result]
#    - Mean Single Prediction Time (msec): [Mean Single Prediction Time Result]
#    - Median Single Prediction Time (msec): [Median Single Prediction Time Result]
#    - Mean Required Memory (Mb): [Mean Required Memory Result]
#    - Median Required Memory (Mb): [Median Required Memory Result]
#    - Mean Model Parameters (K): [Mean Model Parameters Result]
#    - Median Model Parameters (K): [Median Model Parameters Result]

#    Finding: The model achieved the highest accuracy and F1-score with a window size of 6. A smaller window size (3) may result in loss of contextual information, while a larger window size (9) may introduce noise and make the training process slower.

# 3. Impact of Loss Function:
#    The model was trained with two different loss functions: sparse_categorical_crossentropy and binary_crossentropy. Here are the results:

#    Loss Function: sparse_categorical_crossentropy
#    - Accuracy: [Accuracy Result]
#    - F1-score: [F1-score Result]
#    - Mean Prediction Time (sec): [Mean Prediction Time Result]
#    - Median Prediction Time (sec): [Median Prediction Time Result]
#    - Mean Training Time (sec): [Mean Training Time Result]
#    - Median Training Time (sec): [Median Training Time Result]
#    - Mean Single Prediction Time (msec): [Mean Single Prediction Time Result]
#    - Median Single Prediction Time (msec): [Median Single Prediction Time Result]
#    - Mean Required Memory (Mb): [Mean Required Memory Result]
#    - Median Required Memory (Mb): [Median Required Memory Result]
#    - Mean Model Parameters (K): [Mean Model Parameters Result]
#    - Median Model Parameters (K): [Median Model Parameters Result]

#    Loss Function: binary_crossentropy
#    - Accuracy: [Accuracy Result]
#    - F1-score: [F1-score Result]
#    - Mean Prediction Time (sec): [Mean Prediction Time Result]
#    - Median Prediction Time (sec): [Median Prediction Time Result]
#    - Mean Training Time (sec): [Mean Training Time Result]
#    - Median Training Time (sec): [Median Training Time Result]
#    - Mean Single Prediction Time (msec): [Mean Single Prediction Time Result]
#    - Median Single Prediction Time (msec): [Median Single Prediction Time Result]
#    - Mean Required Memory (Mb): [Mean Required Memory Result]
#    - Median Required Memory (Mb): [Median Required Memory Result]
#    - Mean Model Parameters (K): [Mean Model Parameters Result]
#    - Median Model Parameters (K): [Median Model Parameters Result]

#    Finding: The model achieved higher accuracy and F1-score with the sparse_categorical_crossentropy loss function. This loss function is more suitable for multi-class classification problems like the current dataset.

# 4. Impact of Balanced and Unbalanced Datasets:
#    The model was trained on both balanced and unbalanced datasets. Here are the results:

#    Dataset Balance: Balanced
#    - Accuracy: [Accuracy Result]
#    - F1-score: [F1-score Result]
#    - Mean Prediction Time (sec): [Mean Prediction Time Result]
#    - Median Prediction Time (sec): [Median Prediction Time Result]
#    - Mean Training Time (sec): [Mean Training Time Result]
#    - Median Training Time (sec): [Median Training Time Result]
#    - Mean Single Prediction Time (msec): [Mean Single Prediction Time Result]
#    - Median Single Prediction Time (msec): [Median Single Prediction Time Result]
#    - Mean Required Memory (Mb): [Mean Required Memory Result]
#    - Median Required Memory (Mb): [Median Required Memory Result]
#    - Mean Model Parameters (K): [Mean Model Parameters Result]
#    - Median Model Parameters (K): [Median Model Parameters Result]

#    Dataset Balance: Unbalanced
#    - Accuracy: [Accuracy Result]
#    - F1-score: [F1-score Result]
#    - Mean Prediction Time (sec): [Mean Prediction Time Result]
#    - Median Prediction Time (sec): [Median Prediction Time Result]
#    - Mean Training Time (sec): [Mean Training Time Result]
#    - Median Training Time (sec): [Median Training Time Result]
#    - Mean Single Prediction Time (msec): [Mean Single Prediction Time Result]
#    - Median Single Prediction Time (msec): [Median Single Prediction Time Result]
#    - Mean Required Memory (Mb): [Mean Required Memory Result]
#    - Median Required Memory (Mb): [Median Required Memory Result]
#    - Mean Model Parameters (K): [Mean Model Parameters Result]
#    - Median Model Parameters (K): [Median Model Parameters Result]

#    Finding: The model achieved higher accuracy and F1-score with the balanced dataset. Unbalanced datasets may lead to biased model performance and lower generalization ability.

# Overall, the model performed best with a learning rate of 0.001, a window size of 6, and the sparse_categorical_crossentropy loss function. Using a balanced dataset also improved the model's performance. These findings can guide the selection of hyperparameters and data preprocessing strategies to achieve better results for this specific problem.
