import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import scipy.stats as stats
import time
import seaborn as sns
import os
import psutil
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve, auc
from mlxtend.plotting import plot_confusion_matrix

# Import the AdaptiveFeatureScaler class
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

def preprocessData(dataPath):
    df = pd.read_csv(dataPath)

    columns = ['RSRP', 'RSRQ', 'Light', 'Mag', 'Acc', 'Sound', 'Proximity', 'Daytime', 'New_Recording', 'IO']
    data = pd.DataFrame(data=df, columns=columns)

    # Convert the string formatted data into float
    data['RSRP'] = data['RSRP'].astype('float')
    data['RSRQ'] = data['RSRQ'].astype('float')
    data['Light'] = data['Light'].astype('float')
    data['Mag'] = data['Mag'].astype('float')
    data['Acc'] = data['Acc'].astype('float')
    data['Sound'] = data['Sound'].astype('float')
    data['Proximity'] = data['Proximity'].astype('float')
    data['Daytime'] = data['Daytime'].astype('float')
    data['New_Recording'] = data['New_Recording'].astype('float')
    data['IO'] = data['IO'].astype('float')

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
def balanceData(df, data, mode=True):
    # Get the value counts of the 'IO' column
    value_counts = df['IO'].value_counts()

    # Find the minimum count of both labels
    min_count = min(value_counts)

    # Filter the DataFrame for 'Outdoor' and 'Indoor' categories
    if mode == True:
        Outdoor = df[df['IO'] == 0].head(min_count).copy()
        Indoor = df[df['IO'] == 1].head(min_count).copy()
    else:
        # Randomly select half of the minimum count for each category
        half_min_count = min_count // 2

        # Randomly sample the 'Outdoor' and 'Indoor' dataframes
        Outdoor = df[df['IO'] == 0].sample(n=np.random.randint(half_min_count, min_count)).copy()
        Indoor = df[df['IO'] == 1].sample(n=np.random.randint(half_min_count, min_count)).copy()

    balanced_data = pd.DataFrame()
    balanced_data = balanced_data.append([Outdoor, Indoor])
    balanced_data.shape

    # Displaying the balanced data
    print('State Count:',balanced_data['IO'].value_counts())

    balanced_data = pd.concat([Outdoor, Indoor], ignore_index=True)

    return balanced_data



def encodedData(balanced_data):
    # Encoding the Data with suitable labels
    label = LabelEncoder()
    balanced_data['label'] = label.fit_transform(balanced_data['IO'])
    balanced_data.head()

    return balanced_data

def standardizeData(encoded_data):
    X = encoded_data[['RSRP','RSRQ','Light','Mag','Acc','Sound','Proximity','Daytime']]
    y = encoded_data['label']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    scaled_X = pd.DataFrame(data=X, columns=['RSRP','RSRQ','Light','Mag','Acc','Sound','Proximity','Daytime'])
    scaled_X['label'] = y.values

    return scaled_X, X, y

def framedData(scaled_X, X, y, windows_Size=6, features_Number=8):
    frame_size = windows_Size
    hop_size = int(frame_size/2)
    n_features = features_Number

    X.shape, y.shape

    scaled_X = scaled_X.dropna()
    scaled_X = scaled_X.reset_index(drop=True)
    X, y = get_frames(scaled_X, frame_size, hop_size)
    X.shape, y.shape
    return scaled_X, X, y

def get_frames(df, frame_size, hop_size, n_features=8):
    N_FEATURES = n_features

    frames = []
    labels = []
    for i in range(0, len(df) - frame_size, hop_size):

        RSRP = df['RSRP'].values[i: i + frame_size]
        RSRQ = df['RSRQ'].values[i: i + frame_size]
        Light = df['Light'].values[i: i + frame_size]
        Mag = df['Mag'].values[i: i + frame_size]
        Acc = df['Acc'].values[i: i + frame_size]
        Sound = df['Sound'].values[i: i + frame_size]
        Proximity = df['Proximity'].values[i: i + frame_size]
        Daytime = df['Daytime'].values[i: i + frame_size]

        # Retrieve the most often used label in this segment
        label = stats.mode(df['label'][i: i + frame_size])[0][0]
        frames.append([RSRP, RSRQ, Light, Mag, Acc, Sound, Proximity, Daytime])
        labels.append(label)

    # Bring the segments into a better shape
    frames = np.asarray(frames).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)

    return frames, labels

def parallel_CNN_LSTM(input_shape, n_outputs):
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

def drawConfusionMatrix(myModel, X_test, y_test):
    class_labels = ['Outdoor', 'Indoor']
    # Measure the time it takes to predict a single sample for multiple steps
    prediction_times = []
    for _ in range(10):  # Replace 10 with the number of steps you want to measure
        start_time = time.time()
        predict_x = myModel.predict(X_test)
        end_time = time.time()

        # Calculate the prediction time for each step
        prediction_time = end_time - start_time
        prediction_times.append(prediction_time)

    # Calculate mean, median, and standard deviation of prediction times
    mean_prediction_time = np.mean(prediction_times)
    median_prediction_time = np.median(prediction_times)
    std_prediction_time = np.std(prediction_times)

    print('Mean Prediction Time:', mean_prediction_time, 'seconds')
    print('Median Prediction Time:', median_prediction_time, 'seconds')
    print('Standard Deviation of Prediction Time:', std_prediction_time, 'seconds')

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

def evalmodellite(trained_Model_Lite_Path, X, y):
    # Load the TensorFlow Lite model.
    interpreter = tf.lite.Interpreter(trained_Model_Lite_Path)
    interpreter.allocate_tensors()

    # Get the input and output details.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()



    # Load the test data.
    X_test = X 
    y_test = y 



    # Test the model on the test data.
    correct = 0
    y_pred = []
    for i in range(len(X_test)):
        # Preprocess the input data.
        input_data = X_test[i].astype(np.float32)
        input_data = np.expand_dims(input_data, axis=0)

        # Set the input tensor.
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference.
        interpreter.invoke()

        # Get the output tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.argmax(output_data)

        # Check if the prediction is correct.
        if prediction == (y_test[i]):
            correct += 1
        y_pred.append(prediction)

    accuracy = correct / len(X_test)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

def evallite(X, y):
    
    dataset_Path = '/home/bilz/IODNET/Datasets/'
    model_Path = '/home/bilz/IODNET/models/'
    dataset_Path_Full = dataset_Path + 'validatingData.csv'
    trained_Model_Lite_Path = model_Path + 'bd_KIODNet_CLP_V1_W_6.tflite'
    DATA_PATH = dataset_Path_Full
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

    # Load the trained lite model
    #modelLite = loadModelLite(model_Path + 'bd_KIODNet_CLP_V1_W_6.tflite')

    interpreter = tf.lite.Interpreter(model_path=trained_Model_Lite_Path)
    interpreter.allocate_tensors()

    # Get input and output details.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Print model summary.
    print('Input shape:', input_details[0]['shape'])
    print('Input type:', input_details[0]['dtype'])
    print('Output shape:', output_details[0]['shape'])
    print('Output type:', output_details[0]['dtype'])

    evalmodellite(trained_Model_Lite_Path,X, y)


# Main code
if __name__ == "__main__":
    dataset_Path = '/home/bilz/IODNET/Datasets/'
    model_Path = '/home/bilz/IODNET/models/'
    dataset_Path_Full = dataset_Path + 'validatingData.csv'
    DATA_PATH = dataset_Path_Full
    outputs_Number = 2

    # Preprocess data
    df, data = preprocessData(DATA_PATH)

    # Balance the data
    balanced_data = balanceData(df, data, True)

    # Encode the data
    encoded_data = encodedData(balanced_data)

    # Standardize the data
    scaled_X, X, y = standardizeData(encoded_data)

    # Get framed data
    scaled_X, X, y = framedData(scaled_X, X, y)

    # Load the trained model
    trained_model = load_model(model_Path + "bd_KIODNet_CLP_V1_W_6.h5")

    # Evaluate the model on the validation dataset
    predict_x = trained_model.predict(X)
    y_pred = np.argmax(predict_x, axis=1)

    if len(y.shape) > 1 and y.shape[1] > 1:
        y_true = np.argmax(y, axis=1)
    else:
        y_true = y

    test_acc = accuracy_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Accuracy: {test_acc}")
    print(f"F1-score: {test_f1}")
    

    # Display confusion matrix and ROC curve
    drawConfusionMatrix(trained_model, X, y)

    evallite(X, y)
