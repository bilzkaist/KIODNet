import tensorflow as tf
import numpy as np
import pandas as pd
import time


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




    # Convert y_test from one-hot encoded array to 1D array of labels.
    y_test_labels = np.argmax(np.expand_dims(y_test, axis=0), axis=1)





def loadModelLite(model_Path, filename='bd_model.tflite'):
    
    try:
        # load the model from a file
        test_load_lite_model = tf.lite.Interpreter(model_Path + filename)
        print("Lite Version of Trained Model saved and loaded successfully !!!")
        return test_load_lite_model 

    except:
        print("Lite Version of Trained Model is not saved successfully !!!")
        return False


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



# Main code
if __name__ == "__main__":
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

