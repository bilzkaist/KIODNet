import tensorflow as tf
from tensorflow.keras.models import Model

def saveModelLite(trained_model, model_Path, filename='bd_model.tflite'):
    converter = tf.lite.TFLiteConverter.from_keras_model(trained_model)
    try:
        # Convert the model to a TFLite model    
        tflite_model = converter.convert()
    except:
        # Convert the model to a TFLite model with optimization options and Select TF ops
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()
    
    # Save the TFLite model to a file
    with open(model_Path + filename, 'wb') as f:
        f.write(tflite_model)

    try:
        # load the model from a file
        test_load_lite_model = tf.lite.Interpreter(model_Path + filename)
        print("Lite Version of Trained Model saved and loaded successfully !!!")
        return test_load_lite_model 

    except:
        print("Lite Version of Trained Model is not saved successfully !!!")
        return False





# Main code
if __name__ == "__main__":
    dataset_Path = '/home/bilz/IODNET/Datasets/'
    model_Path = '/home/bilz/IODNET/models/'
    dataset_Path_Full = dataset_Path + 'validatingData.csv'
    DATA_PATH = dataset_Path_Full
    outputs_Number = 2

    # Load the trained model
    trained_model = tf.keras.models.load_model(model_Path + 'bd_KIODNet_CLP_V1_W_6.h5')

    # Convert and Save the Lite version of the trained model
    saveModelLite(trained_model, model_Path + 'bd_KIODNet_CLP_V1_W_6.tflite')

