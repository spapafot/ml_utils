from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks
import datetime
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import os


def create_model_checkpoint(model_name, save_path):

    return callbacks.ModelCheckpoint(filepath=os.path.join(save_path,model_name), save_best_only=True)


def create_tensorboard_callback(dir_name, model_name):

    log_dir = dir_name + '/' + model_name + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir)
    return tensorboard_callback


def create_model(model_url, num_classes, img_size):

    feature_extraction_layer = hub.KerasLayer(model_url, trainable=False, input_shape=img_size + (3,))
    model = models.Sequential()
    model.add(feature_extraction_layer)
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


def future_predictions(values, model, window_size, future):

    future_forecast = []
    last_window = values[-window_size:]
    for _ in range(future):
        prediction = model.predict(tf.expand_dims(last_window, axis=0))
        future_forecast.append(prediction)
        last_window = np.append(last_window, prediction)[-window_size:]
    return future_forecast