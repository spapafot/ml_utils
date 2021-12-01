import os
import PIL
from pathlib import Path
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks
import tensorflow_hub as hub


def remove_corrupted_images(filepath, filetype):
    path = Path(filepath).rglob(f"*.{filetype}")
    for img in path:
        try:
            img = PIL.Image.open(img)
        except PIL.UnidentifiedImageError:
            os.remove(img)


def create_model(model_url, num_classes, img_size):
    feature_extraction_layer = hub.KerasLayer(model_url, trainable=False, input_shape=img_size + (3,))
    model = models.Sequential()
    model.add(feature_extraction_layer)
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


def create_tensorboard_callback(dir_name, model_name):
    log_dir = dir_name + '/' + model_name + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir)
    return tensorboard_callback


def create_windows_from_array(x, window_size, horizon):

    step = np.expand_dims(np.arange(window_size + horizon),axis=0)
    index = step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T
    windowed_array = x[index]
    windows, labels = windowed_array[:,:-horizon],windowed_array[:,-horizon:]
    return windows, labels


def create_windows_from_dataframe(df, window_size, y_label):

    df_window = df.copy()
    for i in range(window_size):
        df_window[f'{y_label} + {i+1}'] = df_window[y_label].shift(periods=i+1)

    X = df_window.dropna().drop(y_label, axis=1).astype('float32')
    y = df_window.dropna()[y_label]
    return X, y


def create_model_checkpoint(model_name, save_path):

    return callbacks.ModelCheckpoint(filepath=os.path.join(save_path,model_name), save_best_only=True)


def future_predictions(values, model, window_size, future):
    future_forecast = []
    last_window = values[-window_size:]
    for _ in range(future):
        prediction = model.predict(tf.expand_dims(last_window, axis=0))
        future_forecast.append(prediction)
        last_window = np.append(last_window, prediction)[-window_size:]
    return future_forecast


def load_image(filename, image_shape):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, [image_shape, image_shape])
    img = img/255
    img = tf.expand_dims(img, axis=0)
    return img