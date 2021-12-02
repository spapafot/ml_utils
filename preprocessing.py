import os
import PIL
from pathlib import Path
import numpy as np
import tensorflow as tf


def remove_corrupted_images(filepath, filetype):

    path = Path(filepath).rglob(f"*.{filetype}")
    for img in path:
        try:
            img = PIL.Image.open(img)
        except PIL.UnidentifiedImageError:
            os.remove(img)


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


def load_image(filename, image_shape):

    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, [image_shape, image_shape])
    img = img/255
    img = tf.expand_dims(img, axis=0)
    return img