import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

# TIMESERIES NBEATS PAPER

class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(self, input_size: int, theta_size: int, horizon: int, n_neurons: int, n_layers: int, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers

        self.hidden = [layers.Dense(self.n_neurons, activation='relu') for _ in range(self.n_layers)]
        self.theta_layer = layers.Dense(self.theta_size, activation='linear')

    def call(self, inputs):  # Defines the computation from inputs to outputs
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        theta = self.theta_layer(x)
        backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]

        return backcast, forecast

# IMPLEMENTATION

WINDOW_SIZE = 7
HORIZON = 1
N_EPOCHS = 5000
N_LAYERS = 4
N_NEURONS = 512
N_STACKS = 30
INPUT_SIZE = WINDOW_SIZE * HORIZON
THETA_SIZE = INPUT_SIZE + HORIZON


initial = NBeatsBlock(input_size=INPUT_SIZE,
                      theta_size=THETA_SIZE,
                      horizon=HORIZON,
                      n_neurons=N_NEURONS,
                      n_layers=N_LAYERS,
                      name='Initial_Block')


def create_model():
    input = layers.Input(shape=(INPUT_SIZE), name='Stack_Input')
    residuals, forecast = initial(input)
    for i,_ in enumerate(range(N_STACKS-1)):
        backcast, block_forecast = NBeatsBlock(input_size=INPUT_SIZE,
                                               theta_size=THETA_SIZE,
                                               horizon=HORIZON,
                                               n_neurons=N_NEURONS,
                                               n_layers=N_LAYERS,
                                               name=f'NBEATS_{i}')(residuals)
        residuals = layers.subtract([residuals,backcast], name=f'Subtract_{i}')
        forecast = layers.add([forecast, block_forecast], name=f'Add_{i}')
    return models.Model(input, forecast)