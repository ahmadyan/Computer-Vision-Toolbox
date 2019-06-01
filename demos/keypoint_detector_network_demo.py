import keras
import numpy as np

from keras.utils import plot_model


def load_model(filename):
    with open(filename, "r") as json_file:
        json_model = json_file.read()

    model = keras.models.model_from_json(json_model)
   # model.load_weights(filename + "_weights.hdf5")

    return model

model = load_model('.')
print(model)
plot_model(model, to_file='model.png')