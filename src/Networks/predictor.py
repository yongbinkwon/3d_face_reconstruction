import keras
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from keras.utils.generic_utils import CustomObjectScope

from Networks import mobilenet_v2



class MobilenetPosPredictor():
    def __init__(self, resolution_inp=256, resolution_op=256):
        # -- hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp * 1.1  # this *1.1 is some what of amystery..
        self.model = None

        # set tensorflow session GPU usage
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        set_session(sess)

    def restore(self, model_path):
        print(model_path)
        with CustomObjectScope(
                {'relu6': mobilenet_v2.relu6}):  # ,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D
            self.model = keras.models.load_model(model_path)

    def predict(self, image):
        x = image[np.newaxis, :, :, :]
        pos = self.model.predict(x=x)
        pos = np.squeeze(pos)
        return pos * self.MaxPos

    def predict_batch(self, images):
        raise NotImplementedError


class PosPrediction_6_keras():
    def __init__(self, resolution_inp=256, resolution_op=256):
        # -- hyper settings
        self.resolution_inp = resolution_inp
        self.resolution_op = resolution_op
        self.MaxPos = resolution_inp * 1.1  # this *1.1 is some what of amystery..
        self.model = None

        # set tensorflow session GPU usage
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        set_session(sess)

    def restore(self, model_path):
        self.model = keras.models.load_model(model_path)

    def predict(self, image):
        pos = self.model.predict(x=image[np.newaxis, :, :, :])
        pos = np.squeeze(pos)
        return pos * self.MaxPos

    def predict_batch(self, images):
        raise NotImplementedError
