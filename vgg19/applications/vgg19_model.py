"""VGG19 model for Keras.

# Reference

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556) (ICLR 2015)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend
from keras_applications import get_submodules_from_kwargs
from keras_applications import imagenet_utils
from keras_applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape

from tensorflow.python.keras import backend as K
import pdb

preprocess_input = imagenet_utils.preprocess_input


class VGG19(object):

  def __init__(
          self,
          include_top=True,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000,
          **kwargs):
    """Instantiates the VGG19 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)
    self.include_top = include_top
    self.pooling = pooling
    self.weights = weights
    self.backend = backend
    self.layers = layers
    self.classes = classes

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')
    # Block 1
    self.block1_conv1 = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')
    self.block1_conv2 = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')
    self.block1_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')

    # Block 2
    self.block2_conv1 = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')
    self.block2_conv2 = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')
    self.block2_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')

    # Block 3
    self.block3_conv1 = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')
    self.block3_conv2 = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')
    self.block3_conv3 = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')
    self.block3_conv4 = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv4')
    self.block3_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')

    # Block 4
    self.block4_conv1 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')
    self.block4_conv2 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')
    self.block4_conv3 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')
    self.block4_conv4 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv4')
    self.block4_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')

    # Block 5
    self.block5_conv1 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')
    self.block5_conv2 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')
    self.block5_conv3 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')
    self.block5_conv4 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv4')
    self.block5_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')

    if include_top:
        # Classification block
        self.flatten = layers.Flatten(name='flatten')
        self.fc1 = layers.Dense(4096, activation='relu', name='fc1')
        self.fc2 = layers.Dense(4096, activation='relu', name='fc2')
        self.predict = layers.Dense(classes, activation='softmax', name='predictions')
    else:
        if pooling == 'avg':
            self.pool = layers.GlobalAveragePooling2D()
        elif pooling == 'max':
            self.pool = layers.GlobalMaxPooling2D()


  def build(self, features, label, input_shape=None):
    backend = self.backend
    weights = self.weights
    include_top = self.include_top
    pooling = self.pooling
    layers = self.layers
    classes = self.classes
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)
    img_input = layers.Input(tensor=features, shape=input_shape)
    x = self.block1_conv1(img_input)
    x = self.block1_conv2(x)
    x = self.block1_pool(x)

    x = self.block2_conv1(x)
    x = self.block2_conv2(x)
    x = self.block2_pool(x)

    x = self.block3_conv1(x)
    x = self.block3_conv2(x)
    x = self.block3_conv3(x)
    x = self.block3_conv4(x)
    x = self.block3_pool(x)

    x = self.block4_conv1(x)
    x = self.block4_conv2(x)
    x = self.block4_conv3(x)
    x = self.block4_conv4(x)
    x = self.block4_pool(x)

    x = self.block5_conv1(x)
    x = self.block5_conv2(x)
    x = self.block5_conv3(x)
    x = self.block5_conv4(x)
    x = self.block5_pool(x)

    if include_top:
        # Classification block
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.predict(x)
    else:
        if pooling == 'avg':
            x = self.pool(x)
        elif pooling == 'max':
            x = self.pool(x)
    score_array =  K.categorical_crossentropy(x, label)
    return  tf.reduce_mean(score_array)
