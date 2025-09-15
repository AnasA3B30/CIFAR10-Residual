import tensorflow as tf
import numpy as np



class ResidualConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3,3), pooling=True, pool_size=(2,2), dropout_rate=0.1):
        super().__init__()
        self.pooling = pooling

        # conv path
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same",
                                            activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.drop1 = tf.keras.layers.Dropout(dropout_rate)

        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same",
                                            activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.drop2 = tf.keras.layers.Dropout(dropout_rate)

        self.conv3 = tf.keras.layers.Conv2D(filters, kernel_size, padding="same",
                                            activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.drop3 = tf.keras.layers.Dropout(dropout_rate)

        # projection if input channels ≠ filters
        self.proj = tf.keras.layers.Conv2D(filters, (1,1), padding="same")

        # pooling
        self.pool = tf.keras.layers.MaxPooling2D(pool_size)

    def call(self, inputs, training=False):
        shortcut = inputs

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.drop1(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.drop2(x, training=training)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.drop3(x, training=training)

        # project if needed
        if shortcut.shape[-1] != x.shape[-1]:
            shortcut = self.proj(shortcut)

        # add skip
        x = tf.keras.layers.add([x, shortcut])
        x = tf.nn.relu(x)

        if self.pooling:
            x = self.pool(x)

        return x
    


class DenseBlock(tf.keras.layers.Layer):
    """Customizable dense block with BN, dropout, L2 reg, and flexible activation."""
    def __init__(self, n_neurons=256, activation_fn='relu',
                 batch_norm=True, is_dropout=True, dropout_rate=0.35,
                 is_l2=True, l2_reg=1e-4):
        super(DenseBlock, self).__init__()

        # Store settings
        self.n_neurons = n_neurons
        self.activation_fn = activation_fn
        self.batch_norm = batch_norm
        self.is_dropout = is_dropout
        self.dropout_rate = dropout_rate
        self.is_l2 = is_l2
        self.l2_reg = l2_reg

        # Disable L2 for softmax (best practice)
        if self.activation_fn == 'softmax':
            self.is_l2 = False

        # Dense (no activation here)
        if self.is_l2:
            self.dense = tf.keras.layers.Dense(
                self.n_neurons,
                activation=None,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
            )
        else:
            self.dense = tf.keras.layers.Dense(
                self.n_neurons,
                activation=None
            )

        # Optional BatchNorm
        if self.batch_norm:
            self.bn = tf.keras.layers.BatchNormalization()

        # Separate activation layer (so BN happens before activation)
        self.activation = tf.keras.layers.Activation(self.activation_fn)

        # Optional Dropout
        if self.is_dropout:
            self.drop = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False):
        x = self.dense(inputs)

        if self.batch_norm:
            x = self.bn(x, training=training)

        x = self.activation(x)

        if self.is_dropout:
            x = self.drop(x, training=training)

        return x
    


class ResidualDenseBlock(tf.keras.layers.Layer):
    def __init__(self, n_neurons=256, activation_fn='relu', dropout_rate=0.35, l2_reg=1e-4):
        super().__init__()
        self.dense = tf.keras.layers.Dense(n_neurons, activation=None,
                                           kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.bn = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation(activation_fn)
        self.drop = tf.keras.layers.Dropout(dropout_rate)

        # projection if needed
        self.proj = tf.keras.layers.Dense(n_neurons, activation=None)

    def call(self, inputs, training=False):
        shortcut = inputs
        x = self.dense(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)
        x = self.drop(x, training=training)

        # project if mismatch
        if shortcut.shape[-1] != x.shape[-1]:
            shortcut = self.proj(shortcut)

        x = tf.keras.layers.add([x, shortcut])
        return tf.nn.relu(x)


class ConvNN(tf.keras.Model):

    def __init__(self, input_shape, num_classes):
        super(ConvNN, self).__init__()

        # augmentation block
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])

        # convolution blocks
        self.block1 = ResidualConvBlock(filters=32, kernel_size=(3,3))
        self.block2 = ResidualConvBlock(filters=64, kernel_size=(3,3))
        self.block3 = ResidualConvBlock(filters=128, kernel_size=(3,3))
        self.block4 = ResidualConvBlock(filters=256, kernel_size=(3,3), pooling=False, dropout_rate=0.5)

        # Flatten or GlobalMaxPooling
        self.flatten = tf.keras.layers.GlobalAveragePooling2D()

        # classification
        self.out_layer = DenseBlock(10, activation_fn='softmax', batch_norm=False, is_dropout=False)

    def call(self, inputs, training=False):

      # augmentation
      x = self.data_augmentation(inputs, training=training)

      # conv blocks
      x = self.block1(x)
      x = self.block2(x)
      x = self.block3(x)
      x = self.block4(x)

      # flatten or pool
      x = self.flatten(x)

      # return last layer result
      return self.out_layer(x)
    
    