# This is a modified version of the top section of https://github.com/AprilYuge/scAAnet/blob/main/scAAnet/network.py to allow model saving and loading as h5ad file 
# The framework was adapted from https://github.com/theislab/dca/blob/master/dca/network.py

import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Lambda


ColwiseMultLayer = Lambda(lambda l: l[0]*tf.reshape(l[1], (-1,1)), name="mean")
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)

class DispLayer(tf.keras.layers.Layer):
    def __init__(self, units=2000, **kwargs):
        super(DispLayer, self).__init__(**kwargs)
        self.units = units
        self.w = tf.Variable(initial_value=tf.random.normal([units]), trainable=True, name='dispersion')
        
    def call(self, inputs):
        return DispAct(self.w)
    
    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(self.units,),
            initializer='random_normal',
            trainable=True,
            name='dispersion')
        super(DispLayer, self).build(input_shape)
        
    def get_config(self):
        config = super(DispLayer, self).get_config()
        config.update({
            'units': self.units,
        })
        return config

class ZFixedLayer(tf.keras.layers.Layer):
    def __init__(self, dim_latent_space, name='z_fixed', **kwargs):
        super(ZFixedLayer, self).__init__(name=name, **kwargs)
        self.dim_latent_space = dim_latent_space
        self.w = tf.Variable(initial_value=tf.convert_to_tensor(create_z_fixed(dim_latent_space), tf.keras.backend.floatx()), 
                             trainable=True, name='z_fixed')
        #print(f'what we are lookgin for {self.w}')

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.dim_latent_space),
            initializer=tf.keras.initializers.Constant(
                create_z_fixed(self.dim_latent_space)),
            trainable=True,
            name='z_fixed')
        super(ZFixedLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.w)
        
    def get_config(self):
        config = super(ZFixedLayer, self).get_config()
        config.update({
            'dim_latent_space': self.dim_latent_space,
        })
        return config




# We borrowed the following function from https://github.com/bmda-unibas/DeepArchetypeAnalysis/blob/master/AT_lib/lib_at.py
def create_z_fixed(dim_latent_space):
    """
    Creates Coordinates of the Simplex spanned by the Archetypes.
    The simplex will have its centroid at 0.
    The sum of the vertices will be zero.
    The distance of each vertex from the origin will be 1.
    The length of each edge will be constant.
    The dot product of the vectors defining any two vertices will be - 1 / M.
    This also means the angle subtended by the vectors from the origin
    to any two distinct vertices will be arccos ( - 1 / M ).
    :param dim_latent_space:
    :return:
    """

    z_fixed_t = np.zeros([dim_latent_space, dim_latent_space + 1])

    for k in range(0, dim_latent_space):
        s = 0.0
        for i in range(0, k):
            s = s + z_fixed_t[i, k] ** 2

        z_fixed_t[k, k] = np.sqrt(1.0 - s)

        for j in range(k + 1, dim_latent_space + 1):
            s = 0.0
            for i in range(0, k):
                s = s + z_fixed_t[i, k] * z_fixed_t[i, j]

            z_fixed_t[k, j] = (-1.0 / float(dim_latent_space) - s) / z_fixed_t[k, k]
            z_fixed = np.transpose(z_fixed_t)
    return z_fixed
