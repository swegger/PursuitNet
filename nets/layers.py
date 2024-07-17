import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, SimpleRNN, Dense, Lambda
from abc import abstractmethod
from typing import Union

class pursuitLayer(Layer):
    """A custom, 2 cell layer for control of velocity in x and y directions"""
    def __init__(self):
        super(pursuitLayer, self).__init__(name='targetVelocity')
        self.units = 2

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
            shape=[int(input_shape[-1]),
            self.units])

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)


class MTlayer(Layer):
    """A custom input class that converts velocities into a population of MT neuron-like responses.

    """

    def __init__(self, units=32):
        super(MTlayer, self).__init__()
        self.units = units
        #self.amp = np.random.uniform(10,100,self.units)
        #self.baseline = np.random.uniform(1,10,self.units)
        #self.theta.pref = np.random.uniform(-180,180,self.units)
        #self.theta.sig = np.random.uniform(30,60,self.units)
        #self.speed.pref = 2**np.random.uniform(-1,8,self.units)
        #self.speed.sig = np.random.uniform(1.5,1.8,self.units)

    def build(self, input_shape):
        self.amp = tf.Variable(np.random.uniform(1,1,self.units).astype(np.float32), trainable=False)
        self.baseline = tf.Variable(np.random.uniform(0,0,self.units).astype(np.float32), trainable=False)
        self.thetapref = tf.Variable(np.random.uniform(-180,180,self.units).astype(np.float32), trainable=False)
        self.thetasig = tf.Variable(np.random.uniform(30,60,self.units).astype(np.float32), trainable=False)
        self.speedpref = tf.Variable(2**np.random.uniform(-1,8,self.units).astype(np.float32), trainable=False)
        self.speedsig = tf.Variable(np.random.uniform(1.5,1.8,self.units).astype(np.float32), trainable=False)
        self.latency = 60


    def get_config(self):
        return {"units": self.units}

    def call(self, inputs):
        latency = self.latency
        tmp = tf.zeros_like(inputs)
        inputs2 = tf.concat([tmp[:,:latency,:],inputs[:,latency:,:]],axis=1)
        speed = tf.math.sqrt(tf.reduce_sum(tf.math.square(inputs2[:,:,:2]),axis=-1))
        theta = tf.math.atan2(inputs2[:,:,0],inputs2[:,:,1])*180/np.pi
        theta = tf.expand_dims(theta,axis=2)
        speed = tf.expand_dims(speed,axis=2)
        coh = tf.expand_dims(inputs2[:,:,2]/100,axis=2)
        amp = tf.experimental.numpy.empty_like(coh)
        activationD = tf.experimental.numpy.empty_like(theta)
        activationS = tf.experimental.numpy.empty_like(speed)
        for k in range(self.units):
            amp = tf.concat([amp,
                self.amp[k]*coh],axis=2)
            activationD = tf.concat([activationD,
                tf.math.exp( -tf.math.square(theta - self.thetapref[k])
                /( tf.math.square(self.thetasig[k])/2 ))
                ],axis=2)
            activationS = tf.concat([activationS,
                tf.math.exp( -tf.math.square( tf.experimental.numpy.log2(speed/self.speedpref[k]) )
                / ( tf.math.square(self.speedsig[k])/2 ))
                ],axis=2)
        #return self.amp*tf.keras.layers.Multiply()([directionTuning(inputs),speedTuning(inputs)]) + self.baseline

        amp = amp[:,:,1:]
        activationD = activationD[:,:,1:]
        activationS = activationS[:,:,1:]

        return amp*tf.math.multiply(activationD, activationS) + self.baseline
