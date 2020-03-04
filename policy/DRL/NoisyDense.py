import numpy as np
import mxnet as mx
import mxnet.ndarray as F
import mxnet.gluon as gl
from mxnet import autograd
import math

class NoisyDense(gl.Block):
    def __init__(self, in_units, units, use_bias=True, activation='linear', std_init=0.1, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.in_units = in_units
        self.units = units
        self.std_init = std_init
        self.activation = activation
        self.use_bias = use_bias


        with self.name_scope():
            self.w_mu = self.params.get('weight_mu', shape=(units, in_units), init=mx.init.Xavier())
            self.w_sigma = self.params.get('weight_sigma', shape=(units, in_units), init=mx.init.Xavier())
            self.b_mu = self.params.get('bias_mu', shape=(units,), init=mx.init.Zero())
            self.b_sigma = self.params.get('bias_sigma', shape=(units,), init=mx.init.Zero())

            self.w_mu.initialize()
            self.w_sigma.initialize()
            self.b_mu.initialize()
            self.b_sigma.initialize()


        self.reset_noise()
        self.reset_parameters()

    def forward(self, input, training=True):
        if self.activation != 'linear':
            if training:
                if self.use_bias:
                    return F.Activation(F.FullyConnected(input, self.w_mu.data() + self.w_sigma.data() * self.w_epsilon,
                                    self.b_mu.data() + self.b_sigma.data() * self.b_epsilon, num_hidden=self.units), self.activation)
                else:
                    return F.Activation(F.FullyConnected(input, self.w_mu.data() + self.w_sigma.data() * self.w_epsilon, no_bias=True, num_hidden=self.units), self.activation)
            else:
                if self.use_bias:
                    return F.Activation(F.FullyConnected(input, self.w_mu.data(), self.b_mu.data(), num_hidden=self.units), self.activation)
                else:
                    return F.Activation(F.FullyConnected(input, self.w_mu.data(), no_bias=True, num_hidden=self.units), self.activation)
        else:
            if training:
                if self.use_bias:
                    return F.FullyConnected(input, self.w_mu.data() + self.w_sigma.data() * self.w_epsilon,
                                    self.b_mu.data() + self.b_sigma.data() * self.b_epsilon, num_hidden=self.units)
                else:
                    return F.FullyConnected(input, self.w_mu.data() + self.w_sigma.data() * self.w_epsilon,
                                    no_bias=True, num_hidden=self.units)
            else:
                if self.use_bias:
                    return F.FullyConnected(input, self.w_mu.data(), self.b_mu.data(), num_hidden=self.units)
                else:
                    return F.FullyConnected(input, self.w_mu.data(), no_bias=True, num_hidden=self.units)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_units)
        self.w_mu.set_data(F.random.uniform(-mu_range, mu_range, shape=(self.units, self.in_units)))
        self.w_sigma.set_data((self.std_init / math.sqrt(self.in_units))*F.ones(shape=(self.units, self.in_units)))
        self.b_mu.set_data(F.random.uniform(-mu_range, mu_range, shape=(self.units,)))
        self.b_sigma.set_data((self.std_init / math.sqrt(self.units))*F.ones(shape=(self.units,)))

    def _scale_noise(self, size):
        x = F.array(np.random.randn(size))
        return x.sign()*(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_units)
        epsilon_out = self._scale_noise(self.units)
        self.w_epsilon = (epsilon_out.expand_dims(axis=1)*epsilon_in.expand_dims(axis=0)).copy()
        self.b_epsilon = epsilon_out.copy()
