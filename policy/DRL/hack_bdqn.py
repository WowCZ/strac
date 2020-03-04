import mxnet as mx
import mxnet.gluon as gl
import mxnet.ndarray as nd
import numpy as np
import copy

CTX = mx.cpu()


def softplus(x):
    return nd.log(1. + nd.exp(x))


def log_gaussian(x, mu, sigma):
    return nd.sum(-0.5 * np.log(2.0 * np.pi) - nd.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2))


class BBBNetwork(gl.nn.Block):
    def __init__(self, input_dim, output_dim, n_hidden_layers, hidden_units,
                 n_samples=1, sigma_prior=0.05, sigma_eps=1., map_mode=False, **kwargs):
        gl.nn.Block.__init__(self, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_layers = n_hidden_layers
        self.hidden_units = hidden_units
        self.n_samples = n_samples
        self.sigma_prior = sigma_prior
        self.sigma_eps = sigma_eps
        self.map_mode = map_mode

        with self.name_scope():
            self.weight_mus = []
            self.weight_rhos = []
            self.bias_mus = []
            self.bias_rhos = []
            for i in range(self.n_hidden_layers):
                if i == 0:
                    self.weight_mus.append(self.params.get(name='weight_mu_0', shape=(self.input_dim, self.hidden_units[0])))
                    self.weight_rhos.append(self.params.get(name='weight_rho_0', shape=(self.input_dim, self.hidden_units[0])))
                    self.bias_mus.append(self.params.get(name='bias_mu_0', shape=(self.hidden_units[0], )))
                    self.bias_rhos.append(self.params.get(name='bias_rho_0', shape=(self.hidden_units[0], )))
                else:
                    self.weight_mus.append(self.params.get(
                        name='weight_mu_{}'.format(i), shape=(self.hidden_units[i - 1], self.hidden_units[i])))
                    self.weight_rhos.append(self.params.get(
                        name='weight_rho_{}'.format(i), shape=(self.hidden_units[i - 1], self.hidden_units[i])))
                    self.bias_mus.append(self.params.get(name='bias_mu_{}'.format(i), shape=(self.hidden_units[i], )))
                    self.bias_rhos.append(self.params.get(name='bias_rho_{}'.format(i), shape=(self.hidden_units[i], )))
            self.weight_mus.append(self.params.get(name='weight_mu_out', shape=(self.hidden_units[-1], self.output_dim)))
            self.weight_rhos.append(self.params.get(name='weight_rho_out', shape=(self.hidden_units[-1], self.output_dim)))
            self.bias_mus.append(self.params.get(name='bias_mu_out', shape=(self.output_dim, )))
            self.bias_rhos.append(self.params.get(name='bias_rho_out', shape=(self.output_dim, )))

    def get_sample(self, mu, rho, is_target=False):
        if self.map_mode and is_target:
            # print 'map_mode.'
            return mu
        epsilon = nd.random_normal(loc=0, scale=self.sigma_eps, shape=mu.shape)
        sigma = softplus(rho)
        return mu + sigma * epsilon

    def forward(self, inputs, is_target=False):
        result = None
        loss = 0.
        for _ in range(self.n_samples):
            tmp = inputs

            weights = []
            biases = []
            for i in range(len(self.weight_mus)):
                weights.append(self.get_sample(
                    mu=self.weight_mus[i].data(), rho=self.weight_rhos[i].data(), is_target=is_target))
                biases.append(self.get_sample(mu=self.bias_mus[i].data(), rho=self.bias_rhos[i].data(), is_target=is_target))
                loss = loss + log_gaussian(
                    x=weights[-1], mu=self.weight_mus[i].data(), sigma=softplus(self.weight_rhos[i].data()))
                loss = loss + log_gaussian(x=biases[-1], mu=self.bias_mus[i].data(), sigma=softplus(self.bias_rhos[i].data()))
                loss = loss - log_gaussian(x=weights[-1], mu=0., sigma=self.sigma_prior)
                loss = loss - log_gaussian(x=weights[-1], mu=0., sigma=self.sigma_prior)
            for i in range(len(weights)):
                tmp = nd.dot(tmp, weights[i]) + biases[i]
                if i != len(weights) - 1:
                    tmp = nd.relu(tmp)
            if result is None:
                result = nd.zeros_like(tmp)
            result = result + tmp
        result = result / float(self.n_samples)
        loss = loss / float(self.n_samples)
        return result, loss


class DeepQNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau,
                 num_actor_vars, architecture='duel', h1_size=130, h2_size=50,
                 n_samples=1, batch_size=32, sigma_prior=0.5, n_batches=1000.0,
                 stddev_var_mu=0.01, stddev_var_logsigma=0.1, mean_log_sigma=0.1, importance_sampling=False,
                 alpha_divergence=False, alpha=1.0, sigma_eps=1.0, map_mode=None, **kwargs):
        # self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.architecture = architecture
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.sigma_prior = sigma_prior
        self.sigma_eps = sigma_eps
        self.map_mode = map_mode

        self.qnet = self.create_ddq_network(prefix='qnet_')
        self.target = self.create_ddq_network(prefix='target_')

        self.trainer = gl.Trainer(params=self.qnet.collect_params(),
                                  optimizer='adam',
                                  optimizer_params=dict(learning_rate=self.learning_rate))
        self.count_training = 0

    def create_ddq_network(self, prefix=''):
        network = BBBNetwork(
            input_dim=self.s_dim, output_dim=self.a_dim, n_hidden_layers=2, hidden_units=(self.h1_size, self.h2_size),
            n_samples=self.n_samples, sigma_prior=self.sigma_prior, sigma_eps=self.sigma_eps,
            map_mode=self.map_mode, prefix=prefix)
        network.initialize()
        return network

    def train(self, inputs, action, sampled_q):
        inputs = copy.deepcopy(inputs)
        action = copy.deepcopy(action)
        sampled_q = copy.deepcopy(sampled_q)

        inputs = nd.array(inputs, ctx=CTX)
        action = nd.array(action, ctx=CTX)
        sampled_q = nd.array(sampled_q, ctx=CTX)
        sampled_q = sampled_q.reshape(shape=(sampled_q.shape[0],))

        self.count_training += 1
        with mx.autograd.record():
            outputs, loss = self.qnet(inputs)
            loss = loss * float(self.batch_size) / 1000.
            td_error = nd.sum(data=outputs * action, axis=1) - sampled_q
            for i in range(self.batch_size):
                if nd.abs(td_error[i]) < 1.0:
                    loss = loss + 0.5 * nd.square(td_error[i])
                else:
                    loss = loss + nd.abs(td_error[i]) - 0.5
        loss.backward()
        self.trainer.step(batch_size=self.batch_size)

    def predict(self, inputs):
        return self.qnet(nd.array(inputs, ctx=CTX))[0].asnumpy()

    def predict_target(self, inputs):
        return self.target(nd.array(inputs, ctx=CTX), True)[0].asnumpy()

    def update_target_network(self):
        param_list_qnet = []
        param_list_target = []
        for key, value in self.qnet.collect_params().items():
            param_list_qnet.append(value)
        for key, value in self.target.collect_params().items():
            param_list_target.append(value)
        assert len(param_list_qnet) == len(param_list_target)

        for i in range(len(param_list_qnet)):
            assert (param_list_target[i].name.strip('target') ==
                    param_list_qnet[i].name.strip('qnet'))
            param_list_target[i].set_data(
                param_list_target[i].data() * (1. - self.tau) +
                param_list_qnet[i].data() * self.tau
            )

    def copy_qnet_to_target(self):
        param_list_qnet = []
        param_list_target = []
        for key, value in self.qnet.collect_params().items():
            param_list_qnet.append(value)
        for key, value in self.target.collect_params().items():
            param_list_target.append(value)
        assert len(param_list_qnet) == len(param_list_target)

        for i in range(len(param_list_qnet)):
            assert (param_list_target[i].name.strip('target') ==
                    param_list_qnet[i].name.strip('qnet'))
            param_list_target[i].set_data(param_list_qnet[i].data())

    def load_network(self, load_filename):
        try:
            self.qnet.load_params(filename=load_filename + '_qnet', ctx=CTX)
            print "Successfully loaded:", load_filename + '_qnet'
        except:
            print "Could not find old network weights(qnet)"
            print load_filename

        try:
            self.target.load_params(filename=load_filename + '_target', ctx=CTX)
            print "Successfully loaded:", load_filename + '_target'
        except:
            print "Could not find old network weights(target)"
            print load_filename

        try:
            self.trainer.step(1, ignore_stale_grad=True)
            self.trainer.load_states(fname=load_filename + '_trainer')
            print "Successfully loaded:", load_filename + '_trainer'
        except:
            print "Could not find old network weights(trainer)"
            print load_filename

    def save_network(self, save_filename):
        print 'Saving deepq-network...'
        self.qnet.save_params(filename=save_filename + '_qnet')
        self.target.save_params(filename=save_filename + '_target')
        self.trainer.save_states(fname=save_filename + '_trainer')