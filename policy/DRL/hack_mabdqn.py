import mxnet as mx
import mxnet.gluon as gl
import mxnet.ndarray as nd
import numpy as np
import copy
import collections

CTX = mx.cpu()


BayesDense = collections.namedtuple(typename='BayesDense', field_names=['weight_mu', 'weight_rho', 'bias_mu', 'bias_rho'])


def softplus(x):
    return nd.log(1. + nd.exp(x))


def log_gaussian(x, mu, sigma):
    return nd.sum(-0.5 * np.log(2.0 * np.pi) - nd.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2))


class MultiAgentBBBNetwork(gl.nn.Block):
    def __init__(self, domain_string, input_dim, output_dim,
                 n_hidden_layers, local_hidden_units, global_hidden_units,
                 n_samples=1, sigma_prior=0.05, sigma_eps=1., map_mode=False, **kwargs):
        gl.nn.Block.__init__(self, **kwargs)
        self.domain_string=domain_string
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_hidden_layers = n_hidden_layers
        self.local_hidden_units = local_hidden_units
        self.global_hidden_units = global_hidden_units
        self.n_samples = n_samples
        self.sigma_prior = sigma_prior
        self.sigma_eps = sigma_eps
        self.map_mode = map_mode

        if self.domain_string == 'Laptops11':
            self.slots = ['batteryrating', 'driverange', 'family',
                          'isforbusinesscomputing', 'platform', 'pricerange',
                          'processorclass', 'sysmemory', 'utility', 'warranty',
                          'weightrange']  # output order
            self.slot_dimension = {
                'warranty': (0, 5),
                'family': (130, 136),
                'utility': (136, 145),
                'platform': (145, 152),
                'processorclass': (152, 164),
                'pricerange': (164, 169),
                'batteryrating': (169, 174),
                'sysmemory': (174, 183),
                'weightrange': (183, 188),
                'isforbusinesscomputing': (188, 192),
                'driverange': (192, 197)
            }
            self.global_dimension = [(5, 130), (197, 257)]
            self.input_dimension = 257
            self.global_input_dimension = 257 - 197 + 125
        elif self.domain_string == 'SFRestaurants':
            self.slots = ['allowedforkids', 'area', 'food', 'goodformeal', 'near', 'pricerange']
            self.slot_dimension = {
                'goodformeal': (0, 6),
                'area': (250, 407),
                'food': (407, 468),
                'allowedforkids': (566, 570),
                'near': (570, 581),
                'pricerange': (581, 586)
            }
            self.global_dimension = [(6, 250), (468, 566), (586, 636)]
            self.input_dimension = 636
            self.global_input_dimension = (636 - 586) + (566 - 468) + (250 - 6)
        elif self.domain_string == 'CamRestaurants':
            self.slots = ['area', 'food', 'pricerange']
            self.slot_dimension = {
                'food': (0, 93),
                'pricerange': (93, 98),
                'area': (213, 220)
            }
            self.global_dimension = [(98, 213), (220, 268)]
            self.input_dimension = 268
            self.global_input_dimension = (213 - 98) + (268 - 220)
        else:
            raise ValueError

        with self.name_scope():
            # input -> first hidden layer
            self.local_input_trans = self.get_bayes_dense(
                name='local_input_trans', in_units=22, units=self.local_hidden_units[0])
            self.global_input_trans = self.get_bayes_dense(
                name='global_input_trans', in_units=self.global_input_dimension, units=self.global_hidden_units[0])
            self.local_hidden_trans=[]
            self.global_hidden_trans=[]
            self.local2local_comm = []
            self.local2global_comm = []
            self.global2local_comm = []
            for i in range(self.n_hidden_layers - 1):
                self.local_hidden_trans.append(self.get_bayes_dense(
                    name='local_hidden_trans{}'.format(i),
                    in_units=self.local_hidden_units[i], units=self.local_hidden_units[i + 1]
                ))
                self.global_hidden_trans.append(self.get_bayes_dense(
                    name='global_hidden_trans{}'.format(i),
                    in_units=self.global_hidden_units[i], units=self.global_hidden_units[i + 1]
                ))
                self.local2local_comm.append(self.get_bayes_dense(
                    name='local2local_comm{}'.format(i),
                    in_units=self.local_hidden_units[i], units=self.local_hidden_units[i + 1]
                ))
                self.local2global_comm.append(self.get_bayes_dense(
                    name='local2global_comm{}'.format(i),
                    in_units=self.local_hidden_units[i], units=self.global_hidden_units[i + 1]
                ))
                self.global2local_comm.append(self.get_bayes_dense(
                    name='global2local_comm{}'.format(i),
                    in_units=self.global_hidden_units[i], units=self.local_hidden_units[i + 1]
                ))
            self.local_output_trans=[]
            for i in range(len(self.slots)):
                self.local_output_trans.append(self.get_bayes_dense(
                    name='local_output_trans_slot{}'.format(i),
                    in_units=self.local_hidden_units[-1], units=3
                ))
            self.global_output_trans = self.get_bayes_dense(
                name='global_output_trans',
                in_units=self.global_hidden_units[-1], units=7
            )

    def get_bayes_dense(self, in_units, units, name):
        return BayesDense(
            weight_mu=self.params.get(name=name + '_weight_mu', shape=(in_units, units)),
            weight_rho=self.params.get(name=name + '_weight_rho', shape=(in_units, units)),
            bias_mu=self.params.get(name=name + '_bias_mu', shape=(units, )),
            bias_rho=self.params.get(name=name + 'bias_rho', shape=(units, ))
        )

    def get_sample(self, mu, rho, is_target):
        if self.map_mode and is_target:
            # print 'map_mode.'
            return mu
        epsilon = nd.random_normal(loc=0, scale=self.sigma_eps, shape=mu.shape)
        sigma = softplus(rho)
        return mu + sigma * epsilon

    def bayes_forward(self, x, dense, loss, activation_fn=None, is_target=False):
        weight = self.get_sample(mu=dense.weight_mu.data(), rho=dense.weight_rho.data(), is_target=is_target)
        bias = self.get_sample(mu=dense.bias_mu.data(), rho=dense.bias_rho.data(), is_target=is_target)

        loss = loss + log_gaussian(x=weight, mu=dense.weight_mu.data(), sigma=softplus(dense.weight_rho.data()))
        loss = loss + log_gaussian(x=bias, mu=dense.bias_mu.data(), sigma=softplus(dense.bias_rho.data()))
        loss = loss - log_gaussian(x=weight, mu=0., sigma=self.sigma_prior)
        loss = loss - log_gaussian(x=bias, mu=0., sigma=self.sigma_prior)

        result = nd.dot(x, weight) + bias
        if activation_fn is None:
            return result
        elif activation_fn == 'relu':
            return nd.relu(result)

    def forward(self, input_vec, is_target=False):
        # get inputs for every slot(including global)
        inputs = {}
        for slot in self.slots:
            inputs[slot] = input_vec[:, self.slot_dimension[slot][0]:self.slot_dimension[slot][1]]
        input_global = []
        for seg in self.global_dimension:
            input_global.append(input_vec[:, seg[0]:seg[1]])
        inputs['global'] = nd.concat(*input_global, dim=1)

        # sort the inputs(slots)
        sorted_inputs = []
        for slot in self.slots:
            tmp = inputs[slot][:, :-2].sort(is_ascend=False)
            if tmp.shape[1] < 20:
                tmp = nd.concat(tmp, nd.zeros((tmp.shape[0], 20 - tmp.shape[1]), ctx=CTX), dim=1)
            else:
                tmp = nd.slice_axis(tmp, axis=1, begin=0, end=20)
            sorted_inputs.append(nd.concat(tmp, inputs[slot][:, -2:], dim=1))
        sorted_inputs.append(inputs['global'])

        result = None
        loss = 0.
        for _ in range(self.n_samples):
            layer = [[], ]
            for i in range(len(self.slots)):
                layer[-1].append(self.bayes_forward(
                    x=sorted_inputs[i], dense=self.local_input_trans, loss=loss, activation_fn='relu', is_target=is_target))
            layer[-1].append(self.bayes_forward(
                x=sorted_inputs[-1], dense=self.global_input_trans, loss=loss, activation_fn='relu', is_target=is_target))

            for i in range(self.n_hidden_layers - 1):
                layer.append([])
                for j in range(len(self.slots)):
                    layer[-1].append(self.bayes_forward(
                        x=layer[i][j], dense=self.local_hidden_trans[i], loss=loss, activation_fn='relu', is_target=is_target))
                layer[-1].append(self.bayes_forward(
                    x=layer[i][-1], dense=self.global_hidden_trans[i], loss=loss, activation_fn='relu', is_target=is_target))

                mean_vec = nd.zeros_like(layer[i][0])
                for j in range(len(self.slots)):
                    mean_vec = mean_vec + layer[i][j]
                mean_vec = mean_vec / float(len(self.slots))

                for j in range(len(self.slots)):
                    layer[-1][j] = layer[-1][j] + self.bayes_forward(
                        x=mean_vec, dense=self.local2local_comm[i], loss=loss, activation_fn='relu', is_target=is_target)
                    layer[-1][j] = layer[-1][j] + self.bayes_forward(
                        x=layer[i][-1], dense=self.global2local_comm[i], loss=loss, activation_fn='relu', is_target=is_target)
                for j in range(len(self.slots)):
                    layer[-1][-1] = layer[-1][-1] + self.bayes_forward(
                        x=layer[i][j], dense=self.local2global_comm[i], loss=loss, activation_fn='relu', is_target=is_target)

            tmp = []
            for i in range(len(self.slots)):
                tmp.append(self.bayes_forward(x=layer[-1][i], dense=self.local_output_trans[i], loss=loss, is_target=is_target))
            tmp.append(self.bayes_forward(x=layer[-1][-1], dense=self.global_output_trans, loss=loss, is_target=is_target))

            if result is None:
                result = nd.concat(*tmp, dim=1)
            else:
                result = result + nd.concat(*tmp, dim=1)

        result = result / float(self.n_samples)
        loss = loss / float(self.n_samples)
        return result, loss


class DeepQNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    """

    def __init__(self, domain_string, state_dim, action_dim, learning_rate, tau,
                 num_actor_vars, architecture='duel', h1_size=130, h2_size=50,
                 n_samples=1, batch_size=32, sigma_prior=0.5, n_batches=1000.0,
                 stddev_var_mu=0.01, stddev_var_logsigma=0.1, mean_log_sigma=0.1, importance_sampling=False,
                 alpha_divergence=False, alpha=1.0, sigma_eps=1.0,
                 n_hidden_layers=None, local_hidden_units=None, global_hidden_units=None,
                 map_mode=None):
        self.domain_string=domain_string
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.architecture = architecture
        # self.h1_size = h1_size
        # self.h2_size = h2_size
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.sigma_prior = sigma_prior
        self.sigma_eps = sigma_eps
        self.n_hidden_layers = n_hidden_layers
        self.local_hidden_units = local_hidden_units
        self.global_hidden_units = global_hidden_units
        self.map_mode = map_mode

        self.qnet = self.create_ddq_network(prefix='qnet_')
        self.target = self.create_ddq_network(prefix='target_')

        self.trainer = gl.Trainer(params=self.qnet.collect_params(),
                                  optimizer='adam',
                                  optimizer_params=dict(learning_rate=self.learning_rate))

    def create_ddq_network(self, prefix=''):
        network = MultiAgentBBBNetwork(
            domain_string=self.domain_string,
            input_dim=self.s_dim,
            output_dim=self.a_dim,
            n_hidden_layers=self.n_hidden_layers,
            local_hidden_units=self.local_hidden_units,
            global_hidden_units=self.global_hidden_units,
            n_samples=self.n_samples,
            sigma_prior=self.sigma_prior,
            sigma_eps=self.sigma_eps,
            map_mode=self.map_mode,
            prefix=prefix)
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
        return self.qnet(nd.array(inputs, ctx=CTX), True)[0].asnumpy()

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