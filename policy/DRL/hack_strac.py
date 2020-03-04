#!/usr/bin/env python
# @Time    : 2020-02-25 14:42
# @Author  : Zhi Chen
# @Desc    : hack_strac

import mxnet as mx
import mxnet.gluon as gl
import mxnet.ndarray as nd
import copy
import random
from NoisyDense import NoisyDense
import threading
import numpy as np

CTX = mx.cpu()


class MATransfer(gl.nn.Block):
    def __init__(self, slots, local_in_units, local_units, local_dropout,
                 global_in_units, global_units, global_dropout, activation,
                 concrete_share_rate, dropout_regularizer, use_comm=True,
                 non_local_mode=False, block_mode=False, slots_comm=True,
                 topo_learning_mode=False, message_embedding=False):
        gl.nn.Block.__init__(self)
        self.slots = slots
        self.use_comm = use_comm
        self.local_in_units = local_in_units
        self.local_units = local_units
        self.global_in_units = global_in_units
        self.global_units = global_units
        self.concrete_share_rate = concrete_share_rate
        self.dropout_regularizer = dropout_regularizer
        self.non_local_mode = non_local_mode
        self.block_mode = block_mode
        self.slots_comm = slots_comm
        self.topo_learning_mode = topo_learning_mode
        self.message_embedding = message_embedding

        assert slots_comm is True
        assert concrete_share_rate is False

        with self.name_scope():
            self.local_share_trans = NoisyDense(in_units=local_in_units, units=local_units,
                                                activation=activation)
            self.global_trans = NoisyDense(in_units=global_in_units, units=global_units, activation=activation)
            self.local2local_share_comm = NoisyDense(
                in_units=local_in_units, units=local_units, activation=activation)
            self.local2global_comm = NoisyDense(in_units=local_in_units, units=global_units,
                                                activation=activation)
            self.global2local_comm = NoisyDense(in_units=global_in_units, units=local_units,
                                                activation=activation)

            self.local_dropout_op = gl.nn.Dropout(local_dropout)
            self.global_dropout_op = gl.nn.Dropout(global_dropout)

    def reset_noise(self):
        for name, child in self._children.items():
            if name.find('drop') < 0:
                child.reset_noise()
            else:
                continue

    def forward(self, inputs, loss=None, training=True, commtype='average', topo='FC'):
        assert len(inputs) == self.slots + 1

        local_drop_vec = nd.ones_like(inputs[0])
        local_drop_vec = self.local_dropout_op(local_drop_vec)
        for i in range(self.slots):
            inputs[i] = inputs[i] * local_drop_vec
        inputs[-1] = self.global_dropout_op(inputs[-1])

        if topo == 'FC':
            comm_rate = nd.ones(shape=(self.slots + 1, self.slots + 1))
        elif topo == 'FUC':
            comm_rate = nd.zeros(shape=(self.slots + 1, self.slots + 1))
        elif topo == 'Master':
            comm_rate = nd.ones(shape=(self.slots + 1, self.slots + 1))
            for i in range(self.slots):
                for j in range(self.slots):
                    comm_rate[i][j] = 0

        if self.use_comm and self.topo_learning_mode:
            proba = nd.sigmoid(self.topo.data())

            if random.random() < 1e-2:
                print '---------------------------------------------'
                print proba.asnumpy()
                print '---------------------------------------------'

            u_vec = nd.random_uniform(low=1e-5, high=1. - 1e-5, shape=(self.slots + 1, self.slots + 1))
            comm_rate = nd.sigmoid(10. * (
                    nd.log(proba) - nd.log(1. - proba) +
                    nd.log(u_vec) - nd.log(1. - u_vec)
            ))
            if loss is not None:
                loss.append(4e-4 * nd.sum(proba * nd.log(proba) + (1. - proba) * nd.log(1. - proba)))

        results = []
        for i in range(self.slots):
            results.append(self.local_share_trans.forward(inputs[i], training=training))
        results.append(self.global_trans.forward(inputs[-1], training=training))

        if commtype == 'average':
            for i in range(self.slots):
                tmp = nd.zeros_like(results[i])
                norm = nd.zeros_like(comm_rate[0][0])
                for j in range(self.slots):
                    if i != j:
                        tmp = tmp + self.local2local_share_comm.forward(nd.concat(inputs[j], dim=1),
                                                                        training=training) * comm_rate[j][i]
                        norm = norm + comm_rate[j][i]
                # results[i] = results[i] + self.global2local_comm(inputs[-1]) * comm_rate[-1][i]
                tmp = tmp + self.global2local_comm.forward(nd.concat(inputs[-1], dim=1), training=training) * \
                      comm_rate[-1][i]
                norm = norm + comm_rate[-1][i]
                if nd.sum(norm) > 1e-5:
                    results[i] = results[i] + tmp / norm

            tmp = nd.zeros_like(results[-1])
            norm = nd.zeros_like(comm_rate[0][0])
            for j in range(self.slots):
                tmp = tmp + self.local2global_comm.forward(nd.concat(inputs[j], dim=1), training=training) * \
                      comm_rate[j][-1]
                norm = norm + comm_rate[j][-1]
            if nd.sum(norm) > 1e-5:
                results[-1] = results[-1] + tmp / norm

        elif commtype == 'maxpooling':
            for i in range(self.slots):
                tmp = []
                for j in range(self.slots):
                    if j != i:
                        tmp.append(self.local2local_share_comm.forward(inputs[j], training=training))
                tmp.append(self.global2local_comm.forward(inputs[-1], training=training))

                for k in range(len(tmp)):
                    tmp[k] = tmp[k].reshape((tmp[k].shape[0], 1, tmp[k].shape[1]))

                tmp = nd.concat(*tmp, dim=1)
                maxcomm = nd.max(tmp, axis=1)
                results[i] = results[i] + maxcomm

            tmp = []
            for i in range(self.slots):
                tmp.append(self.local2global_comm.forward(inputs[i], training=training))
            for k in range(len(tmp)):
                tmp[k] = tmp[k].reshape((tmp[k].shape[0], 1, tmp[k].shape[1]))

            tmp = nd.concat(*tmp, dim=1)
            maxcomm = nd.max(tmp, axis=1)
            results[-1] = results[-1] + maxcomm

        return results


class MultiAgentNetwork(gl.nn.Block):
    def __init__(self, domain_string, hidden_layers, local_hidden_units, local_dropouts,
                 global_hidden_units, global_dropouts, private_rate, sort_input_vec,
                 share_last_layer, recurrent_mode, input_comm, concrete_share_rate, dropout_regularizer,
                 non_local_mode, block_mode, slots_comm, topo_learning_mode,
                 use_dueling, dueling_share_last, message_embedding, state_feature, shared_last_layer_use_bias,
                 **kwargs):
        gl.nn.Block.__init__(self, **kwargs)

        assert (non_local_mode and message_embedding) is False

        self.domain_string = domain_string
        self.state_feature = state_feature
        if self.domain_string == 'Laptops11':
            self.slots = ['batteryrating', 'driverange', 'family',
                          'isforbusinesscomputing', 'platform', 'pricerange',
                          'processorclass', 'sysmemory', 'utility', 'warranty',
                          'weightrange']  # output order
            if self.state_feature == 'vanilla':
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
            elif self.state_feature == 'dip':
                self.slot_dimension = {
                    'warranty': (0, 25),
                    'family': (25, 50),
                    'utility': (50, 75),
                    'platform': (75, 100),
                    'processorclass': (100, 125),
                    'pricerange': (125, 150),
                    'batteryrating': (150, 175),
                    'sysmemory': (175, 200),
                    'weightrange': (200, 225),
                    'isforbusinesscomputing': (225, 250),
                    'driverange': (250, 275)
                }
                self.global_dimension = [(275, 349), ]
                self.input_dimension = 349
                self.global_input_dimension = 349 - 275

        elif self.domain_string == 'SFRestaurants':
            self.slots = ['allowedforkids', 'area', 'food', 'goodformeal', 'near', 'pricerange']
            if self.state_feature == 'vanilla':
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
            elif self.state_feature == 'dip':
                self.slot_dimension = {
                    'goodformeal': (0, 25),
                    'area': (25, 50),
                    'food': (50, 75),
                    'allowedforkids': (75, 100),
                    'near': (100, 125),
                    'pricerange': (125, 150)
                }
                self.global_dimension = [(150, 224), ]
                self.input_dimension = 224
                self.global_input_dimension = 224 - 150

        elif self.domain_string == 'CamRestaurants':
            self.slots = ['area', 'food', 'pricerange']
            if self.state_feature == 'vanilla':
                self.slot_dimension = {
                    'food': (0, 93),
                    'pricerange': (93, 98),
                    'area': (213, 220)
                }
                self.global_dimension = [(98, 213), (220, 268)]
                self.input_dimension = 268
                self.global_input_dimension = (213 - 98) + (268 - 220)
            elif self.state_feature == 'dip':
                self.slot_dimension = {
                    'food': (0, 25),
                    'pricerange': (25, 50),
                    'area': (50, 75)
                }
                self.global_dimension = [(75, 149), ]
                self.input_dimension = 149
                self.global_input_dimension = 149 - 75
        else:
            raise ValueError

        self.hidden_layers = hidden_layers
        self.local_hidden_units = local_hidden_units
        self.local_dropouts = local_dropouts
        self.global_hidden_units = global_hidden_units
        self.global_dropouts = global_dropouts
        self.private_rate = private_rate
        self.sort_input_vec = sort_input_vec
        self.share_last_layer = share_last_layer
        self.recurrent_mode = recurrent_mode
        self.conrete_share_rate = concrete_share_rate
        self.dropout_regularizer = dropout_regularizer
        self.non_local_mode = non_local_mode
        self.block_mode = block_mode
        self.slots_comm = slots_comm
        self.topo_learning_mode = topo_learning_mode
        self.use_dueling = use_dueling
        self.dueling_share_last = dueling_share_last
        self.message_embedding = message_embedding
        self.shared_last_layer_use_bias = shared_last_layer_use_bias

        with self.name_scope():
            self.input_trans = MATransfer(
                slots=len(self.slots),
                local_in_units=25,
                local_units=self.local_hidden_units[0],
                local_dropout=0.,
                global_in_units=self.global_input_dimension,
                global_units=self.global_hidden_units[0],
                global_dropout=0.,
                activation='relu',
                use_comm=input_comm,
                concrete_share_rate=self.conrete_share_rate,
                dropout_regularizer=self.dropout_regularizer,
                topo_learning_mode=self.topo_learning_mode,
                message_embedding=self.message_embedding)

            self.ma_trans = []
            for i in range(self.hidden_layers - 1):
                self.ma_trans.append(MATransfer(
                    slots=len(self.slots),
                    local_in_units=self.local_hidden_units[i],
                    local_units=self.local_hidden_units[i + 1],
                    local_dropout=self.local_dropouts[i],
                    global_in_units=self.global_hidden_units[i],
                    global_units=self.global_hidden_units[i + 1],
                    global_dropout=self.global_dropouts[i],
                    activation='relu',
                    concrete_share_rate=self.conrete_share_rate,
                    dropout_regularizer=self.dropout_regularizer,
                    non_local_mode=self.non_local_mode,
                    block_mode=self.block_mode,
                    slots_comm=self.slots_comm,
                    topo_learning_mode=self.topo_learning_mode,
                    message_embedding=self.message_embedding
                ))
                self.register_child(self.ma_trans[-1])

            self.local_out_drop_op = gl.nn.Dropout(self.local_dropouts[-1])
            self.global_out_drop_op = gl.nn.Dropout(self.global_dropouts[-1])

            self.output_trans_local_slotQ = NoisyDense(in_units=self.local_hidden_units[-1], units=1,
                                                       use_bias=False)
            self.output_trans_local_slotP = NoisyDense(in_units=self.local_hidden_units[-1], units=1)
            self.output_trans_global_slotQ = NoisyDense(in_units=self.global_hidden_units[-1], units=1)
            self.output_trans_global_slotP = NoisyDense(in_units=self.global_hidden_units[-1], units=1)
            self.output_trans_local_valueP = NoisyDense(in_units=self.local_hidden_units[-1], units=3)
            self.output_trans_global_valueP = NoisyDense(in_units=self.global_hidden_units[-1], units=7)
            if self.shared_last_layer_use_bias:
                self.value_bias_local = self.params.get(name='value_bias_local', shape=(len(self.slots),))

    def forward(self, input_vec, loss=None, training=True):
        assert input_vec.shape[1] == self.input_dimension

        # get inputs for every slot(including global)
        inputs = {}
        for slot in self.slots:
            inputs[slot] = input_vec[:, self.slot_dimension[slot][0]:self.slot_dimension[slot][1]]
        input_global = []
        for seg in self.global_dimension:
            input_global.append(input_vec[:, seg[0]:seg[1]])
        inputs['global'] = nd.concat(*input_global, dim=1)

        layer = []
        # inputs -> first_hidden_layer
        sorted_inputs = []
        for slot in self.slots:
            sorted_inputs.append(inputs[slot])
        sorted_inputs.append(inputs['global'])
        layer.append(self.input_trans.forward(sorted_inputs, loss, training=training))

        # hidden_layers
        for i in range(self.hidden_layers - 1):
            layer.append(self.ma_trans[i](layer[i], loss))

        if self.share_last_layer is False:
            # dropout of last hidden layer
            for j in range(len(self.slots)):
                layer[-1][j] = self.local_out_drop_op.forward(layer[-1][j])
            layer[-1][-1] = self.global_out_drop_op.forward(layer[-1][-1])

        # last_hidden_layer -> outputs
        slotv_probs = []
        slotqs = []
        slot_probs = []
        top_decision = []
        for i in range(len(self.slots) + 1):
            if i < len(self.slots):
                cur_slotv_prob = self.output_trans_local_valueP.forward(layer[-1][i], training=training)
            else:
                cur_slotv_prob = self.output_trans_global_valueP.forward(layer[-1][i], training=training)

            cur_slotv_prob_adv = cur_slotv_prob - nd.max(cur_slotv_prob, axis=1, keepdims=True)

            if i < len(self.slots):
                cur_slotq = self.output_trans_local_slotQ.forward(layer[-1][i], training=training)
                cur_slot_prob = self.output_trans_local_slotP.forward(layer[-1][i],
                                                                      training=training).reshape(-1, 1)
                if self.shared_last_layer_use_bias:
                    cur_slotq = cur_slotq + nd.slice(self.value_bias_local.data(), begin=(i,), end=(i + 1,))
            else:
                cur_slotq = self.output_trans_global_slotQ.forward(layer[-1][i], training=training)
                cur_slot_prob = self.output_trans_global_slotP.forward(layer[-1][i],
                                                                       training=training).reshape(-1, 1)
            cur_slotv_prob = cur_slot_prob + cur_slotv_prob_adv
            top_decision.append(cur_slot_prob)

            slotv_probs.append(cur_slotv_prob)
            slot_probs.append(cur_slot_prob)
            slotqs.append(cur_slotq)

        batch_slot_slotq = nd.concat(*slotqs, dim=1)
        batch_slotv_prob = nd.softmax(nd.concat(*slotv_probs, dim=1))
        batch_top_decision = nd.softmax(nd.concat(*top_decision, dim=1))

        prob = batch_slotv_prob
        value = nd.sum(batch_top_decision * batch_slot_slotq, axis=1)
        top_decision = batch_top_decision

        return prob, value, top_decision


class STRACNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    """

    def __init__(self, state_dim, action_dim, learning_rate, tau,
                 num_actor_vars, minibatch_size=64, architecture='duel',
                 h1_size=130, h1_drop=None, h2_size=50, h2_drop=None, domain_string=None,
                 hidden_layers=None, local_hidden_units=None, local_dropouts=None,
                 global_hidden_units=None, global_dropouts=None, private_rate=None,
                 sort_input_vec=None, share_last_layer=None, recurrent_mode=None,
                 input_comm=None, target_explore=None, concrete_share_rate=None,
                 dropout_regularizer=None, weight_regularizer=None,
                 non_local_mode=None, block_mode=None, slots_comm=None,
                 topo_learning_mode=None, use_dueling=None, dueling_share_last=None,
                 message_embedding=None, state_feature=None, init_policy=None, shared_last_layer_use_bias=None,
                 seed=None):
        self.domain_string = domain_string
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.architecture = architecture
        self.hidden_layers = hidden_layers
        self.local_hidden_units = local_hidden_units
        self.local_dropouts = local_dropouts
        self.global_hidden_units = global_hidden_units
        self.global_dropouts = global_dropouts
        self.minibatch_size = minibatch_size
        self.private_rate = private_rate
        self.sort_input_vec = sort_input_vec
        self.share_last_layer = share_last_layer
        self.recurrent_mode = recurrent_mode
        self.input_comm = input_comm
        self.target_explore = target_explore
        self.concrete_share_rate = concrete_share_rate
        self.dropout_regularizer = dropout_regularizer
        self.non_local_mode = non_local_mode
        self.block_mode = block_mode
        self.slots_comm = slots_comm
        self.topo_learning_mode = topo_learning_mode
        self.use_dueling = use_dueling
        self.message_embedding = message_embedding
        self.dueling_share_last = dueling_share_last
        self.state_feature = state_feature
        self.init_policy = init_policy
        self.shared_last_layer_use_bias = shared_last_layer_use_bias
        self.seed = seed

        self.actorcritic = self.create_ddq_network(prefix='actorcritic_')

        self.trainer = gl.Trainer(params=self.actorcritic.collect_params(), optimizer='adam',
                                  optimizer_params=dict(learning_rate=self.learning_rate * 0.1, clip_gradient=10))

    def create_ddq_network(self, prefix=''):
        network = MultiAgentNetwork(domain_string=self.domain_string,
                                    hidden_layers=self.hidden_layers,
                                    local_hidden_units=self.local_hidden_units,
                                    local_dropouts=self.local_dropouts,
                                    global_hidden_units=self.global_hidden_units,
                                    global_dropouts=self.local_dropouts,
                                    private_rate=self.private_rate,
                                    sort_input_vec=self.sort_input_vec,
                                    share_last_layer=self.share_last_layer,
                                    recurrent_mode=self.recurrent_mode,
                                    input_comm=self.input_comm,
                                    concrete_share_rate=self.concrete_share_rate,
                                    dropout_regularizer=self.dropout_regularizer,
                                    non_local_mode=self.non_local_mode,
                                    block_mode=self.block_mode,
                                    slots_comm=self.slots_comm,
                                    topo_learning_mode=self.topo_learning_mode,
                                    use_dueling=self.use_dueling,
                                    dueling_share_last=self.dueling_share_last,
                                    message_embedding=self.message_embedding,
                                    state_feature=self.state_feature,
                                    shared_last_layer_use_bias=self.shared_last_layer_use_bias,
                                    prefix=prefix)
        # print network.collect_params()
        # network.initialize(ctx=CTX)
        return network

    def train(self, s_batch, a_batch_one_hot, V_trace, advantage):
        batch_size = s_batch.shape[0]
        action_indx = np.argmax(a_batch_one_hot, axis=1).tolist()
        action_stats = [action_indx.count(action_indx[i]) for i in range(batch_size)]
        action_bp_rate = (1 - np.array(action_stats) / float(batch_size)) ** 2

        s_batch = copy.deepcopy(s_batch)
        a_batch_one_hot = copy.deepcopy(a_batch_one_hot)
        V_trace_batch = copy.deepcopy(V_trace)
        advantage_batch = copy.deepcopy(advantage)

        s_batch = nd.array(s_batch, ctx=CTX)
        a_batch_one_hot = nd.array(a_batch_one_hot, ctx=CTX)
        V_trace_batch = nd.array(V_trace_batch, ctx=CTX)
        advantage_batch = nd.array(advantage_batch, ctx=CTX)
        action_bp_rate = nd.softmax(nd.array(action_bp_rate, ctx=CTX))

        self.actorcritic.collect_params().zero_grad()
        self.reset_noise()
        with mx.autograd.record():
            loss_vec = []
            probs, values, top_decisions = self.actorcritic.forward(s_batch, loss_vec)
            loss = 0.
            for element in loss_vec:
                loss = loss + element
            # print 'loss_dropout:', loss
            logprob = nd.log(nd.sum(data=probs * a_batch_one_hot, axis=1) + 1e-5)
            entropy = -nd.sum(nd.sum(data=probs * nd.log(probs + 1e-5), axis=1), axis=0)
            top_decision_entropy = -nd.sum(nd.sum(data=top_decisions * nd.log(top_decisions + 1e-5), axis=1), axis=0)
            entropy_loss = - entropy
            top_decision_entropy_loss = - top_decision_entropy
            actorloss = -nd.sum(action_bp_rate * (logprob * advantage_batch), axis=0)
            criticloss = nd.sum(action_bp_rate * nd.square(values - V_trace_batch), axis=0)
            # actorloss = -nd.sum(logprob*advantage_batch, axis=0)
            # criticloss = nd.sum(nd.square(values-V_trace_batch), axis=0)
            loss = actorloss + 0.3 * criticloss + 0.001 * entropy_loss

            # loss = actorloss + 0.3*criticloss + 0.0001*top_decision_entropy_loss
        loss.backward()

        grads_list = []
        for name, value in self.actorcritic.collect_params().items():
            if name.find('batchnorm') < 0:
                # grads_list.append(mx.nd.array(value.grad().asnumpy()))
                grads_list.append(value.grad())

        return grads_list, batch_size

    def train_update(self, s_batch, a_batch_one_hot, V_trace, advantage):
        batch_size = s_batch.shape[0]
        action_indx = np.argmax(a_batch_one_hot, axis=1).tolist()
        action_stats = [action_indx.count(action_indx[i]) for i in range(batch_size)]
        action_bp_rate = (1 - np.array(action_stats) / float(batch_size)) ** 2

        s_batch = copy.deepcopy(s_batch)
        a_batch_one_hot = copy.deepcopy(a_batch_one_hot)
        V_trace_batch = copy.deepcopy(V_trace)
        advantage_batch = copy.deepcopy(advantage)

        s_batch = nd.array(s_batch, ctx=CTX)
        a_batch_one_hot = nd.array(a_batch_one_hot, ctx=CTX)
        V_trace_batch = nd.array(V_trace_batch, ctx=CTX)
        advantage_batch = nd.array(advantage_batch, ctx=CTX)
        action_bp_rate = nd.softmax(nd.array(action_bp_rate, ctx=CTX))

        self.actorcritic.collect_params().zero_grad()
        self.reset_noise()
        with mx.autograd.record():
            loss_vec = []
            probs, values, top_decisions = self.actorcritic.forward(s_batch, loss_vec)
            loss = 0.
            for element in loss_vec:
                loss = loss + element
            # print 'loss_dropout:', loss
            logprob = nd.log(nd.sum(data=probs * a_batch_one_hot, axis=1) + 1e-5)
            entropy = -nd.sum(nd.sum(data=probs * nd.log(probs + 1e-5), axis=1), axis=0)
            top_decision_entropy = -nd.sum(nd.sum(data=top_decisions * nd.log(top_decisions + 1e-5), axis=1), axis=0)
            entropy_loss = - entropy
            top_decision_entropy_loss = - top_decision_entropy
            actorloss = -nd.sum(action_bp_rate * (logprob * advantage_batch), axis=0)
            criticloss = nd.sum(action_bp_rate * nd.square(values - V_trace_batch), axis=0)
            # actorloss = -nd.sum(logprob*advantage_batch, axis=0)
            # criticloss = nd.sum(nd.square(values-V_trace_batch), axis=0)
            loss = actorloss + 0.3 * criticloss + 0.001 * entropy_loss

            # loss = actorloss + 0.3*criticloss + 0.0001*top_decision_entropy_loss
        loss.backward()

        self.trainer.step(batch_size=batch_size, ignore_stale_grad=True)

    def reset_noise(self):
        for name, child in self.actorcritic._children.items():
            if name.find('drop') < 0:
                # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                # print(name)
                child.reset_noise()
            else:
                # print('######################################')
                # print(name)
                continue

    def predict_action_value(self, inputs):
        self.reset_noise()
        assert mx.autograd.is_training() is False
        prob, value, top_decisions = self.actorcritic.forward(nd.array(inputs, ctx=CTX))
        prob = prob.asnumpy()
        value = value.asnumpy()
        return prob, value

    def predict_value(self, inputs):
        self.reset_noise()
        assert mx.autograd.is_training() is False
        _, value, _ = self.actorcritic.forward(nd.array(inputs, ctx=CTX))
        value = value.asnumpy()
        return value

    def getPolicy(self, inputs):
        self.reset_noise()
        assert mx.autograd.is_training() is False
        prob, _, _ = self.actorcritic.forward(nd.array(inputs, ctx=CTX))
        prob = prob.asnumpy()
        return prob

    def load_test_network(self, load_filename):
        self.actorcritic.load_parameters(filename=load_filename + '_actorcritic', ctx=CTX)

    def load_trainer(self, load_filename):
        self.trainer.step(1, ignore_stale_grad=True)
        self.trainer.load_states(fname=load_filename + '_trainer')
        print('load network and trainer successfully!!!')

    def load_network(self, load_filename):
        try:
            self.actorcritic.load_parameters(filename=load_filename + '_actorcritic', ctx=CTX)
            self.trainer.step(1, ignore_stale_grad=True)
            self.trainer.load_states(fname=load_filename + '_trainer')
            print "Successfully loaded:", load_filename
        except:
            try:
                init_policy_name = self.init_policy.replace('*', str(self.seed))
                print "Could not find old network weights({}), try self.init_policy({})".format(
                    load_filename, init_policy_name)
                self.actorcritic.load_parameters(filename=init_policy_name + '_actorcritic', ctx=CTX)
                print "Successfully loaded:", self.init_policy
            except:
                print 'no init policy or cannot load it.'

    def save_network(self, save_filename):
        self.actorcritic.save_parameters(filename=save_filename + '_actorcritic')

    def copy_parameters(self, target_policy):
        source_policy = self.actorcritic.collect_params().items()
        i = 0
        for name, value in target_policy.actorcritic.collect_params().items():
            source_policy[i][1].set_data(value.data())
            i += 1

    def save_trainer(self, save_filename):
        # print 'Saving deepq-network periodically...'
        # self.actorcritic.save_params(filename=save_filename + '_actorcritic')
        # self.target.save_params(filename=save_filename + '_target')
        self.trainer.step(1, ignore_stale_grad=True)
        self.trainer.save_states(fname=save_filename + '_trainer')
        print('save trainer as ' + save_filename + '_trainer')
