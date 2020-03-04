import mxnet as mx
import mxnet.gluon as gl
import mxnet.ndarray as nd
import copy
import random
import numpy as np

CTX = mx.cpu()


class MATransfer(gl.nn.Block):
    def __init__(self, slots, local_in_units, local_units, local_dropout,
                 global_in_units, global_units, global_dropout, activation,
                 concrete_share_rate, dropout_regularizer, use_comm=True,
                 non_local_mode=False, block_mode=False, slots_comm=True,
                 topo_learning_mode=False, message_embedding=False, num_heads=8):
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
        self.num_heads = num_heads

        assert slots_comm is True
        assert concrete_share_rate is False

        with self.name_scope():
            if self.topo_learning_mode:
                self.topo = self.params.get(
                    name='topo_mat', shape=(self.slots + 1, self.slots + 1), init=mx.init.Constant(2.))
            if self.non_local_mode is False:
                if self.message_embedding:
                    self.local_trans = gl.nn.Dense(in_units=local_in_units, units=local_units, activation=activation)
                    self.global_trans = gl.nn.Dense(in_units=global_in_units, units=global_units, activation=activation)
                    if self.use_comm:
                        msg_dim = 16
                        emb_dim = 64
                        self.local2global_msg_encode = gl.nn.Dense(in_units=local_in_units, units=msg_dim)
                        self.local2local_msg_encode = gl.nn.Dense(in_units=local_in_units, units=msg_dim)
                        self.global2local_msg_encode = gl.nn.Dense(in_units=global_in_units, units=msg_dim)

                        self.local2global_embedding = gl.nn.Dense(in_units=msg_dim, units=emb_dim)
                        self.local2local_embedding = gl.nn.Dense(in_units=msg_dim, units=emb_dim)
                        self.global2local_embedding = gl.nn.Dense(in_units=msg_dim, units=emb_dim)

                        self.local2global_extract = gl.nn.Dense(in_units=emb_dim, units=global_units)
                        self.local2local_extract = gl.nn.Dense(in_units=emb_dim, units=local_units)
                        self.global2local_extract = gl.nn.Dense(in_units=emb_dim, units=local_units)
                else:
                    self.local_share_trans = gl.nn.Dense(in_units=local_in_units, units=local_units, activation=activation)
                    self.global_trans = gl.nn.Dense(in_units=global_in_units, units=global_units, activation=activation)
                    if self.use_comm:
                        if self.slots_comm:
                            self.local2local_share_comm = gl.nn.Dense(
                                in_units=local_in_units, units=local_units, activation=activation)
                        self.local2global_comm = gl.nn.Dense(in_units=local_in_units, units=global_units, activation=activation)
                        self.global2local_comm = gl.nn.Dense(in_units=global_in_units, units=local_units, activation=activation)
                    # self.local_private_trans = []
                    # self.local2local_private_comm = []
                    # self.share_rate = []
                    # for i in range(self.slots):
                        # self.local_private_trans.append(gl.nn.Dense(
                        #     in_units=local_in_units, units=local_units, activation=activation))
                        # self.register_child(self.local_private_trans[-1])
                        # if self.use_comm and self.slots_comm:
                        #     self.local2local_private_comm.append(gl.nn.Dense(
                        #         in_units=local_in_units, units=local_units, activation=activation))
                        #     self.register_child(self.local2local_private_comm[-1])
                        # if concrete_share_rate:
                        #     self.share_rate.append(self.params.get(
                        #         name='sharerate_{slot_num}'.format(slot_num=i), shape=(1, 1), init=mx.init.Constant([[2.]])))

                    self.local_dropout_op = gl.nn.Dropout(local_dropout)
                    self.global_dropout_op = gl.nn.Dropout(global_dropout)
            else:
                self.local_trans = {}
                self.global_trans = {}
                self.g_local2local = {}
                self.g_global2local = {}
                self.g_local2global = {}
                self.f_emit_local2local = {}
                self.f_emit_local2global = {}
                self.f_emit_global2local = {}
                self.f_rec_local = {}
                self.f_rec_global = {}
                mid_units = 64
                for k in range(self.num_heads):
                    self.local_trans[k] = gl.nn.Dense(in_units=local_in_units, units=local_units, activation=activation)
                    self.global_trans[k] = gl.nn.Dense(in_units=global_in_units, units=global_units, activation=activation)
                    self.g_local2local[k] = gl.nn.Dense(in_units=local_in_units, units=local_units, use_bias=False)
                    self.g_global2local[k] = gl.nn.Dense(in_units=global_in_units, units=local_units, use_bias=False)
                    self.g_local2global[k] = gl.nn.Dense(in_units=local_in_units, units=global_units, use_bias=False)
                    if self.slots_comm:
                        self.f_emit_local2local[k] = gl.nn.Dense(in_units=local_in_units, units=mid_units, use_bias=False)
                    self.f_emit_local2global[k] = gl.nn.Dense(in_units=local_in_units, units=mid_units, use_bias=False)
                    self.f_emit_global2local[k] = gl.nn.Dense(in_units=global_in_units, units=mid_units, use_bias=False)
                    self.f_rec_local[k] = gl.nn.Dense(in_units=local_in_units, units=mid_units, use_bias=False)
                    self.f_rec_global[k] = gl.nn.Dense(in_units=global_in_units, units=mid_units, use_bias=False)
                    self.register_child(self.local_trans[k])
                    self.register_child(self.global_trans[k])
                    self.register_child(self.g_local2local[k])
                    self.register_child(self.g_global2local[k])
                    self.register_child(self.g_local2global[k])
                    self.register_child(self.f_emit_local2local[k])
                    self.register_child(self.f_emit_local2global[k])
                    self.register_child(self.f_emit_global2local[k])
                    self.register_child(self.f_rec_local[k])
                    self.register_child(self.f_rec_global[k])

                self.headLinear_local = gl.nn.Dense(in_units=local_units*self.num_heads, units=local_units, use_bias=False)
                self.headLinear_global = gl.nn.Dense(in_units=global_units*self.num_heads, units=global_units, use_bias=False)

            if self.block_mode:
                self.yz_weight_local = gl.nn.Dense(in_units=local_in_units, units=local_units, use_bias=False)
                self.yz_weight_global = gl.nn.Dense(in_units=global_in_units, units=global_units, use_bias=False)

    def forward_multiheads(self, inputs, loss, headtype='maxpooling'):
        
        comm_rate = nd.ones(shape=(self.slots + 1, self.slots + 1))
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

        f = [[[None] * (self.slots + 1)] * (self.slots + 1)] * (self.num_heads)
        if self.use_comm:
            # local
            multihead_results = []
            for k in range(self.num_heads):
                results = []
                for i in range(self.slots):
                    results.append(self.local_trans[k](inputs[i]))
                results.append(self.global_trans[k](inputs[-1]))

                for i in range(self.slots):
                    norm_fac = None
                    for j in range(self.slots):
                        if i != j:
                            f[k][i][j] = nd.sum(self.f_rec_local[k](inputs[i]) * self.f_emit_local2local[k](inputs[j]), axis=1)
                            f[k][i][j] = nd.exp(f[k][i][j]).reshape((f[k][i][j].shape[0], 1))
                            f[k][i][j] = f[k][i][j] * comm_rate[j][i]
                            if norm_fac is None:
                                norm_fac = nd.zeros_like(f[k][i][j])
                            norm_fac = norm_fac + f[k][i][j]
                    f[k][i][-1] = nd.sum(self.f_rec_local[k](inputs[i]) * self.f_emit_global2local[k](inputs[-1]), axis=1)
                    f[k][i][-1] = nd.exp(f[k][i][-1]).reshape((f[k][i][-1].shape[0], 1))
                    f[k][i][-1] = f[k][i][-1] * comm_rate[-1][i]
                    if norm_fac is None:
                        norm_fac = nd.zeros_like(f[k][i][-1])
                    norm_fac = norm_fac + f[k][i][-1]

                    for j in range(self.slots):
                        if i != j:
                            results[i] = results[i] + (1. / norm_fac) * f[k][i][j] * self.g_local2local[k](inputs[j])
                    results[i] = results[i] + (1. / norm_fac) * f[k][i][-1] * self.g_global2local[k](inputs[-1])

                # global
                norm_fac = None
                for i in range(self.slots):
                    f[k][-1][i] = nd.sum(self.f_rec_global[k](inputs[-1]) * self.f_emit_local2global[k](inputs[i]), axis=1)
                    f[k][-1][i] = nd.exp(f[k][-1][i]).reshape((f[k][-1][i].shape[0], 1))
                    f[k][-1][i] = f[k][-1][i] * comm_rate[i][-1]
                    if norm_fac is None:
                        norm_fac = nd.zeros_like(f[k][-1][i])
                    norm_fac = norm_fac + f[k][-1][i]
                for i in range(self.slots):
                    results[-1] = results[-1] + (1. / norm_fac) * f[k][-1][i] * self.g_local2global[k](inputs[i])

                multihead_results.append(results)

            # local
            # cancate
            # results = []
            # for i in range(self.slots):
            #     tmp = []
            #     for k in range(self.num_heads):
            #         tmp.append(multihead_results[k][i])
            #     results.append(nd.concat(*tmp, dim=1))
            #     results[i] = self.headLinear_local(results[i])

            if headtype == 'sum':
                results = []
                for i in range(self.slots):
                    tmp = nd.zeros_like(multihead_results[0][i])
                    for k in range(self.num_heads):
                        tmp = tmp + multihead_results[k][i]
                    results.append(tmp)

                # global
                # tmp = []
                # for k in range(self.num_heads):
                #     tmp.append(multihead_results[k][-1])
                # results.append(nd.concat(*tmp, dim=1))
                # results[-1] = self.headLinear_global(results[-1])

                tmp = nd.zeros_like(multihead_results[0][-1])
                for k in range(self.num_heads):
                    tmp = tmp + multihead_results[k][-1]
                results.append(tmp) 

            elif headtype == 'maxpooling':
                # local + global
                results = []
                for i in range(self.slots+1):
                    tmp = []
                    for k in range(self.num_heads):
                        tmp.append(multihead_results[k][i].reshape((multihead_results[k][i].shape[0],1,multihead_results[k][i].shape[1])))
                    tmp = nd.concat(*tmp, dim=1)
                    tmp = nd.max(tmp, axis=1)
                    results.append(tmp)
                    # print tmp.shape

        if self.block_mode:
            assert self.local_in_units == self.local_units
            assert self.global_in_units == self.global_units

            for i in range(self.slots):
                results[i] = self.yz_weight_local(results[i]) + inputs[i]
            results[-1] = self.yz_weight_global(results[-1]) + inputs[-1]

        return results


    def forward_non_local(self, inputs, loss):
        results = []
        for i in range(self.slots):
            results.append(self.local_trans(inputs[i]))
        results.append(self.global_trans(inputs[-1]))

        comm_rate = nd.ones(shape=(self.slots + 1, self.slots + 1))
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

        f = [[None] * (self.slots + 1)] * (self.slots + 1)
        if self.use_comm:
            # local
            for i in range(self.slots):
                norm_fac = None
                for j in range(self.slots):
                    if i != j:
                        f[i][j] = nd.sum(self.f_rec_local(inputs[i]) * self.f_emit_local2local(inputs[j]), axis=1)
                        f[i][j] = nd.exp(f[i][j]).reshape((f[i][j].shape[0], 1))
                        f[i][j] = f[i][j] * comm_rate[j][i]
                        if norm_fac is None:
                            norm_fac = nd.zeros_like(f[i][j])
                        norm_fac = norm_fac + f[i][j]
                f[i][-1] = nd.sum(self.f_rec_local(inputs[i]) * self.f_emit_global2local(inputs[-1]), axis=1)
                f[i][-1] = nd.exp(f[i][-1]).reshape((f[i][-1].shape[0], 1))
                f[i][-1] = f[i][-1] * comm_rate[-1][i]
                if norm_fac is None:
                    norm_fac = nd.zeros_like(f[i][-1])
                norm_fac = norm_fac + f[i][-1]

                for j in range(self.slots):
                    if i != j:
                        results[i] = results[i] + (1. / norm_fac) * f[i][j] * self.g_local2local(inputs[j])
                results[i] = results[i] + (1. / norm_fac) * f[i][-1] * self.g_global2local(inputs[-1])

            # global
            norm_fac = None
            for i in range(self.slots):
                f[-1][i] = nd.sum(self.f_rec_global(inputs[-1]) * self.f_emit_local2global(inputs[i]), axis=1)
                f[-1][i] = nd.exp(f[-1][i]).reshape((f[-1][i].shape[0], 1))
                f[-1][i] = f[-1][i] * comm_rate[i][-1]
                if norm_fac is None:
                    norm_fac = nd.zeros_like(f[-1][i])
                norm_fac = norm_fac + f[-1][i]
            for i in range(self.slots):
                results[-1] = results[-1] + (1. / norm_fac) * f[-1][i] * self.g_local2global(inputs[i])

            # norm = [None] * (self.slots + 1)
            # for j in range(self.slots + 1):
            #     norm[j] = nd.zeros_like(f[j][0])
            #     for i in range(self.slots + 1):
            #         if i != j:
            #             norm[j] = norm[j] + f[j][i]

            # for i in range(self.slots + 1):
            #     for j in range(self.slots + 1):
            #         if i == j:
            #             print nd.zeros_like(f[j][i]).asnumpy(),
            #         else:
            #             print (f[j][i] / norm[j]).asnumpy(),
            #     print ''
            # print ''

        if self.block_mode:
            assert self.local_in_units == self.local_units
            assert self.global_in_units == self.global_units

            for i in range(self.slots):
                results[i] = self.yz_weight_local(results[i]) + inputs[i]
            results[-1] = self.yz_weight_global(results[-1]) + inputs[-1]

        return results

    def forward_message_embedding(self, inputs, loss):
        results = []
        for i in range(self.slots):
            results.append(self.local_trans(inputs[i]))
        results.append(self.global_trans(inputs[-1]))

        if self.use_comm:
            for i in range(self.slots):
                tmp = nd.zeros_like(results[i])
                for j in range(self.slots):
                    msg = nd.softmax(self.local2local_msg_encode(inputs[j]))
                    tmp = tmp + self.local2local_extract(self.local2local_embedding(msg))
                msg = nd.softmax(self.global2local_msg_encode(inputs[-1]))
                tmp = tmp + self.global2local_extract(self.global2local_embedding(msg))
                results[i] = results[i] + (tmp / float(self.slots))
            tmp = nd.zeros_like(results[-1])
            for i in range(self.slots):
                msg = nd.softmax(self.local2global_msg_encode(inputs[i]))
                tmp = tmp + self.local2global_extract(self.local2global_embedding(msg))
            results[-1] = results[-1] + (tmp / float(self.slots))
        return results

    def forward(self, inputs, loss=None):
        assert len(inputs) == self.slots + 1

        if self.non_local_mode:
            # return self.forward_non_local(inputs, loss)
            return self.forward_multiheads(inputs, loss)
        if self.message_embedding:
            return self.forward_message_embedding(inputs, loss)
        # if self.multi_heands:
        #     return self.forward_multiheads(inputs, loss)

        local_drop_vec = nd.ones_like(inputs[0])
        local_drop_vec = self.local_dropout_op(local_drop_vec)
        for i in range(self.slots):
            inputs[i] = inputs[i] * local_drop_vec
        inputs[-1] = self.global_dropout_op(inputs[-1])

        # local_share_vec = []
        # local_private_vec = []
        # if self.concrete_share_rate:
        #     raise ValueError('no share_private!!!')
        #     for i in range(self.slots):
        #         proba = nd.sigmoid(data=self.share_rate[i].data())
        #         proba = nd.broadcast_axis(data=proba, axis=(0, 1), size=inputs[0].shape)
        #         u_vec = nd.random_uniform(low=1e-5, high=1. - 1e-5, shape=inputs[0].shape, ctx=CTX)
        #         local_share_vec.append(nd.sigmoid(10. * (
        #             nd.log(proba) - nd.log(1. - proba) +
        #             nd.log(u_vec) - nd.log(1. - u_vec)
        #         )))
        #         local_private_vec.append(1. - local_share_vec[i])
        #         # print 'proba:', proba
        #         # print 'dropout_regularizer:', self.dropout_regularizer
        #         if loss is not None:
        #             loss.append(
        #                 self.dropout_regularizer * nd.sum(proba * nd.log(proba) + (1. - proba) * nd.log(1. - proba)))
        #     if random.random() < 0.01:
        #         for i in range(self.slots):
        #             proba = nd.sigmoid(data=self.share_rate[i].data())
        #             print proba.asnumpy(),
        #         print ''
        # else:
        #     local_share_vec = [nd.ones_like(inputs[0]), ] * self.slots
        #     local_private_vec = [nd.zeros_like(inputs[0]), ] * self.slots
        # local_share_vec = (1. - self.private_rate) * nd.Dropout(
        #     nd.ones(shape=(inputs[0].shape[0], self.local_units)), p=self.private_rate, mode='always')
        # local_private_vec = 1. - local_share_vec

        comm_rate = nd.ones(shape=(self.slots + 1, self.slots + 1))
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
            results.append(self.local_share_trans(inputs[i]))
        results.append(self.global_trans(inputs[-1]))

        if self.use_comm:
            if self.topo_learning_mode:
                assert self.concrete_share_rate is False
                for i in range(self.slots):
                    tmp = nd.zeros_like(results[i])
                    norm = nd.zeros_like(comm_rate[0][0])
                    for j in range(self.slots):
                        if i != j:
                            tmp = tmp + self.local2local_share_comm(inputs[j]) * comm_rate[j][i]
                            norm = norm + comm_rate[j][i]
                    # results[i] = results[i] + self.global2local_comm(inputs[-1]) * comm_rate[-1][i]
                    tmp = tmp + self.global2local_comm(inputs[-1]) * comm_rate[-1][i]
                    norm = norm + comm_rate[-1][i]
                    if nd.sum(norm) > 1e-5:
                        results[i] = results[i] + tmp / norm

                tmp = nd.zeros_like(results[-1])
                norm = nd.zeros_like(comm_rate[0][0])
                for j in range(self.slots):
                    tmp = tmp + self.local2global_comm(inputs[j]) * comm_rate[j][-1]
                    norm = norm + comm_rate[j][-1]
                if nd.sum(norm) > 1e-5:
                    results[-1] = results[-1] + tmp / norm
            else:
                for i in range(self.slots):
                    tmp = nd.zeros_like(results[i])
                    for j in range(self.slots):
                        if j != i:
                            tmp = tmp + self.local2local_share_comm(inputs[j])
                    tmp = tmp + self.global2local_comm(inputs[-1])
                    results[i] = results[i] + (tmp / float(self.slots))

                tmp = nd.zeros_like(results[-1])
                for i in range(self.slots):
                    tmp = tmp + self.local2global_comm(inputs[i])
                results[-1] = results[-1] + (tmp / float(self.slots))

        if self.block_mode:
            assert self.local_in_units == self.local_units
            assert self.global_in_units == self.global_units

            for i in range(self.slots):
                results[i] = self.yz_weight_local(results[i]) + inputs[i]
            results[-1] = self.yz_weight_global(results[-1]) + inputs[-1]

        return results


class MultiAgentNetwork(gl.nn.Block):
    def __init__(self, domain_string, hidden_layers, local_hidden_units, local_dropouts,
                 global_hidden_units, global_dropouts, private_rate, sort_input_vec,
                 share_last_layer, recurrent_mode, input_comm, concrete_share_rate, dropout_regularizer,
                 non_local_mode, block_mode, slots_comm, topo_learning_mode,
                 use_dueling, dueling_share_last, message_embedding, state_feature, shared_last_layer_use_bias, **kwargs):
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
            assert self.state_feature != 'dip' or not self.sort_input_vec

            if (self.sort_input_vec is False) and (self.state_feature != 'dip'):
                self.input_trans = {}
                for slot in self.slots:
                    in_units = self.slot_dimension[slot][1] - self.slot_dimension[slot][0]
                    self.input_trans[slot] = \
                        gl.nn.Dense(in_units=in_units, units=self.local_hidden_units[0], activation='relu')
                    self.register_child(self.input_trans[slot])
                self.input_trans['global'] = \
                    gl.nn.Dense(in_units=self.global_input_dimension,
                                units=self.global_hidden_units[0], activation='relu')
                self.register_child(self.input_trans['global'])
            elif self.sort_input_vec:
                self.input_trans = MATransfer(
                    slots=len(self.slots),
                    local_in_units=22,
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
            elif self.state_feature == 'dip':
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

            if self.recurrent_mode is False:
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
            else:
                assert self.local_hidden_units == (self.local_hidden_units[0], ) * self.hidden_layers
                assert self.local_dropouts == (self.local_dropouts[0], ) * self.hidden_layers
                assert self.global_hidden_units == (self.global_hidden_units[0], ) * self.hidden_layers
                assert self.global_dropouts == (self.global_dropouts[0], ) * self.hidden_layers

                self.ma_trans = MATransfer(
                    slots=len(self.slots),
                    local_in_units=self.local_hidden_units[0],
                    local_units=self.local_hidden_units[0],
                    local_dropout=self.local_dropouts[0],
                    global_in_units=self.global_hidden_units[0],
                    global_units=self.global_hidden_units[0],
                    global_dropout=self.global_dropouts[0],
                    activation='relu',
                    concrete_share_rate=self.conrete_share_rate,
                    dropout_regularizer=self.dropout_regularizer,
                    non_local_mode=self.non_local_mode,
                    block_mode=self.block_mode,
                    slots_comm=self.slots_comm,
                    topo_learning_mode=self.topo_learning_mode,
                    message_embedding=self.message_embedding
                )
            self.local_out_drop_op = gl.nn.Dropout(self.local_dropouts[-1])
            self.global_out_drop_op = gl.nn.Dropout(self.global_dropouts[-1])

            if self.use_dueling is False:
                if self.share_last_layer is False:
                    self.output_trans = []
                    for i in range(len(self.slots)):
                        self.output_trans.append(gl.nn.Dense(in_units=self.local_hidden_units[-1], units=3))
                        self.register_child(self.output_trans[-1])
                    self.output_trans.append(gl.nn.Dense(in_units=self.global_hidden_units[-1], units=7))
                    self.register_child(self.output_trans[-1])
                else:
                    self.output_trans_local = gl.nn.Dense(in_units=self.local_hidden_units[-1], units=3, use_bias=False)
                    if self.shared_last_layer_use_bias:
                        self.output_trans_local_biases = []
                        for i in range(len(self.slots)):
                            self.output_trans_local_biases.append(self.params.get(name='output_trans_local_bias_slot{}'.format(i), shape=(3,)))

                    self.output_trans_global = gl.nn.Dense(in_units=self.global_hidden_units[-1], units=7)
            else:
                assert self.sort_input_vec or self.state_feature == 'dip'
                if self.dueling_share_last:
                    self.output_trans_local_value = gl.nn.Dense(in_units=self.local_hidden_units[-1], units=1, use_bias=False)
                    self.output_trans_global_value = gl.nn.Dense(in_units=self.global_hidden_units[-1], units=1)
                    if self.shared_last_layer_use_bias:
                        self.value_bias_local = self.params.get(name='value_bias_local', shape=(len(self.slots), ))
                else:
                    self.output_trans_value = []
                    for i in range(len(self.slots)):
                        self.output_trans_value.append(gl.nn.Dense(in_units=self.local_hidden_units[-1], units=1))
                        self.register_child(self.output_trans_value[-1])
                    self.output_trans_value.append(gl.nn.Dense(in_units=self.global_hidden_units[-1], units=1))
                    self.register_child(self.output_trans_value[-1])

                self.output_trans_local_advantage = gl.nn.Sequential()
                slot_dim = 22 if self.sort_input_vec else 25
                self.output_trans_local_advantage.add(
                    gl.nn.Dense(in_units=slot_dim, units=64, activation='relu'),
                    gl.nn.Dense(in_units=64, units=3))

                self.output_trans_global_advantage = gl.nn.Sequential()
                self.output_trans_global_advantage.add(
                    gl.nn.Dense(in_units=self.global_input_dimension, units=100, activation='relu'),
                    gl.nn.Dense(in_units=100, units=7))

            # for key, value in self.collect_params().items():
            #     print key, value
            # exit(0)

    def forward(self, input_vec, loss=None):
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
        if (not self.sort_input_vec) and self.state_feature != 'dip':
            layer.append([])
            for slot in self.slots:
                layer[0].append(self.input_trans[slot](inputs[slot]))
            layer[0].append(self.input_trans['global'](inputs['global']))
        elif self.state_feature == 'dip':
            sorted_inputs = []
            for slot in self.slots:
                sorted_inputs.append(inputs[slot])
            sorted_inputs.append(inputs['global'])
            layer.append(self.input_trans(sorted_inputs, loss))
        elif self.sort_input_vec:
            sorted_inputs = []
            for slot in self.slots:
                tmp = inputs[slot][:, :-2].sort(is_ascend=False)
                if tmp.shape[1] < 20:
                    tmp = nd.concat(tmp, nd.zeros((tmp.shape[0], 20 - tmp.shape[1]), ctx=CTX), dim=1)
                else:
                    tmp = nd.slice_axis(tmp, axis=1, begin=0, end=20)
                sorted_inputs.append(nd.concat(tmp, inputs[slot][:, -2:], dim=1))
            sorted_inputs.append(inputs['global'])
            layer.append(self.input_trans(sorted_inputs, loss))

        # hidden_layers
        for i in range(self.hidden_layers - 1):
            if self.recurrent_mode is False:
                # equal to 'layer.append(self.ma_trans[i](layer[-1], loss))'
                layer.append(self.ma_trans[i](layer[i], loss))
            else:
                layer.append(self.ma_trans(layer[i], loss))

        if self.share_last_layer is False:
            # dropout of last hidden layer
            for j in range(len(self.slots)):
                layer[-1][j] = self.local_out_drop_op(layer[-1][j])
            layer[-1][-1] = self.global_out_drop_op(layer[-1][-1])

            # last_hidden_layer -> outputs
            outputs = []
            for i in range(len(self.slots) + 1):
                if self.use_dueling is False:
                    outputs.append(self.output_trans[i](layer[-1][i]))
                else:
                    if i < len(self.slots):
                        tmp_adv = self.output_trans_local_advantage(sorted_inputs[i])
                    else:
                        tmp_adv = self.output_trans_global_advantage(sorted_inputs[-1])
                    if self.dueling_share_last:
                        if i < len(self.slots):
                            cur_value = self.output_trans_local_value(layer[-1][i])
                            if self.shared_last_layer_use_bias:
                                cur_value = cur_value + nd.slice(self.value_bias_local.data(), begin=(i, ), end=(i + 1, ))
                        else:
                            cur_value = self.output_trans_global_value(layer[-1][i])
                    else:
                        cur_value = self.output_trans_value[i](layer[-1][i])
                    outputs.append(
                        cur_value +
                        tmp_adv - tmp_adv.mean(axis=1).reshape(
                            (tmp_adv.shape[0], 1)).broadcast_axes(axis=1, size=tmp_adv.shape[1]))
        else:
            outputs = []
            for i in range(len(self.slots)):
                output_i = self.output_trans_local(layer[-1][i])
                if self.shared_last_layer_use_bias:
                    output_i = output_i + self.output_trans_local_biases[i].data()
                outputs.append(output_i)
            outputs.append(self.output_trans_global(layer[-1][-1]))
        return nd.concat(*outputs, dim=1)


class DeepQNetwork(object):
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
                 message_embedding=None, state_feature=None, init_policy=None, shared_last_layer_use_bias=None, seed=None):
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

        self.qnet = self.create_ddq_network(prefix='qnet_')
        self.target = self.create_ddq_network(prefix='target_')

        self.trainer = gl.Trainer(params=self.qnet.collect_params(), optimizer='adam',
                                  optimizer_params=dict(learning_rate=self.learning_rate, wd=weight_regularizer))

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
        print network.collect_params()
        network.initialize(ctx=CTX)
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
            loss_vec = []
            outputs = self.qnet(inputs, loss_vec)
            loss = 0.
            for element in loss_vec:
                loss = loss + element
            # print 'loss_dropout:', loss
            td_error = nd.sum(data=outputs * action, axis=1) - sampled_q
            for i in range(self.minibatch_size):
                if nd.abs(td_error[i]) < 1.0:
                    loss = loss + 0.5 * nd.square(td_error[i])
                else:
                    loss = loss + nd.abs(td_error[i]) - 0.5
            # print loss
        loss.backward()
        self.trainer.step(batch_size=self.minibatch_size, ignore_stale_grad=True)

    def predict(self, inputs):
        assert mx.autograd.is_training() is False
        return self.qnet(nd.array(inputs, ctx=CTX)).asnumpy()

    def predict_target(self, inputs):
        if self.target_explore:
            with mx.autograd.train_mode():
                return self.target(nd.array(inputs, ctx=CTX)).asnumpy()
        else:
            assert mx.autograd.is_training() == False
            return self.target(nd.array(inputs, ctx=CTX)).asnumpy()

    def update_target_network(self):
        param_list_qnet = []
        param_list_target = []
        for key, value in self.qnet.collect_params().items():
            param_list_qnet.append(value)
        for key, value in self.target.collect_params().items():
            param_list_target.append(value)
        assert len(param_list_qnet) == len(param_list_target)

        for i in range(len(param_list_qnet)):
            # print param_list_target[i].name.lstrip('target'), param_list_qnet[i].name.lstrip('qnet')
            assert (param_list_target[i].name.lstrip('target') == param_list_qnet[i].name.lstrip('qnet'))
            param_list_target[i].set_data(param_list_target[i].data() * (1. - self.tau) +
                                          param_list_qnet[i].data() * self.tau)

    def copy_qnet_to_target(self):
        param_list_qnet = []
        param_list_target = []
        for key, value in self.qnet.collect_params().items():
            param_list_qnet.append(value)
        for key, value in self.target.collect_params().items():
            param_list_target.append(value)
        assert len(param_list_qnet) == len(param_list_target)

        for i in range(len(param_list_qnet)):
            assert param_list_target[i].name.strip('target') == param_list_qnet[i].name.strip('qnet')
            param_list_target[i].set_data(param_list_qnet[i].data())

    def load_network(self, load_filename):
        try:
            self.qnet.load_params(filename=load_filename + '_qnet', ctx=CTX)
            self.target.load_params(filename=load_filename + '_target', ctx=CTX)
            self.trainer.step(1, ignore_stale_grad=True)
            self.trainer.load_states(fname=load_filename + '_trainer')
            print "Successfully loaded:", load_filename
        except:
            try:
                init_policy_name = self.init_policy.replace('*', str(self.seed))
                print "Could not find old network weights({}), try self.init_policy({})".format(
                    load_filename, init_policy_name)
                need_dict = gl.ParameterDict()
                for key, value in self.qnet.collect_params().items():
                    if not key.endswith('_value_bias_local'):
                        need_dict._params[key] = value
                need_dict.load(filename=init_policy_name+ '_qnet', ctx=CTX, ignore_extra=True, restore_prefix='qnet_')

                need_dict = gl.ParameterDict()
                for key, value in self.target.collect_params().items():
                    if not key.endswith('_value_bias_local'):
                        need_dict._params[key] = value
                need_dict.load(filename=init_policy_name + '_target', ctx=CTX, ignore_extra=True, restore_prefix='target_')
                print "Successfully loaded:", self.init_policy
            except:
                print 'no init policy or cannot load it.'

    def save_network(self, save_filename):
        print 'Saving deepq-network...'
        self.qnet.save_params(filename=save_filename + '_qnet')
        self.target.save_params(filename=save_filename + '_target')
        self.trainer.save_states(fname=save_filename + '_trainer')
