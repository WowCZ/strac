###############################################################################
# PyDial: Multi-domain Statistical Spoken Dialogue System Software
###############################################################################
#
# Copyright 2015 - 2017
# Cambridge University Engineering Department Dialogue Systems Group
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###############################################################################

'''
A2CPolicy.py - Advantage Actor-Critic policy
==================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies:

    import :class:`Policy`
    import :class:`utils.ContextLogger`

.. warning::
        Documentation not done.


************************

'''
import copy
import os
import json
import numpy as np
import scipy
import scipy.signal
import cPickle as pickle
import random
import utils
import time
from utils import ContextLogger, DiaAct, Settings

import ontology.FlatOntologyManager as FlatOnt
# import tensorflow as tf
from DRL.replay_buffer_episode_a2c import ReplayBufferEpisode
from DRL.replay_prioritised_episode import ReplayPrioritisedEpisode
import DRL.utils as drlutils
import DRL.hack_strac as strac
import Policy
import SummaryAction
from Policy import TerminalAction, TerminalState
from DIP.DIP_parametrisation import DIP_state
import threading

lock = Settings.load_lock()

# --- for flattening the belief --- #
def flatten_belief(belief, domainUtil, merge=False):
    belief = belief.getDomainState(domainUtil.domainString)
    if isinstance(belief, TerminalState):
        if domainUtil.domainString == 'CamRestaurants':
            return [0] * 268
        elif domainUtil.domainString == 'CamHotels':
            return [0] * 111
        elif domainUtil.domainString == 'SFRestaurants':
            return [0] * 636
        elif domainUtil.domainString == 'SFHotels':
            return [0] * 438
        elif domainUtil.domainString == 'Laptops11':
            return [0] * 257
        elif domainUtil.domainString == 'TV':
            return [0] * 188

    policyfeatures = ['full', 'method', 'discourseAct', 'requested',
                      'lastActionInformNone', 'offerHappened', 'inform_info']

    flat_belief = []
    for feat in policyfeatures:
        add_feature = []
        if feat == 'full':
            # for slot in self.sorted_slots:
            for slot in domainUtil.ontology['informable']:
                for value in domainUtil.ontology['informable'][slot]:  # + ['**NONE**']:
                    add_feature.append(belief['beliefs'][slot][value])

                try:
                    add_feature.append(belief['beliefs'][slot]['**NONE**'])
                except:
                    add_feature.append(0.)  # for NONE
                try:
                    add_feature.append(belief['beliefs'][slot]['dontcare'])
                except:
                    add_feature.append(0.)  # for dontcare

        elif feat == 'method':
            add_feature = [belief['beliefs']['method'][method] for method in domainUtil.ontology['method']]
        elif feat == 'discourseAct':
            add_feature = [belief['beliefs']['discourseAct'][discourseAct]
                           for discourseAct in domainUtil.ontology['discourseAct']]
        elif feat == 'requested':
            add_feature = [belief['beliefs']['requested'][slot] \
                           for slot in domainUtil.ontology['requestable']]
        elif feat == 'lastActionInformNone':
            add_feature.append(float(belief['features']['lastActionInformNone']))
        elif feat == 'offerHappened':
            add_feature.append(float(belief['features']['offerHappened']))
        elif feat == 'inform_info':
            add_feature += belief['features']['inform_info']
        else:
            logger.error('Invalid feature name in config: ' + feat)

        flat_belief += add_feature

    return flat_belief


# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class STRACPolicy(Policy.Policy):
    '''Derived from :class:`Policy`
    '''
    def __init__(self, in_policy_file, out_policy_file, ontology, cfg, logger, SetObj, domainString='CamRestaurants', is_training=False):
        super(STRACPolicy, self).__init__(domainString, ontology, cfg, logger, SetObj, is_training)

        self.domainString = domainString
        self.domainUtil = FlatOnt.FlatDomainOntology(self.domainString, cfg, ontology.OntologyUtils, SetObj)
        self.in_policy_file = in_policy_file
        self.out_policy_file = out_policy_file
        self.is_training = is_training
        self.accum_belief = []

        self.prev_state_check = None
        self.ontology = ontology
        self.logger = logger
        self.SetObj =SetObj

        # parameter settings
        if 0:#cfg.has_option('dqnpolicy', 'n_in'): #ic304: this was giving me a weird error, disabled it until i can check it deeper
            self.n_in = cfg.getint('dqnpolicy', 'n_in')
        else:
            self.n_in = self.get_n_in(domainString)

        self.actor_critic_combine = True

        self.learning_rate = 0.001
        if cfg.has_option('dqnpolicy', 'learning_rate'):
            self.learning_rate = cfg.getfloat('dqnpolicy', 'learning_rate')

        self.tau = 0.001
        if cfg.has_option('dqnpolicy', 'tau'):
            self.tau = cfg.getfloat('dqnpolicy', 'tau')

        self.randomseed = 1234
        if cfg.has_option('GENERAL', 'seed'):
            self.randomseed = cfg.getint('GENERAL', 'seed')

        self.gamma = 1.0
        if cfg.has_option('dqnpolicy', 'gamma'):
            self.gamma = cfg.getfloat('dqnpolicy', 'gamma')
        self.gamma = 0.9

        self.regularisation = 'l2'
        if cfg.has_option('dqnpolicy', 'regularisation'):
            self.regularisation = cfg.get('dqnpolicy', 'regulariser')

        self.exploration_type = 'e-greedy'  # Boltzman
        if cfg.has_option('dqnpolicy', 'exploration_type'):
            self.exploration_type = cfg.get('dqnpolicy', 'exploration_type')

        self.episodeNum = 1000
        if cfg.has_option('dqnpolicy', 'episodeNum'):
            self.episodeNum = cfg.getfloat('dqnpolicy', 'episodeNum')

        self.maxiter = 5000
        if cfg.has_option('dqnpolicy', 'maxiter'):
            self.maxiter = cfg.getfloat('dqnpolicy', 'maxiter')
        self.maxiter += 2000

        self.epsilon = 1
        if cfg.has_option('dqnpolicy', 'epsilon'):
            self.epsilon = cfg.getfloat('dqnpolicy', 'epsilon')

        self.epsilon_start = 1
        if cfg.has_option('dqnpolicy', 'epsilon_start'):
            self.epsilon_start = cfg.getfloat('dqnpolicy', 'epsilon_start')
        # self.epsilon_start = 0.9

        self.epsilon_end = 1
        if cfg.has_option('dqnpolicy', 'epsilon_end'):
            self.epsilon_end = cfg.getfloat('dqnpolicy', 'epsilon_end')

        self.save_step = 100
        if cfg.has_option('policy', 'save_step'):
            self.save_step = cfg.getint('policy', 'save_step')

        self.priorProbStart = 1.0
        if cfg.has_option('dqnpolicy', 'prior_sample_prob_start'):
            self.priorProbStart = cfg.getfloat('dqnpolicy', 'prior_sample_prob_start')

        self.priorProbEnd = 0.1
        if cfg.has_option('dqnpolicy', 'prior_sample_prob_end'):
            self.priorProbEnd = cfg.getfloat('dqnpolicy', 'prior_sample_prob_end')

        self.policyfeatures = []
        if cfg.has_option('dqnpolicy', 'features'):
            self.logger.info('Features: ' + str(cfg.get('dqnpolicy', 'features')))
            self.policyfeatures = json.loads(cfg.get('dqnpolicy', 'features'))

        self.max_k = 5
        if cfg.has_option('dqnpolicy', 'max_k'):
            self.max_k = cfg.getint('dqnpolicy', 'max_k')

        self.learning_algorithm = 'drl'
        if cfg.has_option('dqnpolicy', 'learning_algorithm'):
            self.learning_algorithm = cfg.get('dqnpolicy', 'learning_algorithm')
            self.logger.info('Learning algorithm: ' + self.learning_algorithm)

        self.minibatch_size = 32
        if cfg.has_option('dqnpolicy', 'minibatch_size'):
            self.minibatch_size = cfg.getint('dqnpolicy', 'minibatch_size')

        self.capacity = 1000  # max(self.minibatch_size, 2000)
        if cfg.has_option('dqnpolicy', 'capacity'):
            self.capacity = max(cfg.getint('dqnpolicy', 'capacity'), 2000)

        self.replay_type = 'vanilla'
        if cfg.has_option('dqnpolicy', 'replay_type'):
            self.replay_type = cfg.get('dqnpolicy', 'replay_type')

        self.architecture = 'vanilla'
        if cfg.has_option('dqnpolicy', 'architecture'):
            self.architecture = cfg.get('dqnpolicy', 'architecture')

        self.q_update = 'double'
        if cfg.has_option('dqnpolicy', 'q_update'):
            self.q_update = cfg.get('dqnpolicy', 'q_update')

        self.h1_size = 130
        if cfg.has_option('dqnpolicy', 'h1_size'):
            self.h1_size = cfg.getint('dqnpolicy', 'h1_size')

        self.h1_drop = None
        if cfg.has_option('dqnpolicy', 'h1_drop'):
            self.h1_drop = cfg.getfloat('dqnpolicy', 'h1_drop')

        self.h2_size = 130
        if cfg.has_option('dqnpolicy', 'h2_size'):
            self.h2_size = cfg.getint('dqnpolicy', 'h2_size')

        self.h2_drop = None
        if cfg.has_option('dqnpolicy', 'h2_drop'):
            self.h2_drop = cfg.getfloat('dqnpolicy', 'h2_drop')

        self.nature_mode = None
        if cfg.has_option('dqnpolicy', 'nature_mode'):
            self.nature_mode = cfg.getboolean('dqnpolicy', 'nature_mode')

        self.madqn_hidden_layers = None
        if cfg.has_option('dqnpolicy', 'madqn_hidden_layers'):
            self.madqn_hidden_layers = cfg.getint('dqnpolicy', 'madqn_hidden_layers')

        self.madqn_local_hidden_units = None
        if cfg.has_option('dqnpolicy', 'madqn_local_hidden_units'):
            self.madqn_local_hidden_units = cfg.get('dqnpolicy', 'madqn_local_hidden_units')
            self.madqn_local_hidden_units = eval(self.madqn_local_hidden_units)

        self.madqn_local_dropouts = None
        if cfg.has_option('dqnpolicy', 'madqn_local_dropouts'):
            self.madqn_local_dropouts = cfg.get('dqnpolicy', 'madqn_local_dropouts')
            self.madqn_local_dropouts = eval(self.madqn_local_dropouts)

        self.madqn_global_hidden_units = None
        if cfg.has_option('dqnpolicy', 'madqn_global_hidden_units'):
            self.madqn_global_hidden_units = cfg.get('dqnpolicy', 'madqn_global_hidden_units')
            self.madqn_global_hidden_units = eval(self.madqn_global_hidden_units)

        self.madqn_global_dropouts = None
        if cfg.has_option('dqnpolicy', 'madqn_global_dropouts'):
            self.madqn_global_dropouts = cfg.get('dqnpolicy', 'madqn_global_dropouts')
            self.madqn_global_dropouts = eval(self.madqn_global_dropouts)

        self.madqn_private_rate = None
        if cfg.has_option('dqnpolicy', 'madqn_private_rate'):
            self.madqn_private_rate = cfg.getfloat('dqnpolicy', 'madqn_private_rate')

        self.madqn_sort_input_vec = False
        if cfg.has_option('dqnpolicy', 'madqn_sort_input_vec'):
            self.madqn_sort_input_vec = cfg.getboolean('dqnpolicy', 'madqn_sort_input_vec')

        self.madqn_share_last_layer = False
        if cfg.has_option('dqnpolicy', 'madqn_share_last_layer'):
            self.madqn_share_last_layer = cfg.getboolean('dqnpolicy', 'madqn_share_last_layer')

        self.madqn_shared_last_layer_use_bias = True
        if cfg.has_option('dqnpolicy', 'madqn_shared_last_layer_use_bias'):
            self.madqn_shared_last_layer_use_bias = cfg.getboolean('dqnpolicy', 'madqn_shared_last_layer_use_bias')

        self.madqn_recurrent_mode = False
        if cfg.has_option('dqnpolicy', 'madqn_recurrent_mode'):
            self.madqn_recurrent_mode = cfg.getboolean('dqnpolicy', 'madqn_recurrent_mode')

        self.madqn_input_comm = True
        if cfg.has_option('dqnpolicy', 'madqn_input_comm'):
            self.madqn_input_comm = cfg.getboolean('dqnpolicy', 'madqn_input_comm')

        self.madqn_target_explore = False
        if cfg.has_option('dqnpolicy', 'madqn_target_explore'):
            self.madqn_target_explore = cfg.getboolean('dqnpolicy', 'madqn_target_explore')

        self.madqn_concrete_share_rate = False
        if cfg.has_option('dqnpolicy', 'madqn_concrete_share_rate'):
            self.madqn_concrete_share_rate = cfg.getboolean('dqnpolicy', 'madqn_concrete_share_rate')

        self.madqn_dropout_regularizer = 0.
        if cfg.has_option('dqnpolicy', 'madqn_dropout_regularizer'):
            self.madqn_dropout_regularizer = cfg.getfloat('dqnpolicy', 'madqn_dropout_regularizer')

        self.madqn_weight_regularizer = 0.
        if cfg.has_option('dqnpolicy', 'madqn_weight_regularizer'):
            self.madqn_weight_regularizer = cfg.getfloat('dqnpolicy', 'madqn_weight_regularizer')

        self.madqn_non_local_mode = False
        if cfg.has_option('dqnpolicy', 'madqn_non_local_mode'):
            self.madqn_non_local_mode = cfg.getboolean('dqnpolicy', 'madqn_non_local_mode')

        self.madqn_block_mode = False
        if cfg.has_option('dqnpolicy', 'madqn_block_mode'):
            self.madqn_block_mode = cfg.getboolean('dqnpolicy', 'madqn_block_mode')

        self.madqn_slots_comm = True
        if cfg.has_option('dqnpolicy', 'madqn_slots_comm'):
            self.madqn_slots_comm = cfg.getboolean('dqnpolicy', 'madqn_slots_comm')

        self.madqn_use_dueling = False
        if cfg.has_option('dqnpolicy', 'madqn_use_dueling'):
            self.madqn_use_dueling = cfg.getboolean('dqnpolicy', 'madqn_use_dueling')

        self.madqn_topo_learning_mode = False
        if cfg.has_option('dqnpolicy', 'madqn_topo_learning_mode'):
            self.madqn_topo_learning_mode = cfg.getboolean('dqnpolicy', 'madqn_topo_learning_mode')

        self.madqn_message_embedding = False
        if cfg.has_option('dqnpolicy', 'madqn_message_embedding'):
            self.madqn_message_embedding = cfg.getboolean('dqnpolicy', 'madqn_message_embedding')

        self.madqn_dueling_share_last = False
        if cfg.has_option('dqnpolicy', 'madqn_dueling_share_last'):
            self.madqn_dueling_share_last = cfg.getboolean('dqnpolicy', 'madqn_dueling_share_last')

        self.state_feature = 'vanilla'
        if cfg.has_option('dqnpolicy', 'state_feature'):
            self.state_feature = cfg.get('dqnpolicy', 'state_feature')

        self.init_policy = None
        if cfg.has_option('dqnpolicy', 'init_policy'):
            self.init_policy = cfg.get('dqnpolicy', 'init_policy')

        self.training_frequency = 2
        if cfg.has_option('dqnpolicy', 'training_frequency'):
            self.training_frequency = cfg.getint('dqnpolicy', 'training_frequency')

        self.importance_sampling = True
        if cfg.has_option('dqnpolicy', 'importance_sampling'):
            self.importance_sampling = cfg.getint('dqnpolicy', 'importance_sampling')

        # domain specific parameter settings (overrides general policy parameter settings)
        if cfg.has_option('dqnpolicy_' + domainString, 'n_in'):
            self.n_in = cfg.getint('dqnpolicy_' + domainString, 'n_in')

        if cfg.has_option('dqnpolicy_' + domainString, 'learning_rate'):
            self.learning_rate = cfg.getfloat('dqnpolicy_' + domainString, 'learning_rate')

        if cfg.has_option('dqnpolicy_' + domainString, 'tau'):
            self.tau = cfg.getfloat('dqnpolicy_' + domainString, 'tau')

        if cfg.has_option('dqnpolicy_' + domainString, 'gamma'):
            self.gamma = cfg.getfloat('dqnpolicy_' + domainString, 'gamma')

        if cfg.has_option('dqnpolicy_' + domainString, 'regularisation'):
            self.regularisation = cfg.get('dqnpolicy_' + domainString, 'regulariser')

        if cfg.has_option('dqnpolicy_' + domainString, 'exploration_type'):
            self.exploration_type = cfg.get('dqnpolicy_' + domainString, 'exploration_type')

        if cfg.has_option('dqnpolicy_' + domainString, 'episodeNum'):
            self.episodeNum = cfg.getfloat('dqnpolicy_' + domainString, 'episodeNum')

        if cfg.has_option('dqnpolicy_' + domainString, 'maxiter'):
            self.maxiter = cfg.getfloat('dqnpolicy_' + domainString, 'maxiter')

        if cfg.has_option('dqnpolicy_' + domainString, 'epsilon'):
            self.epsilon = cfg.getfloat('dqnpolicy_' + domainString, 'epsilon')

        if cfg.has_option('dqnpolicy_' + domainString, 'epsilon_start'):
            self.epsilon_start = cfg.getfloat('dqnpolicy_' + domainString, 'epsilon_start')

        if cfg.has_option('dqnpolicy_' + domainString, 'epsilon_end'):
            self.epsilon_end = cfg.getfloat('dqnpolicy_' + domainString, 'epsilon_end')

        if cfg.has_option('policy_' + domainString, 'save_step'):
            self.save_step = cfg.getint('policy_' + domainString, 'save_step')

        if cfg.has_option('dqnpolicy_' + domainString, 'prior_sample_prob_start'):
            self.priorProbStart = cfg.getfloat('dqnpolicy_' + domainString, 'prior_sample_prob_start')

        if cfg.has_option('dqnpolicy_' + domainString, 'prior_sample_prob_end'):
            self.priorProbEnd = cfg.getfloat('dqnpolicy_' + domainString, 'prior_sample_prob_end')

        if cfg.has_option('dqnpolicy_' + domainString, 'features'):
            self.logger.info('Features: ' + str(cfg.get('dqnpolicy_' + domainString, 'features')))
            self.policyfeatures = json.loads(cfg.get('dqnpolicy_' + domainString, 'features'))

        if cfg.has_option('dqnpolicy_' + domainString, 'max_k'):
            self.max_k = cfg.getint('dqnpolicy_' + domainString, 'max_k')

        if cfg.has_option('dqnpolicy_' + domainString, 'learning_algorithm'):
            self.learning_algorithm = cfg.get('dqnpolicy_' + domainString, 'learning_algorithm')
            self.logger.info('Learning algorithm: ' + self.learning_algorithm)

        if cfg.has_option('dqnpolicy_' + domainString, 'minibatch_size'):
            self.minibatch_size = cfg.getint('dqnpolicy_' + domainString, 'minibatch_size')

        if cfg.has_option('dqnpolicy_' + domainString, 'capacity'):
            self.capacity = max(cfg.getint('dqnpolicy_' + domainString, 'capacity'), 2000)

        if cfg.has_option('dqnpolicy_' + domainString, 'replay_type'):
            self.replay_type = cfg.get('dqnpolicy_' + domainString, 'replay_type')

        if cfg.has_option('dqnpolicy_' + domainString, 'architecture'):
            self.architecture = cfg.get('dqnpolicy_' + domainString, 'architecture')

        if cfg.has_option('dqnpolicy_' + domainString, 'q_update'):
            self.q_update = cfg.get('dqnpolicy_' + domainString, 'q_update')

        if cfg.has_option('dqnpolicy_' + domainString, 'h1_size'):
            self.h1_size = cfg.getint('dqnpolicy_' + domainString, 'h1_size')

        if cfg.has_option('dqnpolicy_' + domainString, 'h1_drop'):
            self.h1_drop = cfg.getfloat('dqnpolicy_' + domainString, 'h1_drop')

        if cfg.has_option('dqnpolicy_' + domainString, 'h2_size'):
            self.h2_size = cfg.getint('dqnpolicy_' + domainString, 'h2_size')

        if cfg.has_option('dqnpolicy_' + domainString, 'h2_drop'):
            self.h2_drop = cfg.getfloat('dqnpolicy_' + domainString, 'h2_drop')


        if cfg.has_option('dqnpolicy_' + domainString, 'training_frequency'):
            self.training_frequency = cfg.getint('dqnpolicy_' + domainString, 'training_frequency')

        """
        self.shuffle = False
        if cfg.has_option('dqnpolicy_'+domainString, 'experience_replay'):
            self.shuffle = cfg.getboolean('dqnpolicy_'+domainString, 'experience_replay')
        if not self.shuffle:
            # If we don't use experience replay, we don't need to maintain
            # sliding window of experiences with maximum capacity.
            # We only need to maintain the data of minibatch_size
            self.capacity = self.minibatch_size
        """

        self.episode_ave_max_q = []

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # init session
        # self.sess = tf.Session()
        # with tf.device("/cpu:0"):

        np.random.seed(self.randomseed)
        # tf.set_random_seed(self.randomseed)

        # initialise an replay buffer
        if self.replay_type == 'vanilla':
            self.episodes[self.domainString] = ReplayBufferEpisode(self.capacity, self.minibatch_size, self.randomseed)
        elif self.replay_type == 'prioritized':
            self.episodes[self.domainString] = ReplayPrioritisedEpisode(self.capacity, self.minibatch_size, self.randomseed)

        self.samplecount = 0
        self.episodecount = 0

        # construct the models
        self.state_dim = self.n_in
        self.summaryaction = SummaryAction.SummaryAction(domainString, self.ontology, self.SetObj)
        self.action_dim = len(self.summaryaction.action_names)
        action_bound = len(self.summaryaction.action_names)
        self.stats = [0 for _ in range(self.action_dim)]

        import tube
        self.strac = strac.STRACNetwork(self.state_dim, self.action_dim, \
                                    self.learning_rate, self.tau, action_bound, self.minibatch_size,
                                    self.architecture, self.h1_size, self.h1_drop,
                                    self.h2_size, self.h2_drop, self.domainString,
                                    self.madqn_hidden_layers,
                                    self.madqn_local_hidden_units, self.madqn_local_dropouts,
                                    self.madqn_global_hidden_units, self.madqn_global_dropouts,
                                    self.madqn_private_rate, self.madqn_sort_input_vec,
                                    self.madqn_share_last_layer, self.madqn_recurrent_mode,
                                    self.madqn_input_comm, self.madqn_target_explore,
                                    concrete_share_rate=self.madqn_concrete_share_rate,
                                    dropout_regularizer=self.madqn_dropout_regularizer,
                                    weight_regularizer=self.madqn_weight_regularizer,
                                    non_local_mode=self.madqn_non_local_mode,
                                    block_mode=self.madqn_block_mode,
                                    slots_comm=self.madqn_slots_comm,
                                    topo_learning_mode=self.madqn_topo_learning_mode,
                                    use_dueling=self.madqn_use_dueling,
                                    dueling_share_last=self.madqn_dueling_share_last,
                                    message_embedding=self.madqn_message_embedding,
                                    state_feature=self.state_feature,
                                    init_policy=self.init_policy,
                                    shared_last_layer_use_bias=self.madqn_shared_last_layer_use_bias,
                                    seed=tube.seed)

        lock.acquire()
        self.loadPolicy(self.in_policy_file)
        self.savePolicyInc()
        lock.release()
        print(self.domainString + ' loaded replay size: ' + str(self.episodes[self.domainString].size()))

        Settings.load_policy(self.strac, threading.currentThread().getName())

    def get_n_in(self, domain_string):
        if domain_string == 'CamRestaurants':
            return 268
        elif domain_string == 'CamHotels':
            return 111
        elif domain_string == 'SFRestaurants':
            return 636
        elif domain_string == 'SFHotels':
            return 438
        elif domain_string == 'Laptops6':
            return 268 # ic340: this is wrong
        elif domain_string == 'Laptops11':
            return 257
        elif domain_string is 'TV':
            return 188
        else:
            print 'DOMAIN {} SIZE NOT SPECIFIED, PLEASE DEFINE n_in'.format(domain_string)

    def act_on(self, state, hyps=None):
        if self.lastSystemAction is None and self.startwithhello:
            systemAct, nextaIdex = 'hello()', -1
        else:
            systemAct, nextaIdex = self.nextAction(state)
        self.lastSystemAction = systemAct
        self.summaryAct = nextaIdex
        self.prevbelief = state

        systemAct = DiaAct.DiaAct(systemAct)
        return systemAct

    def record(self, reward, domainInControl=None, weight=None, state=None, action=None):
        if domainInControl is None:
            domainInControl = self.domainString
        if self.actToBeRecorded is None:
            self.actToBeRecorded = self.summaryAct

        if state is None:
            state = self.prevbelief
        if action is None:
            action = self.actToBeRecorded

        cState, cAction = self.convertStateAction(state, action)

        # normalising total return to -1~1
        reward /= 20.0
        cur_cState = np.vstack([np.expand_dims(x, 0) for x in [cState]])

        prob, value = self.strac.predict_action_value(cur_cState)
        policy_mu = prob[0][cAction]  # self.strac.getPolicy([cState])[0][0][cAction]

        if self.replay_type == 'vanilla':
            self.episodes[domainInControl].record(state=cState, \
                    state_ori=state, action=cAction, reward=reward, value=value[0], distribution=policy_mu)
        elif self.replay_type == 'prioritized':
            self.episodes[domainInControl].record(state=cState, \
                    state_ori=state, action=cAction, reward=reward, value=value[0], distribution=policy_mu)

        self.actToBeRecorded = None

    def finalizeRecord(self, reward, domainInControl=None):
        if domainInControl is None:
            domainInControl = self.domainString
        if self.episodes[domainInControl] is None:
            self.logger.warning("record attempted to be finalized for domain where nothing has been recorded before")
            return

        # normalising total return to -1~1
        reward /= 20.0

        terminal_state, terminal_action = self.convertStateAction(TerminalState(), TerminalAction())

        value = 0.0  # not effect on experience replay

        def calculate_advantage(r_episode, v_episode):
            #########################################################################
            # Here we take the rewards and values from the rollout, and use them to
            # generate the advantage and discounted returns.
            # The advantage function uses "Generalized Advantage Estimation"
            bootstrap_value = 0.0
            self.v_episode_plus = np.asarray(v_episode + [bootstrap_value])
            advantage = r_episode + self.gamma * self.v_episode_plus[1:] - self.v_episode_plus[:-1]
            advantage = discount(advantage,self.gamma)
            #########################################################################
            return advantage

        if self.replay_type == 'vanilla':
            self.episodes[domainInControl].record(state=terminal_state, \
                    state_ori=TerminalState(), action=terminal_action, reward=reward, value=value, terminal=True, distribution=None)
        elif self.replay_type == 'prioritized':
            episode_r, episode_v = self.episodes[domainInControl].record_final_and_get_episode(state=terminal_state, \
                    state_ori=TerminalState(), action=terminal_action, reward=reward, value=value, terminal=True, distribution=None)

            # TD_error is a list of td error in the current episode
            TD_error = calculate_advantage(episode_r, episode_v)
            episodic_TD = np.mean(np.absolute(TD_error))
            self.episodes[domainInControl].insertPriority(episodic_TD)

        self.samplecount += 1

    def convertStateAction(self, state, action):
        '''
        nnType = 'dnn'
        #nnType = 'rnn'
        # expand one dimension to match the batch size of 1 at axis 0
        if nnType == 'rnn':
            belief = np.expand_dims(belief,axis=0)
        '''
        if isinstance(state, TerminalState):
            if self.domainUtil.domainString == 'CamRestaurants':
                return [0] * 268, action
            elif self.domainUtil.domainString == 'CamHotels':
                return [0] * 111, action
            elif self.domainUtil.domainString == 'SFRestaurants':
                return [0] * 633, action
            elif self.domainUtil.domainString == 'SFHotels':
                return [0] * 438, action
            elif self.domainUtil.domainString == 'Laptops11':
                return [0] * 257, action
            elif self.domainUtil.domainString == 'TV':
                return [0] * 188, action
        elif self.state_feature != 'dip':
            # print(threading.currentThread().getName() + ' Feature ' + str(self.domainUtil))
            flat_belief = flatten_belief(state, self.domainUtil, self.logger)
            self.prev_state_check = flat_belief
            return flat_belief, action
        elif self.state_feature == 'dip':
            tmp = [0, 0, 0]
            for i in range(len(self.stats) - 7):
                tmp[i % 3] += self.stats[i]
            for i in range(len(self.stats) - 7, len(self.stats)):
                tmp.append(self.stats[i])

            af = 1. / (1. + np.array(tmp))
            dip_state = DIP_state(belief=state,
                                  ontology=self.ontology,
                                  domainString=self.domainString,
                                  action_freq=af)
            beliefVec = []
            for slot in self.domainUtil.ontology['informable']:
                if slot != 'name' and slot != 'price':
                    beliefVec.extend(dip_state.DIP_state[slot])
                    # print slot, ":", len(beliefVec)
            beliefVec.extend(np.concatenate(
                [dip_state.DIP_state['joint'], dip_state.DIP_state['general']]))

            return beliefVec, action

    def nextAction(self, beliefstate):
        '''
        select next action

        :param beliefstate:
        :param hyps:
        :returns: (int) next summary action
        '''
        if self.state_feature == 'dip':
            tmp = [0, 0, 0]
            for i in range(len(self.stats) - 7):
                tmp[i % 3] += self.stats[i]
            for i in range(len(self.stats) - 7, len(self.stats)):
                tmp.append(self.stats[i])

            af = 1. / (1. + np.array(tmp))
            dip_state = DIP_state(belief=beliefstate, ontology=self.ontology, domainString=self.domainString, action_freq=af)
            beliefVec = []
            for slot in self.domainUtil.ontology['informable']:
                if slot != 'name' and slot != 'price':
                    beliefVec.extend(dip_state.DIP_state[slot])
                    # print slot, ":", len(beliefVec)
            beliefVec.extend(np.concatenate([dip_state.DIP_state['joint'], dip_state.DIP_state['general']]))
        else:
            # print(threading.currentThread().getName() + ' Feature ' + str(self.domainUtil.domainString))
            beliefVec = flatten_belief(beliefstate, self.domainUtil, self.logger)

        execMask = self.summaryaction.getExecutableMask(beliefstate, self.lastSystemAction)

        action_prob, value = self.strac.predict_action_value(np.reshape(beliefVec, (1, len(beliefVec))))# + (1. / (1. + i + j))
        action_Q_admissible = np.add(action_prob, np.array(execMask)) # enforce Q of inadmissible actions to be -inf
        action_prob = drlutils.softmax(action_Q_admissible)[0]
        greedyNextaIdex = np.argmax(action_prob)

        if self.is_training and self.SetObj.random.rand() < self.epsilon:
            # nextaIdex = np.random.choice(len(action_prob), p=action_prob)
            admissible = [i for i, x in enumerate(execMask) if x == 0.0]
            random.shuffle(admissible)
            nextaIdex = admissible[0]
        else:
            # nextaIdex = np.random.choice(len(action_prob), p=action_prob)
            nextaIdex = greedyNextaIdex

        self.stats[nextaIdex] += 1
        summaryAct = self.summaryaction.action_names[nextaIdex]
        beliefstate = beliefstate.getDomainState(self.domainUtil.domainString)
        masterAct = self.summaryaction.Convert(beliefstate, summaryAct, self.lastSystemAction)
        return masterAct, nextaIdex

    def _sample_and_updateV(self):
        s_batch, s_ori_batch, a_batch, r_batch, s2_batch, s2_ori_batch, t_batch, idx_batch, v_batch, mu_policy = \
            self.episodes[self.domainString].sample_batch()

        # print('HackA2Cpolicy.py #784')
        # print(s_batch)
        s_batch = np.concatenate(np.array(s_batch), axis=0)

        def Vtrace(mu_policy, mu_hat, c_hat, nstep):
            ngamma = np.asarray([self.gamma**i for i in range(nstep)])
            mu_policy = np.asarray(mu_policy)
            lenghts = []  # to properly divde on dialogues pi_policy later on
            for mu in mu_policy:
                lenghts.append(len(mu))
            mu_policy = np.concatenate(np.array(mu_policy), axis=0).tolist()  # concatenate all behavioral probs
            lengths = np.cumsum(lenghts)  # time steps for ends of dialogues

            pi_policy, value = self.strac.predict_action_value(s_batch)  # policy given s_t

            columns = np.asarray([np.concatenate(a_batch, axis=0).tolist()]).astype(int)  # actions taken at s_t
            rows = np.asarray([ii for ii in range(len(pi_policy))])
            pi_policy = pi_policy[rows, columns].astype(np.longdouble)[0]
            weights = np.asarray(pi_policy) / np.asarray(mu_policy)


            c_hat = (c_hat*np.ones_like(weights.shape)).tolist()
            mu_hat = (mu_hat*np.ones_like(weights.shape)).tolist()
            c_whole = np.minimum(c_hat, weights.tolist()).tolist()
            mu_whole = np.minimum(mu_hat, weights.tolist()).tolist()

            c = []
            mu = []
            for ii in range(len(lengths)):
                if ii == 0:
                    c.append(c_whole[0:lengths[ii]])
                    mu.append(mu_whole[0:lengths[ii]])
                else:
                    c.append(c_whole[lengths[ii-1]:lengths[ii]])
                    mu.append(mu_whole[lengths[ii-1]:lengths[ii]])

            sigmaV = []
            for item_r, item_v, item_mu in zip(r_batch, v_batch, mu):
                bootstrap_value = 0.0
                item_r = np.asarray(item_r)
                item_mu = np.asarray(item_mu)
                item_v = np.asarray(item_v + [bootstrap_value])
                tmp = item_mu*(item_r + self.gamma * item_v[1:] - item_v[:-1])
                sigmaV.append(tmp.tolist())

            V_trace = []
            for item_c, item_sigmaV, item_v in zip(c, sigmaV, v_batch):
                c_mul_item = []
                trace_item = []
                for idx in list(reversed(range(len(item_c)))):
                    if len(item_c) - idx <= nstep:
                        tmp = 1.0
                        c_mul_subitem = [1.0]
                        for item in item_c[idx:]:
                            tmp *= item
                            c_mul_subitem.append(tmp)
                        trace_item_value = np.sum(np.asarray(ngamma[:len(item_c)-idx])*np.asarray(c_mul_subitem[:-1])*np.asarray(item_sigmaV[idx:]))
                        c_mul_item.append(c_mul_subitem)
                        trace_item.append(trace_item_value)
                    else:
                        tmp = [1.0] + (item_c[idx]*np.asarray(c_mul_item[-1][:-1])).tolist()
                        trace_item_value = np.sum(np.asarray(ngamma)*np.asarray(tmp[:-1])*np.asarray(item_sigmaV[idx:idx+nstep]))
                        c_mul_item.append(tmp)
                        trace_item.append(trace_item_value)
                trace_item = np.array(list(reversed(trace_item))) + np.array(item_v)
                V_trace.append(trace_item.tolist())

            advantage = []
            for ii in range(len(lengths)):  # over dialogues
                # first case
                if ii == 0:
                    tmp = np.asarray(r_batch[ii]) + self.gamma*np.asarray(V_trace[ii][1:] + [0.0]) - np.asarray(value[0:lengths[ii]])
                    tmp = np.asarray(mu[ii]) * tmp
                    advantage.append(tmp.tolist())
                else:
                    tmp = np.asarray(r_batch[ii]) + self.gamma*np.asarray(V_trace[ii][1:] + [0.0]) - np.asarray(value[lengths[ii-1]:lengths[ii]])
                    tmp = np.asarray(mu[ii]) * tmp
                    advantage.append(tmp.tolist())

            V_trace = np.concatenate(np.array(V_trace), axis=0).tolist()
            mu = np.concatenate(np.array(mu), axis=0)
            advantage = np.concatenate(np.array(advantage), axis=0).tolist()
            return V_trace, mu, advantage

        V_trace, mu, advantage = Vtrace(mu_policy, mu_hat=1.0, c_hat=1.0, nstep=2)
        a_batch_one_hot = np.eye(self.action_dim)[np.concatenate(a_batch, axis=0).tolist()]

        return s_batch, a_batch_one_hot, V_trace, advantage

    def train(self):
        '''
        call this function when the episode ends
        '''

        if not self.is_training:
            self.logger.info("Not in training mode")
            return
        else:
            self.logger.info("Update a2c policy parameters.")

        self.episodecount += 1
        self.logger.info("Sample Num so far: %s" % self.samplecount)
        self.logger.info("Episode Num so far: %s" % self.episodecount)

        Settings.add_count()
        globalEpisodeCount = copy.deepcopy(Settings.get_count())
        self.loadLastestPolicy()

        if self.samplecount >= self.minibatch_size * 1 and globalEpisodeCount % self.training_frequency == 0:
            self.logger.info('start training...')

            assert len(Settings.global_policysaver) == Settings.global_threadsnum
            # self.dqn.reset_noise()
            total_batch_size = 0
            for k, thread_policy in Settings.global_policysaver.items():
                s_batch, a_batch_one_hot, V_trace, advantage = Settings.global_hackpolicysaver[k]._sample_and_updateV()
                grad, batch_size = thread_policy.train(s_batch, a_batch_one_hot, V_trace, advantage)
                total_batch_size += batch_size
                Settings.load_grad(grad, k)

            assert len(Settings.global_gradsaver) == Settings.global_threadsnum
            grads_list = Settings.grad_sum()
            self._load_and_update(grads_list, total_batch_size)
            self.savePolicyInc()

    def _load_and_update(self, grads_list, batch_size):
        # grads_list = grads_list.tolist()
        values = []
        for name, value in self.strac.actorcritic.collect_params().items():
            if name.find('batchnorm') < 0:
                values.append(value)

        assert len(grads_list) == len(values)
        for i in range(len(grads_list)):
            for arr in values[i]._check_and_get(values[i]._grad, list):
                arr[:] = grads_list[i]

        self.strac.trainer.step(batch_size=batch_size, ignore_stale_grad=True)

    def savePolicy(self, FORCE_SAVE=False):
        """
        Does not use this, cause it will be called from agent after every episode.
        we want to save the policy only periodically.
        """
        tmp = self.out_policy_file.split('/')
        filename = tmp[-1].split('-')[1:-1]
        filename.append('shared-lastest')
        filename = '-'.join(filename)
        tmp[-1] = filename
        filename = '/'.join(tmp)
        self.strac.save_network(filename + '.dqn.ckpt')

    def savePolicyInc(self, FORCE_SAVE=False):
        """
        save model and replay buffer
        """
        # print('Save the latest parameters...')
        self.savePolicy(FORCE_SAVE=False)

    def saveTrainerAndReplay(self):
        self.strac.save_trainer(self.out_policy_file + '.dqn.ckpt')

        f = open(self.out_policy_file + '-' + self.domainString + '.episode', 'wb')
        for obj in [self.samplecount, self.episodes[self.domainString]]:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        print('save replay as ' + self.out_policy_file + '-' + self.domainString + '.episode')
        # logger.info("Saving model to %s and replay buffer..." % save_path)

    def loadLastestPolicy(self):
        # print('Load the latest parameters...')
        tmp = self.out_policy_file.split('/')
        filename = tmp[-1].split('-')[1:-1]
        filename.append('shared-lastest')
        filename = '-'.join(filename)
        tmp[-1] = filename
        filename = '/'.join(tmp)

        self.strac.load_test_network(filename + '.dqn.ckpt')

    def loadPolicy(self, filename):
        """
        load model and replay buffer
        """
        tmp = self.out_policy_file.split('/')
        nfilename = tmp[-1].split('-')[1:-1]
        nfilename.append('shared-lastest')
        nfilename = '-'.join(nfilename)
        tmp[-1] = nfilename
        nfilename = '/'.join(tmp)

        tmp = filename.split('-')

        if tmp[-1].split('.')[-1] == '0':
            self.strac.load_network('init.a2c')
        else:
            # lock.acquire()
            self.strac.load_test_network(nfilename + '.dqn.ckpt')
            self.strac.load_trainer(filename + '.dqn.ckpt')
            # lock.release()

        # load replay buffer
        try:
            print 'load from: ', filename
            print(filename + '-' + self.domainString +'.episode')
            f = open(filename + '-' + self.domainString +'.episode', 'rb')
            loaded_objects = []
            for i in range(2):  # load nn params and collected data
                loaded_objects.append(pickle.load(f))
            self.samplecount = int(loaded_objects[0])
            self.episodes[self.domainString] = copy.deepcopy(loaded_objects[1])
            print("Loading both model from %s and replay buffer..." % filename)
            f.close()
        except:
            print("Loading only models...")

    def restart(self):
        self.summaryAct = None
        self.lastSystemAction = None
        self.prevbelief = None
        self.actToBeRecorded = None
        self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * float(
            self.episodeNum + self.episodecount) / float(self.maxiter)
        self.episode_ave_max_q = []

# END OF FILE
