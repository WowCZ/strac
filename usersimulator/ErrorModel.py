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
ErrorModel.py - error simulation
===============================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`usersimulator.ConfidenceScorer` |.|
    import :mod:`utils.DiaAct` |.|
    import :mod:`utils.ContextLogger`

************************

''' 

__author__ = "cued_dialogue_systems_group"

import ConfidenceScorer
import NBestGenerator
import ConfusionModel
from utils import DiaAct
from utils import ContextLogger
import os
# logger = ContextLogger.getLogger('')



class DomainsErrorSimulator(object):
    def __init__(self, domain_string, error_rate, ontology, cfg, logger, SetObj):
        """
        Single domain error simulation module. Operates on semantic acts.
        :param: (str) conf_scorer_name
        :returns None:
        """

        self.error_rate = error_rate
        self.logger = logger
        self.cfg = cfg
        self.ontology = ontology
        self.SetObj = SetObj

        # DEFAULTS:
        self.nBestSize = 1
        self.confusionModelName = 'RandomConfusions'
        self.nBestGeneratorName = 'UniformNBestGenerator'
        self.conf_scorer_name = 'additive'
        self.paramset = 'config/defaultUM.cfg'

        # CONFIG:
        if cfg.has_option('errormodel', 'nbestsize'):
            self.nBestSize = cfg.getint('errormodel','nbestsize')
        if cfg.has_option('errormodel','confusionmodel'):
            self.confusionModelName = cfg.get('errormodel','confusionmodel')
        if cfg.has_option('errormodel','nbestgeneratormodel'):
            self.nBestGeneratorName = cfg.get('errormodel','nbestgeneratormodel')
        if cfg.has_option('errormodel', 'confscorer'):
            self.conf_scorer_name = cfg.get('errormodel', 'confscorer')
        if cfg.has_option('errormodel', 'configfile'):
            self.paramset = cfg.get('errormodel', 'configfile')
        if cfg.has_option('errormodel_' + domain_string, 'configfile'):
            self.paramset = cfg.get('errormodel_' + domain_string, 'configfile')
        if cfg.has_option('errormodel_' + domain_string, 'errorrate'):
            self.error_rate = cfg.getfloat('errormodel_' + domain_string, 'errorrate') / 100
            
        self.paramset = self._check_paramset(self.paramset)

        self.logger.info('N-best list size: ' + str(self.nBestSize))
        self.logger.info('N-best generator model: '+ self.nBestGeneratorName)
        self.logger.info('Confusion model: '+ self.confusionModelName)

        # Create confusion model.
        self._set_confusion_model(domain_string)

        # Create N-best generator.
        self._set_nbest_generator()

        # Create Confidence scorer.
        self._set_confidence_scorer()

    def _set_confusion_model(self, domain_string):
        if self.confusionModelName == 'RandomConfusions':
            self.confusionModel = ConfusionModel.EMRandomConfusionModel(domain_string, self.ontology, self.cfg, self.logger, self.SetObj)
        elif self.confusionModelName == 'LevenshteinConfusions':
            self.confusionModel = ConfusionModel.EMLevenshteinConfusionModel(domain_string, self.ontology, self.cfg, self.logger, self.SetObj)
        else:
            self.logger.error('Confusion model '+self.confusionModelName+' is not implemented.')

    def _set_nbest_generator(self):
        if self.nBestGeneratorName == 'UniformNBestGenerator':
            self.nBestGenerator = NBestGenerator.EMNBestGenerator(self.confusionModel, self.error_rate, self.nBestSize, self.logger, self.SetObj)
        elif self.nBestGeneratorName == 'SampledNBestGenerator':
            self.logger.warning('Note the original C++ implementation of EMSampledNBestGenerator was actually the same to EMUniformNBestGenerator.')
            self.logger.warning('Here the size of N-best list is also sampled from uniform distribution of [1,..,N].')
            self.nBestGenerator = NBestGenerator.EMSampledNBestGenerator(self.confusionModel, self.error_rate, self.nBestSize, self.logger, self.SetObj)
        elif self.nBestGeneratorName == 'DSTC2NBestGenerator':
            self.nBestGenerator = NBestGenerator.DSTC2NBestGenerator(self.confusionModel, self.error_rate, self.nBestSize, self.logger, self.SetObj, self.paramset)
        else:
            self.logger.error('N-best generator '+self.nBestGeneratorName+' is not implemented.')

    def _set_confidence_scorer(self):
        conf_scorer_name = self.conf_scorer_name.lower()
        self.logger.info('Confidence scorer: %s' % conf_scorer_name)
        if conf_scorer_name == 'additive':
            self.confScorer = ConfidenceScorer.AdditiveConfidenceScorer()
        elif conf_scorer_name == 'dstc2':
            self.confScorer = ConfidenceScorer.DSTC2ConfidenceScorer(self.logger, self.SetObj, self.paramset)
        else:
            self.logger.error('Invalid confidence scorer: {}. Using additive scorer.'.format(conf_scorer_name))
        return
    
    def _check_paramset(self, paramset):
        # check if file path points to an existing file. if not, try searching for file relative to root
        if not os.path.isfile(paramset):
            paramset = os.path.join(self.SetObj.root,paramset)
            if not os.path.isfile(paramset):
                self.logger.error('Error model config file "{}" does not exist'.format(paramset))
        return paramset
        
    def confuse_act(self, last_user_act):
        """Clean act in --> Confused act out. 

        :param: (str) simulated users semantic action
        :returns (list) of confused user acts.
        """
        uact = DiaAct.DiaActWithProb(last_user_act)
        n_best = self.nBestGenerator.getNBest(uact)
        n_best = self.confScorer.assignConfScores(n_best)
        
        # Normalise confidence scores
        dSum = 0.0
        for hyp in n_best:
            dSum += hyp.P_Au_O # P_Au_O is the confidence score
        for hyp in n_best:
            hyp.P_Au_O /= dSum
        
        return n_best


#END OF FILE
