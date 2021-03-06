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
SimulatedUsersManager.py - combines simulated components into functional simulator 
==================================================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies: 

    import :mod:`usersimulator.UserModel` |.|
    import :mod:`usersimulator.ErrorSimulator` |.|
    import :mod:`ontology.Ontology` |.|
    import :mod:`ontology.OntologyUtils` |.|
    import :mod:`utils.self.SetObj` |.|
    import :mod:`utils.DiaAct` |.|
    import :mod:`utils.ContextLogger`

************************

'''


__author__ = "cued_dialogue_systems_group"

import UserModel
import ErrorModel
from utils import DiaAct, ContextLogger
from ontology import Ontology
# logger = ContextLogger.getLogger('')


class DomainsSimulatedUser(object):
    '''User Simulator for a single domain. Comprised of a behaviour component: :class:`UserModel` to produce a semantic act
    and an error simulator to produce from the semantic act a list of semantic hypotheses.

    :param str: domain string
    '''
    def __init__(self, domainString, error_rate, ontology, cfg, logger, SetObj):
        '''
        '''
        self.SetObj = SetObj
        self.um = UserModel.UM(domainString, ontology, cfg, logger, SetObj)
        self.error_simulator = ErrorModel.DomainsErrorSimulator(domainString, error_rate, ontology, cfg, logger, SetObj)
        
        self.randomLearning = False
        if cfg.has_option("mogp_"+domainString, "randomweightlearning"):
            self.randomLearning = cfg.getboolean("mogp_"+domainString, "randomweightlearning")
            
        if self.randomLearning:
            import policy.morl.WeightGenerator as wg
            self.weightGen = wg.WeightGenerator(domainString)
 
    def restart(self, otherDomainsConstraints):
        '''Resets all components (**User Model**) that are statefull.

        :param otherDomainsConstraints: of domain goal tuples (slot=val)
        :type otherDomainsConstraints: list
        :returns: None
        '''
        self.um.init(otherDomainsConstraints) 
        
        if self.randomLearning:
            self.weightGen.updateWeights()
        
    def act_on(self, sys_act_string):
        '''Thru the UserModel member, receives the system action and then responds.

        :param sys_act_string: system action
        :type sys_act_string: unicode str
        :returns: (str) user action
        '''
        sys_act = DiaAct.DiaAct(sys_act_string)
        self.um.receive(sys_act)
        user_act = self.um.respond()
        return user_act


class SimulatedUsersManager(object):
    """
    The multidomain simulated user, which is made up of a dictionary of simulated users indexed by domain. 
    :param (list): of domain strings
    """
    def __init__(self, error_rate, ontology, cfg, logger, SetObj):
        self.possible_domains = ontology.possible_domains
        self.ontology = ontology
        self.logger = logger
        self.SetObj = SetObj
        if cfg.has_option("GENERAL", "testdomains"):
            self.possible_domains = cfg.get("GENERAL", "testdomains").split(',')
        if cfg.has_option("GENERAL", "traindomains"):
            self.possible_domains = cfg.get("GENERAL", "traindomains").split(',')
        self.error_rate = error_rate
        logger.info('Simulating with error rate: ' + str(error_rate))
        self.simUserManagers = dict.fromkeys(self.ontology.OntologyUtils.available_domains, None)
        
        
        # DEFAULTS:
        self.MIN_DOMAINS_PER_DIALOG = 1
        self.MAX_DOMAINS_PER_DIALOG = 3
        self.INCLUDE_DOMAIN_PROB = 0.6
        self.CONDITIONAL_BEHAVIOUR = False
        self.forceNullPositive = False
        self.traceDialog = 2
        self.temp_domains = []
        
        # CONFIG OPTIONS: 
        if cfg.has_option("simulate", "forcenullpositive"):
            self.forceNullPositive = cfg.getboolean("simulate", "forcenullpositive")
        if cfg.has_option("simulate","includedomainprob"):
            self.INCLUDE_DOMAIN_PROB = cfg.getfloat("simulate","includedomainprob")
            assert(self.INCLUDE_DOMAIN_PROB <= 1.0 and self.INCLUDE_DOMAIN_PROB > 0)
        if cfg.has_option("simulate","maxdomainsperdialog"):
            self.MAX_DOMAINS_PER_DIALOG = cfg.getint("simulate","maxdomainsperdialog")
        if cfg.has_option("simulate","mindomainsperdialog"):
            self.MIN_DOMAINS_PER_DIALOG = cfg.getint("simulate","mindomainsperdialog")
            assert(self.MIN_DOMAINS_PER_DIALOG <= self.MAX_DOMAINS_PER_DIALOG)
            assert(self.MIN_DOMAINS_PER_DIALOG <= len(ontology.possible_domains))
        if cfg.has_option("conditional","conditionalsimuser"):
            self.CONDITIONAL_BEHAVIOUR = cfg.getboolean("conditional","conditionalsimuser")
        if cfg.has_option("GENERAL", "tracedialog"):
            self.traceDialog = cfg.getint("GENERAL", "tracedialog")
        if cfg.has_option("simulate", "domainsampling"):
            self.domainSampling = cfg.get("simulate", "domainsampling")
        else:
            self.domainSampling = "random"

        self.cfg = cfg



    def set_allowed_codomains(self, ROOTDOMAIN):
        """
        Sets member (list) *allowed_codomains* given a root domain name (ie the domain of the first constraint)
        Uses the hardcoded rules in self.SetObj.py to do so. Also, based on determined allowed_codomains, sets
        the probability of each being included, independently.

        :param ROOTDOMAIN: domain tag
        :type ROOTDOMAIN: str
        :returns: None
        """
        if self.CONDITIONAL_BEHAVIOUR and ROOTDOMAIN in self.ontology.OntologyUtils.ALLOWED_CODOMAIN_RULES:
            #NB: These next few lines depend on hard coding of OntologyUtils.ALLOWED_CODOMAIN_RULES
            self.allowed_codomains = self.ontology.OntologyUtils.ALLOWED_CODOMAIN_RULES[ROOTDOMAIN]
            
            if self.allowed_codomains is not None:
                if len(self.allowed_codomains) > 1:
                    if isinstance(self.allowed_codomains[1],list):
                        randindex = self.SetObj.random.randint(0,len(self.ontology.OntologyUtils.ALLOWED_CODOMAIN_RULES[ROOTDOMAIN]))
                        self.allowed_codomains = self.ontology.OntologyUtils.ALLOWED_CODOMAIN_RULES[ROOTDOMAIN][randindex]
            else:
                self.allowed_codomains = []
                    
            # based on the allowed_codomains, set the prob of each one independently being in the dialog:
            #self.INCLUDE_DOMAIN_PROB = min(0.3,1.0/len(self.allowed_codomains))
        else:
            self.allowed_codomains = list(self.possible_domains)
            self.allowed_codomains.remove(ROOTDOMAIN)
        return


    def sample_domains(self):
        """Randomly select a set of domains from those available. 
            The selected domains will be used by the simulated user over a single dialog.

        :param None:
        :returns: None
        """

        if self.domainSampling == "random":
            # sample from possible_domains
            self.using_domains = []
            self.using_domains.append(self.SetObj.random.choice(self.possible_domains))  # must have at least 1 element
            root_domain = self.using_domains[0] # the first chosen domain - will affect which codomains can be partnered with
            self.set_allowed_codomains(ROOTDOMAIN=root_domain)
            shuffled_possible_domains = list(self.possible_domains)
            self.SetObj.random.shuffle(shuffled_possible_domains)

            for dstring in shuffled_possible_domains:
                if len(self.using_domains) == self.MAX_DOMAINS_PER_DIALOG:
                    break
                if dstring not in self.allowed_codomains:
                    continue
                if dstring in self.using_domains:
                    continue
                if len(self.using_domains) < self.MIN_DOMAINS_PER_DIALOG:
                    self.using_domains.append(dstring)
                elif self.SetObj.random.rand() < self.INCLUDE_DOMAIN_PROB:
                    self.using_domains.append(dstring)
            # Note - we may not have met the Min domains limit at this point - for example the allowed codomains for the root_domain
            # may have been too low.
            if len(self.using_domains) < self.MIN_DOMAINS_PER_DIALOG:
                self.logger.warning("Found {} domains only, which is less than the config set minimum domains per dialog of {}".format(\
                                                                        len(self.using_domains), self.MIN_DOMAINS_PER_DIALOG))

            self.SetObj.random.shuffle(self.using_domains) # list order is persistent. Simulated user will act in this order now.
            self.logger.info('Order sim user will execute goals:'+str(self.using_domains))

        elif self.domainSampling == "roundrobin":
            if self.MIN_DOMAINS_PER_DIALOG > 1:
                self.logger.warning('With "roundrobin" domains ampling "mindomainsperdialog" cannot be larger than 1, setting it to 1')
                self.MIN_DOMAINS_PER_DIALOG = 1
            if self.MAX_DOMAINS_PER_DIALOG > 1:
                self.logger.warning('With "roundrobin" domain sampling "maxdomainsperdialog" cannot be larger than 1, setting it to 1')
                self.MAX_DOMAINS_PER_DIALOG = 1
            if self.temp_domains == []:
                self.temp_domains = list(self.possible_domains)
                self.SetObj.random.shuffle(self.temp_domains)
            self.using_domains=[self.temp_domains.pop()]

        self.number_domains_this_dialog = len(self.using_domains)
        self.uncompleted_domains = list(self.using_domains)
        return

    def restart(self):
        """Restarts/boots up the selected domains simulated user components. Shuts down those running and not needed for 
            the next dialog.

        :param None:
        :returns: None
        """ 
        # sample domain for this dialog and randomise order:
        self.sample_domains()

        # reset domain simulators:
        otherDomainsConstraints = []  # used to conditionally generate domain goals

        for dstring in self.using_domains: # doing this way to generate goals/behaviour in an order.
            # fire up or check if it is running
            if self.simUserManagers[dstring] is None:
                self.simUserManagers[dstring] = DomainsSimulatedUser(dstring, self.error_rate, self.ontology, self.cfg, self.logger, self.SetObj)
            self.simUserManagers[dstring].restart(otherDomainsConstraints)
            
            # DEBUG prints to inspect goals we have generated:
            self.logger.debug(str(self.simUserManagers[dstring].um.goal))
            self.logger.debug(str(self.simUserManagers[dstring].um.goal.copied_constraints))
            self.logger.debug(str(self.simUserManagers[dstring].um.hdcSim.agenda.agenda_items))
            self.logger.debug("DOMAIN-----"+dstring)
            #raw_input('goal and agenda for domain '+dstring)
            
            
            if self.CONDITIONAL_BEHAVIOUR:
                otherDomainsConstraints += self.simUserManagers[dstring].um.goal.constraints
             

        for dstring in self.possible_domains:  #STATELESS, no memory here. Fine to actually `kill' domains not using 
            if dstring not in self.using_domains:
                self.simUserManagers[dstring] = None 
        return 

    def act_on(self, sys_act):
        """ First produce a single semantic act from the simulated user. Then create from it a list of semantic hypotheses which
        include simulated errors. 
        """
        user_act, user_actsDomain = self._user_act(sys_act)
        hyps = self._confuse_user_act_and_enforce_null(user_act, user_actsDomain)
        return user_act, user_actsDomain, hyps
    
    def _user_act(self, sys_act):
        '''Produces the next user semantic act from the simulated user. Also returns the domain that the act came from 
        --> which avoids doing topictracking during simulate
        '''
        # TODO - this is just a start. lots needs thinking about here.
        # -- needs to return the current simulation domain explictly for now
        # return dstring too -  use this information for error simulation. 
        self.logger.debug('simulated users uncompleted domains:'+str(self.uncompleted_domains))
        for dstring in self.using_domains:
            if dstring in self.uncompleted_domains:
                user_act = self.simUserManagers[dstring].act_on(sys_act)
                if 'bye(' in user_act.to_string():
                    sys_act = 'hello()'
                    self.uncompleted_domains.remove(dstring)
                    if len(self.uncompleted_domains):
                        continue
                    else:
                        break
                else:
                    break        
        return  user_act, dstring
    
    def _confuse_user_act_and_enforce_null(self, user_act, user_actsDomain):
        '''Simulate errors in the semantic parses. Returns a set of confused hypotheses.
        Also enforces a null() act if config set to do so. 
        '''
        # Confused user act.
        hyps = self.simUserManagers[user_actsDomain].error_simulator.confuse_act(user_act) 
        null_prob = 0.0
        for h in hyps:
            act = h.to_string()
            prob = h.P_Au_O
            if act == 'null()':
                null_prob += prob
            if self.traceDialog>1:
                print '   Semi >', act, '[%.6f]' % prob
            self.logger.info('| Semi > %s [%.6f]' % (act,prob))
        if self.forceNullPositive and null_prob < 0.001:
            nullAct = DiaAct.DiaActWithProb('null()')
            nullAct.P_Au_O = 0.001
            hyps.append(nullAct)
            if self.traceDialog>1:
                print '| Semi > null() [0.001]'
            self.logger.info('   Semi > null() [0.001]')
        return hyps
    
# END OF FILE
