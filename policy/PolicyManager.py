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
PolicyManager.py - container for all policies
===============================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. seealso:: CUED Imports/Dependencies:

    import :mod:`utils.Settings` |.|
    import :mod:`utils.ContextLogger` |.|
    import :mod:`ontology.Ontology` |.|
    import :mod:`ontology.OntologyUtils`

************************

'''
__author__ = "cued_dialogue_systems_group"

from utils import ContextLogger, Settings
from ontology import Ontology,OntologyUtils
import threading
import time

lock = Settings.load_lock()

class PolicyManager(object):
    '''
    The policy manager manages the policies for all domains.

    It provides the interface to get the next system action based on the current belief state in :func:`act_on` and to initiate the learning in the policy in :func:`train`.
    '''
    def __init__(self, ontology, cfg, logger, SetObj):
        self.cfg = cfg
        self.logger = logger
        self.ontology = ontology
        self.SetObj = SetObj
        self.domainPolicies = dict.fromkeys(ontology.OntologyUtils.available_domains, None)
        self.committees = self._load_committees()
        self.shared_params = None


        self.SPECIAL_DOMAINS = ['topicmanager','wikipedia']

#         self.prevbelief = None
#         self.lastSystemAction = None

        for dstring in self.domainPolicies:
            if self.cfg.has_option("policy_"+dstring,"preload"):
                preload = self.cfg.getboolean("policy_"+dstring,"preload")
                if preload:
                    self.bootup(dstring)

    def savePolicy(self, FORCE_SAVE=False):
        """
        Initiates the policies of all domains to be saved.

        :param FORCE_SAVE: used to force cleaning up of any learning and saving when we are powering off an agent.
        :type FORCE_SAVE: bool
        """
        for dstring in self.domainPolicies.keys():
            if self.domainPolicies[dstring] is not None:
                lock.acquire()
                self.domainPolicies[dstring].savePolicy(FORCE_SAVE)
                lock.release()
        return

    def saveTrainerAndReplay(self):
        for dstring in self.domainPolicies.keys():
            if self.domainPolicies[dstring] is not None:
                lock.acquire()
                self.domainPolicies[dstring].saveTrainerAndReplay()
                lock.release()
        return

    def bootup(self, domainString):
        '''Loads a policy for a given domain.
        '''
        # with BCM if domain was in a committee -- then its policy can have already been loaded. check first:
        if self.cfg.has_option('policycommittee', 'singlemodel') \
                and self.cfg.getboolean('policycommittee', 'singlemodel'):
            if self.shared_params is None:
                self.shared_params = {}
            self._load_domains_policy(domainString)
            self.domainPolicies[domainString].restart()

        elif self.domainPolicies[domainString] is not None:
            self.logger.warning('{} policy is already loaded'.format(domainString))
        else:
            self._load_domains_policy(domainString)
            self.domainPolicies[domainString].restart()
        return

    def act_on(self, dstring, state):
        '''
        Main policy method which maps the provided belief to the next system action. This is called at each turn by :class:`~Agent.DialogueAgent`

        :param dstring: the domain string unique identifier.
        :type dstring: str
        :param state: the belief state the policy should act on
        :type state: :class:`~utils.DialogueState.DialogueState`
        :returns: the next system action as :class:`~utils.DiaAct.DiaAct`
        '''
        if self.domainPolicies[dstring] is None:
            self.bootup(dstring)

        if self.committees[dstring] is not None:
            systemAct = self.committees[dstring].act_on(state=state, domainInControl=dstring)
        else:
            systemAct = self.domainPolicies[dstring].act_on(state=state)

        return systemAct

    def train(self, training_vec = None):
        '''
        Initiates the training for the policies of all domains. This is called at the end of each dialogue by :class:`~Agent.DialogueAgent`
        '''
        # grad = []
        # batch_size = 1
        # for domain in self.domainPolicies:
        #     if self.domainPolicies[domain] is not None and self.domainPolicies[domain].learning:
        #         if training_vec is not None:
        #             if training_vec[domain]:
        #                 lock.acquire()
        #                 grad, batch_size = self.domainPolicies[domain].train()
        #                 lock.release()
        #             else: 
        #                 print("No training due to evaluator decision.")
        #         else:
        #             lock.acquire()
        #             grad, batch_size = self.domainPolicies[domain].train()
        #             lock.release()

        # if grad != []:
        #     Settings.load_grad(curThreadName, grad)
        # curThreadName = threading.currentThread().getName()
        # Settings.global_traincontroller[curThreadName] = True
        # print('******' + curThreadName + ' is waitting ...')
        # start_time = time.time()
        # while True:
        #     if len(Settings.global_traincontroller.items()) == Settings.global_threadsnum and Settings.allTrue():
        #         if grad != []:
        #             grads_list = Settings.grad_sum()
        #             # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        #             # print(grad)
        #             for domain in self.domainPolicies:
        #                 if self.domainPolicies[domain] is not None and self.domainPolicies[domain].learning:
        #                     if training_vec is not None:
        #                         if training_vec[domain]:
        #                             lock.acquire()
        #                             self.domainPolicies[domain].load_and_update(grads_list, Settings.global_threadsnum*batch_size)
        #                             self.domainPolicies[domain].savePolicyInc()
        #                             lock.release()
        #                             print(curThreadName + ' 1training!!!!')
        #                         else:
        #                             self.logger.info("No training due to evaluator decision.")
        #                     else:
        #                         lock.acquire()
        #                         self.domainPolicies[domain].load_and_update(grads_list, Settings.global_threadsnum*batch_size)
        #                         self.domainPolicies[domain].savePolicyInc()
        #                         lock.release()
        #                         print('2training!!!!')
        #         Settings.setFalse()
        #         print(curThreadName + ' waits for ' + str(time.time()-start_time))
        #         break
        #     elif Settings.allFalse():
        #         print(curThreadName + ' waits for ' + str(time.time()-start_time))
        #         break
        #     else:
        #         time.sleep(0.1)
        #         continue

        for domain in self.domainPolicies:
           if self.domainPolicies[domain] is not None and self.domainPolicies[domain].learning:
               if training_vec is not None:
                   if training_vec[domain]:
                       lock.acquire()
                       self.domainPolicies[domain].train()
                       lock.release()
                   else:
                       self.logger.info("No training due to evaluator decision.")
               else:
                   lock.acquire()
                   self.domainPolicies[domain].train()
                   lock.release()

        curThreadName = threading.currentThread().getName()
        Settings.global_traincontroller[curThreadName] = True
        print('******' + curThreadName + ' is waitting ...')
        start_time = time.time()
        while True:
            if len(Settings.global_traincontroller.items()) == Settings.global_threadsnum and Settings.allTrue():
                Settings.setFalse()
                print(curThreadName + ' waits for ' + str(time.time()-start_time))
                break
            elif Settings.allFalse():
                print(curThreadName + ' waits for ' + str(time.time()-start_time))
                break
            else:
                time.sleep(10e-5)
                continue

    def record(self, reward, domainString):
        '''
        Records the current turn reward for the given domain. In case of a committee, the recording is delegated.

        This method is called each turn by the :class:`~Agent.DialogueAgent`.

        :param reward: the turn reward to be recorded
        :type reward: int
        :param domainString: the domain string unique identifier of the domain the reward originates in
        :type domainString: str
        :returns: None
        '''
        if self.committees[domainString] is not None:
            self.committees[domainString].record(reward, domainString)
        else:
            self.domainPolicies[domainString].record(reward)

    def finalizeRecord(self, domainRewards):
        '''
        Records the final rewards of all domains. In case of a committee, the recording is delegated.

        This method is called once at the end of each dialogue by the :class:`~Agent.DialogueAgent`. (One dialogue may contain multiple domains.)

        :param domainRewards: a dictionary mapping from domains to final rewards
        :type domainRewards: dict
        :returns: None
        '''
        for dstring in self.domainPolicies:
            if self.domainPolicies[dstring] is not None:
                domains_reward = domainRewards[dstring]
                if domains_reward is not None:
                    if self.committees[dstring] is not None:
                        self.committees[dstring].finalizeRecord(domains_reward,dstring)
                    elif self.domainPolicies[dstring] is not None:
                        self.domainPolicies[dstring].finalizeRecord(domains_reward,dstring)
                else:
                    self.logger.warning("Final reward in domain: "+dstring+" is None - Should mean domain wasnt used in dialog")

    def getLastSystemAction(self, domainString):
        '''
        Returns the last system action of the specified domain.

        :param domainString: the domain string unique identifier.
        :type domainString: str
        :returns: the last system action of the given domain or None
        '''
        if self.domainPolicies[domainString] is not None:
            return self.domainPolicies[domainString].lastSystemAction
        return None

    def restart(self):
        '''
        Restarts all policies of all domains and resets internal variables.
        '''
#         self.lastSystemAction = None
#         self.prevbelief = None
        i = 0
        for dstring in self.domainPolicies.keys():
            if self.domainPolicies[dstring] is not None:
                i += 1
                # print(threading.currentThread().getName() + ' Domain: ' +dstring + ' Training '+str(i))
                self.domainPolicies[dstring].restart()
        return

    def printEpisodes(self):
        '''
        Prints the recorded episode of the current dialogue.
        '''
        for dString in self.domainPolicies:
            if self.domainPolicies[dString] is not None:
                print "---------- Episodes for domain {}".format(dString)
                for domain in self.domainPolicies[dString].episodes:
                    if self.domainPolicies[dString].episodes[domain] is not None:
                        print domain
                        self.domainPolicies[dString].episodes[domain].tostring()

    def _load_domains_policy(self, domainString=None):
        '''
        Loads and instantiates the respective policy as configured in config file. The new object is added to the internal
        dictionary.

        Default is 'hdc'.

        .. Note:
            To dynamically load a class, the __init__() must take one argument: domainString, learning

        :param domainString: the domain the policy will work on. Default is None.
        :type domainString: str
        :returns: the new policy object
        '''

        # 1. get type:
        policy_type = 'hdc'  # domain+resource independent default
        in_policy_file = ''
        out_policy_file = ''
        learning = False
        useconfreq = False


        if not self.cfg.has_section('policy_'+domainString):
            if not self.cfg.has_section('policy'):
                self.logger.warning("No policy section specified for domain: "+domainString+" - defaulting to HDC")
            else:
                self.logger.info("No policy section specified for domain: " + domainString + " - using values from 'policy' section")
        if self.cfg.has_option('policy', 'policytype'):
            policy_type = self.cfg.get('policy', 'policytype')
        if self.cfg.has_option('policy', 'learning'):
            learning = self.cfg.getboolean('policy', 'learning')
        if self.cfg.has_option('policy', 'useconfreq'):
            useconfreq = self.cfg.getboolean('policy', 'useconfreq')
        if self.cfg.has_option('policy', 'inpolicyfile'):
            in_policy_file = self.cfg.get('policy', 'inpolicyfile')
        if self.cfg.has_option('policy', 'outpolicyfile'):
            out_policy_file = self.cfg.get('policy', 'outpolicyfile')

        if self.cfg.has_option('policy_'+domainString, 'policytype'):
            policy_type = self.cfg.get('policy_'+domainString, 'policytype')
        if self.cfg.has_option('policy_'+domainString, 'learning'):
            learning = self.cfg.getboolean('policy_'+domainString, 'learning')
        if self.cfg.has_option('policy_'+domainString, 'useconfreq'):
            useconfreq = self.cfg.getboolean('policy_'+domainString, 'useconfreq')
        if self.cfg.has_option('policy_'+domainString, 'inpolicyfile'):
            in_policy_file = self.cfg.get('policy_'+domainString, 'inpolicyfile')
        if self.cfg.has_option('policy_'+domainString, 'outpolicyfile'):
            out_policy_file = self.cfg.get('policy_'+domainString, 'outpolicyfile')

        # print('PManager: '+in_policy_file)
        # exit(0)

        if domainString in self.SPECIAL_DOMAINS:
            if domainString == 'topicmanager':
                policy_type = 'hdc_topicmanager'
                from policy import HDCTopicManager
                self.domainPolicies[domainString] = HDCTopicManager.HDCTopicManagerPolicy()
            elif domainString == "wikipedia":
                policy_type = 'hdc_wikipedia'
                import WikipediaTools
                self.domainPolicies[domainString] = WikipediaTools.WikipediaDM()
        else:
            if policy_type == 'hdc':
                from policy import HDCPolicy
                self.domainPolicies[domainString] = HDCPolicy.HDCPolicy(domainString)
            elif policy_type == 'gp':
                from policy import GPPolicy
                self.domainPolicies[domainString] = GPPolicy.GPPolicy(domainString, learning, self.shared_params)

            elif policy_type == 'hack_madqn':
                from policy import HackPolicy
                self.domainPolicies[domainString] = HackPolicy.DQNPolicy(in_policy_file, out_policy_file, self.ontology, self.cfg, self.logger, self.SetObj, domainString, learning)
            elif policy_type == 'hack_bdqn':
                from policy import HackBDQNPolicy
                self.domainPolicies[domainString] = HackBDQNPolicy.BDQNPolicy(in_policy_file, out_policy_file, domainString, learning)
            elif policy_type == 'hack_mabdqn':
                from policy import HackMABDQNPolicy
                self.domainPolicies[domainString] = HackMABDQNPolicy.BDQNPolicy(in_policy_file, out_policy_file, domainString, learning)
            elif policy_type == 'hack_a2c':
                from policy import HackA2CPolicy
                self.domainPolicies[domainString] = HackA2CPolicy.STRACPolicy(in_policy_file, out_policy_file, self.ontology, self.cfg, self.logger, self.SetObj, domainString, learning)
            elif policy_type == 'hack_marainbow':
                from policy import HackRBPolicy
                self.domainPolicies[domainString] = HackRBPolicy.RBDQNPolicy(in_policy_file, out_policy_file, self.ontology, self.cfg, self.logger, self.SetObj, domainString, learning)

            elif policy_type == 'dqn':
                from policy import DQNPolicy
                self.domainPolicies[domainString] = DQNPolicy.DQNPolicy(in_policy_file, out_policy_file, domainString, learning)
            elif policy_type == 'a2c':
                from policy import A2CPolicy
                self.domainPolicies[domainString] = A2CPolicy.A2CPolicy(in_policy_file, out_policy_file, domainString, learning)
            elif policy_type == 'enac':
                from policy import ENACPolicy
                self.domainPolicies[domainString] = ENACPolicy.ENACPolicy(in_policy_file, out_policy_file, domainString, learning)
            elif policy_type == 'bdqn':
                from policy import BDQNPolicy
                self.domainPolicies[domainString] = BDQNPolicy.BDQNPolicy(in_policy_file, out_policy_file, domainString, learning)
            elif policy_type == 'acer':
                from policy import ACERPolicy
                self.domainPolicies[domainString] = ACERPolicy.ACERPolicy(in_policy_file, out_policy_file, domainString, learning)
            elif policy_type == 'tracer':
                from policy import TRACERPolicy
                self.domainPolicies[domainString] = TRACERPolicy.TRACERPolicy(in_policy_file, out_policy_file, domainString, learning)
            else:
                try:
                    # try to view the config string as a complete module path to the class to be instantiated
                    components = policy_type.split('.')
                    packageString = '.'.join(components[:-1])
                    classString = components[-1]
                    mod = __import__(packageString, fromlist=[classString])
                    klass = getattr(mod, classString)
                    self.domainPolicies[domainString] = klass(domainString,learning)
                except ImportError as e:
                    self.logger.error('Invalid policy type "{}" for domain "{}" raising error {}'.format(policy_type, domainString, e))

            Settings.load_hackpolicy(self.domainPolicies[domainString], threading.currentThread().getName())
            #------------------------------
            # TODO - Not currently implemented as we aren't currently using these policy types
#             elif True:
#                 exit('NOT IMPLEMENTED... see msg at this point in code')
#             elif policy_type == 'type':
#                 from policy import TypePolicy
#                 policy = TypePolicy.TypePolicy()
#             elif policy_type == 'select':
#                 from policy import SelectPolicy
#                 policy = SelectPolicy.SelectPolicy(use_confreq=useconfreq)
#             elif policy_type == 'nn':
#                 from policy import NNPolicy
#                 # TODO - further change here - train is now implmented in config file. below needs updating
#                 policy = NNPolicy.NNPolicy(use_confreq=useconfreq, is_training=train)
            #------------------------------
        return

    def _load_committees(self):
        '''
        Loads and instantiates the committee as configured in config file. The new object is added to the internal
        dictionary.
        '''
        committees = dict.fromkeys(self.ontology.OntologyUtils.available_domains, None)
        useBCM = False
        learningMethod = "singleagent"

        if self.cfg.has_option("policycommittee","bcm"):
            useBCM = self.cfg.getboolean("policycommittee","bcm")

        if not useBCM:
            return committees # return an empty committee dict to indicate that committees are not used

        from policy import PolicyCommittee
        if self.cfg.has_option("policycommittee","learningmethod"):
            learningMethod = self.cfg.get("policycommittee","learningmethod")

        if self.cfg.has_option("policycommittee","pctype"):
            pcType =  self.cfg.get("policycommittee","pctype")
        if pcType == 'hdc':
            # handcrafted committees are a bit strange, I think they should be removed
            predefinedCommittees = {}
            predefinedCommittees['Electronics'] = ['Laptops6','Laptops11','TV']
            predefinedCommittees['SF'] = ['SFHotels','SFRestaurants']

            for key in predefinedCommittees:
                committee = PolicyCommittee.PolicyCommittee(self,predefinedCommittees[key],learningMethod, self.SetObj)
                self._check_committee(committee)

                for domain in predefinedCommittees[key]:
                    committees[domain] = committee


        elif pcType == 'configset':
            # TODO extend settings to allow multiple committees
            try:
                committeeMembers = self.cfg.get('policycommittee', 'configsetcommittee')
            except Exception as e: #ConfigParser.NoOptionError:  # can import ConfigParser if you wish
                print e
                self.logger.error('When using the configset committee - you need to set configsetcommittee in the config file.')

            committeeMembers = committeeMembers.split(',')
            committee = PolicyCommittee.PolicyCommittee(self,committeeMembers,learningMethod)
            self._check_committee(committee)

            for domain in committeeMembers:
                committees[domain] = committee

        else:
            self.logger.error("Unknown policy committee type %s" % pcType)

        return committees

    def _check_committee(self,committee):
        '''
        Safety tool - should check some logical requirements on the list of domains given by the config

        :param committee: the committee be be checked
        :type committee: :class:`~policy.PolicyCommittee.PolicyCommittee`
        '''
        committeeMembers = committee.members

        if len(committeeMembers) < 2:
            self.logger.warning('Only 1 domain given')

        # Ensure required objects are available to committee (ontology for domains etc)
        for dstring in committeeMembers:
            self.ontology.ensure_domain_ontology_loaded(dstring)


        # TODO - should check that domain tags given are valid according to OntologyUtils.py
        return

