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
OntologyUtils.py - paths and rules for ontology
===========================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

.. warning:: 

    content partly hard-coded (paths, dicts, etc.)

.. seealso:: CUED Imports/Dependencies: 
    
    import :mod:`self.SetObj` |.|
    import :mod:`utils.ContextLogger` |.|
    

************************

'''
__author__ = "cued_dialogue_systems_group"
from utils import ContextLogger
import os
logger = ContextLogger.getLogger('')


# self.ont_db_pairs = {
#                     'Laptops6':{'ontology':'ontology/ontologies/Laptops6-rules.json',
#                             'database':'ontology/ontologies/Laptops6-dbase.'},
#                     'Laptops11':{'ontology':'ontology/ontologies/Laptops11-rules.json',
#                              'database':'ontology/ontologies/Laptops11-dbase.'},
#                     'TV':{'ontology':'ontology/ontologies/TV-rules.json',
#                                'database':'ontology/ontologies/TV-dbase.'},
#                     'TSBextHD':{'ontology':'ontology/ontologies/TSBextHD-rules.json',
#                                 'database':'ontology/ontologies/TSBextHD-dbase.'},
#                     'TSBplayer':{'ontology':'ontology/ontologies/TSBplayer-rules.json',
#                                  'database':'ontology/ontologies/TSBplayer-dbase.'},
#                     'SFRestaurants':{'ontology':'ontology/ontologies/SFRestaurants-rules.json',
#                            'database':'ontology/ontologies/SFRestaurants-dbase.'},
#                     'SFHotels':{'ontology':'ontology/ontologies/SFHotels-rules.json',
#                            'database':'ontology/ontologies/SFHotels-dbase.'},
#                     'CamRestaurants':{'ontology':'ontology/ontologies/CamRestaurants-rules.json',
#                           'database':'ontology/ontologies/CamRestaurants-dbase.'},
#                     'CamShops':{'ontology':'ontology/ontologies/CamShops-rules.json',
#                                'database':'ontology/ontologies/CamShops-dbase.'},
#                     'CamAttractions':{'ontology':'ontology/ontologies/CamAttractions-rules.json',
#                                  'database':'ontology/ontologies/CamAttractions-dbase.'},
#                     'CamTransport':{'ontology':'ontology/ontologies/CamTransport-rules.json',
#                                 'database':'ontology/ontologies/CamTransport-dbase.'},
#                     'CamHotels':{'ontology':'ontology/ontologies/CamHotels-rules.json',
#                                  'database':'ontology/ontologies/CamHotels-dbase.'},
#             }

class OntologyUtils(object):

    def __init__(self, SetObj):

        self.ont_db_pairs = {}
        self.available_domains = []
        self.ALLOWED_CODOMAIN_RULES = {}
        self.MULTIDOMAIN_GROUPS = {}
        self.BINARY_SLOTS = {}
        self.SetObj = SetObj

    def initUtils(self):
        self.ont_db_pairs.clear()
        del self.available_domains[:]
        self.ALLOWED_CODOMAIN_RULES.clear()
        self.MULTIDOMAIN_GROUPS.clear()
        self.BINARY_SLOTS.clear()
        
        
        ontopath = os.path.join('ontology','ontologies')
        onlyfiles = [f for f in os.listdir(os.path.join(self.SetObj.root,ontopath)) if os.path.isfile(os.path.join(os.path.join(self.SetObj.root,ontopath), f))]
        domains = set([s.split('-')[0] for s in onlyfiles])
        for domain in domains:
            if domain not in self.ont_db_pairs:
                dbpath = os.path.join(ontopath,domain + '-dbase.db')
                rulespath = os.path.join(ontopath,domain + '-rules.json')
                if os.path.exists(os.path.join(self.SetObj.root,dbpath)) and os.path.exists(os.path.join(self.SetObj.root,rulespath)):
            #                 'Laptops6':{'ontology':'ontology/ontologies/Laptops6-rules.json',
            #                             'database':'ontology/ontologies/Laptops6-dbase.'},
                    self.ont_db_pairs[domain] = {'ontology':rulespath,
                                'database':os.path.join(ontopath,domain + '-dbase.')}




        self.available_domains.extend(self.ont_db_pairs.keys())
    #     self.available_domains = self.ont_db_pairs.keys()
        self.available_domains.append('topicmanager')   #add the topicmanager key to available domains
        self.available_domains.append('wikipedia')   #add the wikipedia key to available domains
        self.available_domains.append('ood')   #add the ood key to available domains


        # TODO - fix this
        # For Multi-domain dialog - determining which of the allowed (config specified) domains can be paired together:
        # Dont want to have a combinatorial explosion here - so make it linear and set allowed partners for each domain:
        # -- NB: these group restrictions only apply to simulate
        # TODO - just specifying GROUPS may be a simpler approach here ... 
        #-------- Hand Coded codomain rules:
        self.ALLOWED_CODOMAIN_RULES.update(dict.fromkeys(self.ont_db_pairs.keys()))
        self.ALLOWED_CODOMAIN_RULES["Laptops6"] = ["TV"]
        self.ALLOWED_CODOMAIN_RULES["Laptops11"] = ["TV"]
        self.ALLOWED_CODOMAIN_RULES["TV"] = [["Laptops6"], ["Laptops11"]]
        
        self.ALLOWED_CODOMAIN_RULES["SFRestaurants"] = ["SFHotels"]
        self.ALLOWED_CODOMAIN_RULES["SFHotels"] = ["SFRestaurants"]
        self.ALLOWED_CODOMAIN_RULES["CamRestaurants"] = ["CamHotels","CamShops", "CamAttractions","CamTransport"]
        self.ALLOWED_CODOMAIN_RULES["CamTransport"] = ["CamHotels","CamShops", "CamAttractions","CamRestaurants"]
        self.ALLOWED_CODOMAIN_RULES["CamAttractions"] = ["CamHotels","CamShops", "CamRestaurants","CamTransport"]
        self.ALLOWED_CODOMAIN_RULES["CamShops"] = ["CamHotels","CamRestaurants", "CamAttractions","CamTransport"]
        self.ALLOWED_CODOMAIN_RULES["CamHotels"] = ["CamRestaurants","CamShops", "CamAttractions","CamTransport"]
        
    #     self.ALLOWED_CODOMAIN_RULES["Laptops6"] = ["TV","TSBextHD","TSBplayer"]
    #     self.ALLOWED_CODOMAIN_RULES["Laptops11"] = ["TV","TSBextHD","TSBplayer"]
    #     self.ALLOWED_CODOMAIN_RULES["TV"] = [["Laptops6","TSBextHD","TSBplayer"], ["Laptops11","TSBextHD","TSBplayer"]]
    #     self.ALLOWED_CODOMAIN_RULES["TSBextHD"] = [["TV","Laptops11","TSBplayer"], ["Laptops6","TV","TSBplayer"]]
    #     self.ALLOWED_CODOMAIN_RULES["TSBplayer"] = [["TV","TSBextHD","Laptops6"],["TV","TSBextHD","Laptops11"]]
    #-----------------------------------------


        self.MULTIDOMAIN_GROUPS.update(dict.fromkeys(['camtourist','sftourist','electronics','all']))
        self.MULTIDOMAIN_GROUPS['camtourist'] = ["CamHotels","CamShops", "CamAttractions","CamRestaurants", "CamTransport"]
        self.MULTIDOMAIN_GROUPS['sftourist'] = ['SFRestaurants','SFHotels']
        self.MULTIDOMAIN_GROUPS['electronics'] = ["TV","Laptops11"]          # Laptops6 or Laptops11 here
    #     self.MULTIDOMAIN_GROUPS['electronics'] = ["TV","Laptops11","TSBplayer","TSBextHD"]          # Laptops6 or Laptops11 here
        self.MULTIDOMAIN_GROUPS['all'] = list(self.available_domains)     # list copies - needed since we remove some elements 
        self.MULTIDOMAIN_GROUPS['all'].remove('topicmanager')        # remove special domains
        self.MULTIDOMAIN_GROUPS['all'].remove('wikipedia')
        self.MULTIDOMAIN_GROUPS['all'].remove('ood')

        # TODO - fix this
        #TODO add these for each domain  - or write something better like a tool to determine this from ontology
        # Note that ALL ONTOLOGIES should be representing binary values as 0,1  (Not true,false for example)
        # These are used by SEMI to check whether we can process a yes/no response as e.g. an implicit inform(true)
        
        self.BINARY_SLOTS.update(dict.fromkeys(self.ont_db_pairs.keys()))
        self.BINARY_SLOTS['CamHotels'] = ['hasparking']
        self.BINARY_SLOTS['SFHotels'] = ['dogsallowed','hasinternet','acceptscreditcards']
        self.BINARY_SLOTS['SFRestaurants'] = ['allowedforkids']
        self.BINARY_SLOTS['Laptops6'] = ['isforbusinesscomputing']
        self.BINARY_SLOTS['Laptops11'] = ['isforbusinesscomputing']
        self.BINARY_SLOTS['TV'] = ['usb']
        self.BINARY_SLOTS['CamRestaurants'] = []


    #==============================================================================================================
    # Methods 
    #==============================================================================================================
    def get_ontology_path(self, domainString):
        '''Required function just to handle repository root location varying if running on grid machines etc
        :rtype: object
        '''
        return os.path.join(self.SetObj.root, self.ont_db_pairs[domainString]['ontology'])

    def get_database_path(self, domainString):
        '''Required function just to handle repository root location varying if running on grid machines etc
        '''
        return os.path.join(self.SetObj.root, self.ont_db_pairs[domainString]['database'])

    def get_domains_group(self, domains):
        '''domains has (needs to have) been checked to be in ['camtourist','sftourist','electronics','all']:
        '''
        if domains == 'camtourist':
            return self.MULTIDOMAIN_GROUPS['camtourist']
        elif domains == 'sftourist':
            return self.MULTIDOMAIN_GROUPS['sftourist']
        elif domains == 'electronics':
            return self.MULTIDOMAIN_GROUPS['electronics']
        elif domains == 'all':
            return self.MULTIDOMAIN_GROUPS['all']
        else:
            logger.error('Invalid domain group: ' + domains) 


    #END OF FILE