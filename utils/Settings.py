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
Settings.py - global variables: config, random num generator 
=============================================================

Copyright CUED Dialogue Systems Group 2015 - 2017

Creates and makes accessible system wide a config parser and random number generator.
Also contains hardcodings of the paths to the ontologies.

[GENERAL]
    root = ''    - which is the path to root directory of python system. Use when running grid jobs.

Globals::
  
    config:       python ConfigParser.ConfigParser() object
    random:       numpy.random.RandomState() random number generator
    root:         location of root directory of repository - for running on the grid

.. seealso:: CUED Imports/Dependencies: 

    none

************************

'''

__author__ = "cued_dialogue_systems_group"
__version__ = '1.0'         # version setting for all modules in repository root only.

import ConfigParser
import numpy.random as nprandom
import os.path
import threading
import mxnet.ndarray as nd


#==============================================================================================================
# global variants and mrthods
#==============================================================================================================
global_currentturn = None
global_currentcount = 0
global_traincontroller = {}
global_gradsaver = {}
global_grads_weight = [0.2, 0.3, 0.5]
global_policysaver = {}
global_hackpolicysaver = {}
global_threadsnum = 3
lock = threading.RLock()

def load_policy(item, key):
    global global_policysaver
    global_policysaver[key] = item

def load_hackpolicy(item, key):
    global global_hackpolicysaver
    global_hackpolicysaver[key] = item

def load_grad(item, key):
    global global_gradsaver
    global_gradsaver[key] = item

def grad_sum():
    global global_gradsaver
    grads_list = []
    key_list = []
    length = 0
    for k, v in global_gradsaver.items():
        key_list.append(k)
        length = len(v)

    for i in range(length):
        grad = global_gradsaver[key_list[0]][i]
        for j in range(1,len(key_list)):
            grad = grad + global_gradsaver[key_list[j]][i]
        grads_list.append(grad)

    return grads_list

def add_count():
    global global_currentcount
    global_currentcount += 1
    return global_currentcount

def get_count():
    return global_currentcount

def load_lock():
    global lock
    return lock

def allTrue():
    global global_traincontroller
    for k, v in global_traincontroller.items():
        if not v:
            return False
    return True

def allFalse():
    global global_traincontroller
    for k, v in global_traincontroller.items():
        if v:
            return False
    return True

def setFalse():
    global global_traincontroller
    for k, v in global_traincontroller.items():
        global_traincontroller[k] = False

#==============================================================================================================
# class Settings
#==============================================================================================================

class Settings(object):

    def __init__(self, config_file, seed):
        self.config = None
        self.random = None
        self.seed = seed
        self.randomCount = 0
        self.root = ''
        self.load_config(config_file)
        self.load_root()

        # Seed:
        #-----------------------------------------
        if self.seed is None:
            # no seed given at cmd line (the overriding input), so check config for a seed, else use None (which means use clock).    
            if self.config.has_option("GENERAL",'seed'):
                self.seed = self.config.getint("GENERAL","seed")
        self.seed = self.set_seed()


    def load_config(self, config_file):
        '''
        Loads the passed config file into a python ConfigParser().
           
        :param config_file: path to config
        :type config_file: str
        '''
        if config_file is not None:
            try:
                self.config = ConfigParser.ConfigParser()
                self.config.read(config_file)
            except Exception as inst:
                print 'Failed to parse file', inst
        else:
            # load empty config
            self.config = ConfigParser.ConfigParser()

    def load_root(self, rootIn=None):
        '''
        Root is the location (ie full path) of the cued-python repository. This is used when running things on the grid (non local
        machines in general).
        '''
        if self.config is not None:     
            if self.config.has_option("GENERAL",'root'):
                self.root = self.config.get("GENERAL",'root')
        if rootIn is not None:  # just used when called by SemI parser without a config file
            self.root = rootIn

    def set_seed(self):
        '''
        Intialise np random num generator

        :param seed: None
        :type seed: int
        '''
        if self.seed is None:
            random1 = nprandom.RandomState(None)
            self.seed = random1.randint(1000000000)
        self.random = nprandom.RandomState(self.seed)

        return self.seed
        
    def randomProfiling(self, t = None):
        self.randomCount += 1
        print "Random access: {} from class {}".format(randomCount,t)
        return self.random

    def locate_file(self, filename):
        '''
        Locate file either as given or relative to root

        :param filename: file to check
        :return: filename possibly prepended with root
        '''
        if os.path.exists(filename):
            return filename         # file exists as given
        else:
            return self.root+filename    # note this may or may not exist

# END OF FILE
