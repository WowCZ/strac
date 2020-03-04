from __future__ import print_function

import os
import thread
import time
import sys

assert len(sys.argv) == 4

finished = 0

# /mnt/lustre/sjtu/users/zc825/workspace/github/ParallelPyDial/parallelConfig/parallel

def submit(seed):
    arglist = sys.argv[1].split('-')[0]
    os.system('mkdir ' + arglist + '_parallel_log')
    runfile = open(arglist + '_parallel_log/run' + str(seed) + '.sh', 'w')
    runfile.write('#!/bin/sh\n')
    runfile.write('/mnt/lustre/sjtu/home/zc825/.local/bin/python pydial.py parallelTrain '
                  'parallelConfig/a2c/' + sys.argv[1] + '.cfg ' + 'parallelConfig/a2c/' + sys.argv[2] + '.cfg ' +'parallelConfig/a2c/'+ sys.argv[3] +'.cfg --seed=' + str(seed) + '\n')
    runfile.flush()

    os.system('sbatch -p cpu -n 1 -c 2 -o ' + arglist + '_parallel_log/LOG' + str(seed) + ' -e '+ arglist + '_parallel_log/ERR ' + arglist + '_parallel_log/run' + str(seed) + '.sh')
    print('sbatch -p cpu -n 1 -c 2 -o ' + arglist + '_parallel_log/LOG' + str(seed) + ' -e '+ arglist + '_parallel_log/ERR ' + arglist + '_parallel_log/run' + str(seed) + '.sh')


try:
    for seed in range(0, 10):
        thread.start_new_thread(submit, (seed, ))
except:
    print('Error!')

while True:
    pass
