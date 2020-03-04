import os
os.system('sbatch -p cpu -n 1 -c 2 -o env1_parallel_log/LOG0 -e env1_parallel_log/ERR env1_parallel_log/run0.sh')
