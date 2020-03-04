import os

def logSparate(infile, outfile=None):
	if not os.path.exists(infile):
		os.mkdir(thread1)

	infile_list = infile.split('/')
	dir_str = '/'.join(infile[:-1])
	filename = infile_list[-1]
	filename_list = filename.split('-')
	filename_list[2] = 'CR-parallel'
	CR_fn = '-'.join(filename_list)
	filename_list[2] = 'SFR-parallel'
	SFR_fn = '-'.join(filename_list)
	filename_list[2] = 'LAP-parallel'
	LAP_fn = '-'.join(filename_list)


	CR = os.path.join(outfile, CR_fn)
	wt1 = open(CR,'w')
	SFR = os.path.join(outfile, SFR_fn)
	wt2 = open(SFR,'w')
	LAP = os.path.join(outfile, LAP_fn)
	wt3 = open(LAP,'w')

	with open(infile, 'r') as inf:
		for line in inf.readlines():
			# print(line.strip())
			if 'Thread-2' in line:
				wt1.write(line)
			elif 'Thread-3' in line:
				wt2.write(line)
			elif 'Thread-4' in line:
				wt3.write(line)
			else:
				print('Bug: ')
				print(line)


infile = '/home/slhome/zc825/wowcz_github/Pydial/czresult_final/final/_env{}_UNIVERSAL_madqn_logs/env{}-madqn-CR-seed{}-30.1-20.train.log'
outfile = '/home/slhome/zc825/wowcz_github/Pydial/czresult_final/final/_env{}_UNIVERSAL_madqn_logs'

for i in range(6,7):
	for j in range(10):
		logSparate(infile.format(str(i), str(i), str(j)), outfile.format(str(i)))