import subprocess

if __name__=='__main__':
	num_chunks = [2**i for i in xrange(16)]
	num_trials = 5
	fd = open('sweep_profile_estimate_normals.txt','w')
	for nc in num_chunks:
		for i in xrange(num_trials):
			print '(%d,%d)'%(nc,i)
			P = subprocess.Popen(['python','profile_estimate_normals.py',str(nc)],\
				stdout=subprocess.PIPE)
			output = P.communicate()[0]
			fd.write(output)
	fd.close()
