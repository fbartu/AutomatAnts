import sys
from functions import argparser
from Model import *

if __name__ == '__main__':
	params = argparser()
	print(params, flush=True)
	m = Model(**params)
	for i in params:
		if hasattr(m, i):
			print(i, getattr(m, i))
		else:
			if hasattr(m.agents[0], i):
				print(i, getattr(m.agents[0], i))
			else:
				print('Model has no attribute', i)
	sys.exit(0)