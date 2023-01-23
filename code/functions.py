import numpy as np
import math
from scipy.spatial import distance

""""""""""""""""""
""" FUNCTIONS  """
""""""""""""""""""

def dist(origin, target):
	return distance.euclidean(origin, target)

def rotate(x, y, theta = math.pi / 2):
	x1 = round(x * math.cos(theta) - y * math.sin(theta), 2)
	y1 = round(x * math.sin(theta) + y * math.cos(theta), 2)
	return x1, y1

def discretize_time(x, t, solve = -1):
	
	x = np.array(x)
	t = np.array(t)
	result = []
	for i in range(int(round(max(t)))):
		idx = np.where((t > (i-1)) & (t <= i))
		if len(idx[0]):

			result.append(np.mean(x[idx]))
		else:
      
			if solve == -1:
				if i > 0:
					result.append(result[-1])
				else:
					result.append(0)
     
			else:
				result.append(0)
		
	return result

def moving_average(x, t, overlap = 0):
	slide = t // 2
	if overlap > slide:
		overlap = slide
		
	x0 = slide
	xn = len(x) - slide - 1
	sq = range(x0, xn, (slide - overlap + 1))
 
	v = [np.mean(x[(i - slide):(i+slide)]) for i in sq]
 
	if overlap == slide:
		return np.concatenate([x[:slide], v, x[xn:len(x)]])
	else:
		return v


	
