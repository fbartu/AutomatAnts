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

def moving_average(x, t, overlap = 0, tvec = None):
	
	slide = t // 2
	if overlap > slide:
		overlap = slide
	
	if tvec is None:
			
		x0 = slide
		xn = len(x) - slide - 1
		sq = range(x0, xn, (slide - overlap + 1))
	
		v = [np.mean(x[(i - slide):(i+slide)]) for i in sq]
	
		if overlap == slide:
			return np.concatenate([x[:slide], v, x[xn:len(x)]])
		else:
			return v

	else:
		if len(tvec) != len(x):
			raise ValueError('tvec must have the same length as x')
		
		x = np.array(x)
		tvec = np.array(tvec)

		v = []
		for i in range(np.min(tvec), np.max(tvec), (slide - overlap + 1)):
			idx = np.where((tvec > (i - slide)) & (tvec <= (i + slide)))
			if len(idx[0]):
				v.append(np.mean(x[idx]))
			else:
				v.append(0)
	
		return v

'''
REPASSAR  !!!!!!!!!!!!!
'''

def direction(x):
	x = np.round(np.array(x), 5)
	if np.all(x[2] == x[0]):
		last_move = 0
  
	elif np.logical_and(x[2][0] < x[0][0], x[2][1] > x[0][1]):
     
		if x[2][1] > x[1][1]:
			last_move = 1
		else:
			last_move = -1
	elif np.logical_and(x[2][0] > x[0][0], x[2][1] < x[0][1]):
		if x[2][1] == x[1][1]:
			last_move = -1
		else:
			last_move = 1
	elif np.logical_and(x[2][0] < x[0][0], x[2][1] < x[0][1]):
		if x[2][1] < x[1][1]:
			last_move = -1
		else:
			last_move = 1
	elif np.logical_and(x[2][0] > x[0][0], x[2][1] > x[0][1]):
		if x[2][1] == x[1][1]:
			last_move = 1
		else:
			last_move = -1
  
	else:
		if np.logical_and(x[2][0] == x[0][0], x[2][1] > x[0][1]):
			if x[1][0] > x[2][0]:
				last_move = -1
			else:
				last_move = 1
		elif np.logical_and(x[2][0] == x[0][0], x[2][1] < x[0][1]):
			if x[1][0] > x[2][0]:
				last_move = 1
			else:
				last_move = -1
		else:
			print('Unexpected scenario')
			last_move = np.nan
   
	return last_move

 

# def func(x):
	
# 	x = np.array(x)
# 	x2x0 = x[2] - x[0]
# 	x1x0 = x[1] - x[0]
# 	if np.all(x2x0 == np.array((0,0))):
# 		last_move = 0
  
# 	elif np.all(x2x0 == np.array([1, 1])):
# 		if np.all(x1x0 == np.array([0, 1])):
# 			last_move = 1
# 		else:
# 			last_move = -1
   
# 	elif np.all(x2x0 == np.array((0, 2))):
     
# 		if np.all(x1x0 == np.array((0, 1))):
# 			last_move = -1
# 		else:
# 			last_move = 1
# 	elif np.all(x2x0 == np.array((0, -2))):
# 		if np.all(x1x0 == np.array((0, -1))):
# 			last_move = 1
# 		else:
# 			last_move = -1
# 	elif np.all(x2x0 == np.array((1, -1))):
# 		if np.all(x1x0 == np.array((0, -1))):
# 			last_move = -1
   
# 		else:
# 			last_move = 1
# 	else:
# 		# x1x0 = x[1] - x[0]
# 		if np.all(x2x0 == np.array([-1, 1])):
# 			if np.all(x1x0 == np.array([0, 1])):
# 				last_move = -1
# 			else:
# 				last_move = 1

# 		elif np.all(x2x0 == np.array([-1, -1])):
# 			if np.all(x1x0 == np.array([0, -1])):
# 				last_move = 1
# 			else:
# 				last_move = -1
	
# 	return last_move


# pos = np.array([(2, 17), (2, 18), (3,18), (3, 17), (3, 16), (2, 16)])
# pos = np.array([(2, 17), (2, 16), (3, 16), (3, 17), (3,18), (2, 18)])

# coords = [m.coords[i] for i in [(2, 17), (2, 18), (3,18), (3, 17), (3, 16), (2, 16)]]

# for i in range(len(pos)):

# 	if i == 4:
# 		x = pos[(4, 5, 0), :]
# 	elif i == 5:
# 		x = pos[(5, 0, 1), :]
# 	else:
# 		x = pos[range(i, i+3)]
  
# 	print(func(x))
  
# # hexagon cap a la dreta
# coords = np.array([m.coords[i] for i in [(2, 17), (2, 18), (3,18), (3, 17), (3, 16), (2, 16)]])
# # hexagon cap a l'esquerra
# coords = np.array([m.coords[i] for i in [(2, 17), (2, 16), (3, 16), (3, 17), (3,18), (2, 18)]])

# for i in range(len(coords)):
    
# 	if i == 4:
# 		x = coords[(4, 5, 0), :]
# 	elif i == 5:
# 		x = coords[(5, 0, 1), :]
# 	else:
# 		x = coords[range(i, i+3)]
  
# 	print(func(x))


# # caminet d'esquerra a dreta
# coords = np.array([m.coords[i] for i in [(2,18), (3,18), (3, 17), (4, 17), (4, 18), (5, 18)]])
# for i in range(len(coords)):
    
# 	if i == 4:
# 		x = np.array([m.coords[(4, 18)], m.coords[(5, 18)], m.coords[(5, 17)]])
# 	elif i == 5:
# 		x = np.array([m.coords[(5, 18)], m.coords[(5, 17)], m.coords[(6, 17)]])
# 	else:
# 		x = coords[range(i, i+3)]
  
# 	print(func(x))
 
 
# # caminet de dreta a esquerra
# coords = np.array([m.coords[i] for i in [(6, 17), (5, 17), (5, 18), (4, 18), (4, 17), (3, 17), (3,18), (2,18)]])
# for i in range(len(coords)-2):

# 	x = coords[range(i, i+3)]
  
# 	print(func(x))
 
# # caminet d'esquerra a dreta
# coords = np.array([m.coords[i] for i in [(2, 16), (3, 16), (3, 17), (4, 17), (4, 16), (5, 16), (5,17), (6, 17)]])
# for i in range(len(coords)-2):

# 	x = coords[range(i, i+3)]
  
# 	print(func(x))
 
# # caminet de dreta a esquerra
# coords = np.array([m.coords[i] for i in [(6, 17), (5,17),(5, 16),(4, 16), (4, 17), (3, 17),(3, 16), (2, 16)]])
# for i in range(len(coords)-2):

# 	x = coords[range(i, i+3)]
  
# 	print(func(x))

# def convert_movement(move_history, neighbors):
	
# 	x1, x2, x3 = move_history
# 	if x2[0] == x3[0]:
# 		if x2[1] < x3[1]:
# 			m2 = 'up'
# 		else:
# 			m2 = 'down'
   
	
# 	d = [np.array(neighbors[i])- np.array(curr_xy) for i in neighbors]
	



	
