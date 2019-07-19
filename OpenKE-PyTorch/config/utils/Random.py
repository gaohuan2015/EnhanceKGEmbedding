import random
import time

def rand(a, b):
	'''
	create a random int [min, max-1]
	'''
	lef = min(a, b)
	rig = max(a, b)
	return random.randint(lef, rig - 1)

def _randReset(conArg, inData):
	'''
	为每个线程分配一个随机数
	'''
	workThreads = conArg.workThreads

	next_random = []
	for i in range(workThreads):
		next_random.append(rand(0, 2147483647))

	inData.next_random = next_random
	

def randd(next_random, id):

	return rand(0, 2147483647)

	# next_random[id] = next_random[id] * 25214903917 + 11
	# return next_random[id]

def rand_max(next_random, id, x):
	'''
	为当前的线程返回一个[0,x - 1]的随机数
	'''


	res = randd(next_random, id) % x
	while (res < 0):
		res += x

	return res