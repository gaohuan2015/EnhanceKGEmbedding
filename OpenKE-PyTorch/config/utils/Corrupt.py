from .Random import rand_max, rand
import random

def  corrupt_head(id, h, r, inData):
	'''
	id 是线程编号
	意义还有待理解，到底是打乱head还是打乱tail
	测试情况：
	'''
	entityTotal = inData.entityTotal
	trainHead, lefHead, rigHead = inData.trainHead, inData.lefHead, inData.rigHead 
	next_random = inData.next_random

	# 二分搜索，在trainHead中确定三元组(h, r, *)的所在范围[ll, rr]
	lef = lefHead[h] - 1
	rig = rigHead[h]
	while (lef + 1 < rig):
		mid = (lef + rig) >> 1
		if (trainHead[mid]['r'] >= r):
			rig = mid 
		else:
			lef = mid

	ll = rig
	lef = lefHead[h]
	rig = rigHead[h] + 1
	while (lef + 1 < rig):
		mid = (lef + rig) >> 1
		if (trainHead[mid]['r'] <= r):
			lef = mid 
		else:
			rig = mid
	rr = lef


	tmp = rand_max(next_random, id, entityTotal - (rr - ll + 1)) # tmp是随机生成的t编号

	# 一定在该范围外，返回
	if tmp < trainHead[ll]['t']:
		return tmp

	if tmp > trainHead[rr]['t'] - rr + ll - 1:
		return tmp + rr - ll + 1

	lef = ll
	rig = rr + 1

	while lef + 1 < rig:
		mid = (lef + rig) >> 1
		if (trainHead[mid]['t'] - mid + ll - 1 < tmp):
			lef = mid
		else:
			rig = mid

	return tmp + lef - ll + 1


def  corrupt_tail(id, t, r, inData):
	'''

	'''
	entityTotal = inData.entityTotal
	trainTail, lefTail, rigTail = inData.trainTail, inData.lefTail, inData.rigTail 
	next_random = inData.next_random

	lef = lefTail[t] - 1
	rig = rigTail[t]
	while (lef + 1 < rig):
		mid = (lef + rig) >> 1
		if (trainTail[mid]['r'] >= r):
			rig = mid 
		else:
			lef = mid

	ll = rig
	lef = lefTail[t]
	rig = rigTail[t] + 1
	while (lef + 1 < rig):
		mid = (lef + rig) >> 1
		if (trainTail[mid]['r'] <= r):
			lef = mid 
		else:
			rig = mid
	rr = lef
	tmp = rand_max(next_random, id, entityTotal - (rr - ll + 1))
	if (tmp < trainTail[ll]['h']):
		return tmp
	if (tmp > trainTail[rr]['h'] - rr + ll - 1):
		return tmp + rr - ll + 1
	lef = ll
	rig = rr + 1
	while (lef + 1 < rig):
		mid = (lef + rig) >> 1
		if (trainTail[mid]['h'] - mid + ll - 1 < tmp):
			lef = mid
		else:
			rig = mid
	
	return tmp + lef - ll + 1


def corrupt_rel(id, h, t, inData):
	
	relationTotal = inData.relationTotal
	trainRel, lefRel, rigRel = inData.trainRel, inData.lefRel, inData.rigRel 
	next_random = inData.next_random

	lef = lefRel[h] - 1
	rig = rigRel[h]
	while (lef + 1 < rig):
		mid = (lef + rig) >> 1
		if (trainRel[mid]['t'] >= t):
			rig = mid 
		else:
			lef = mid
	ll = rig
	lef = lefRel[h]
	rig = rigRel[h] + 1
	while (lef + 1 < rig):
		mid = (lef + rig) >> 1
		if (trainRel[mid]['t'] <= t):
			lef = mid 
		else:
			rig = mid
	rr = lef
	tmp = rand_max(next_random, id, relationTotal - (rr - ll + 1))
	if (tmp < trainRel[ll]['r']):
		return tmp
	if (tmp > trainRel[rr]['r'] - rr + ll - 1):
		return tmp + rr - ll + 1
	lef = ll, rig = rr + 1
	while (lef + 1 < rig):
		mid = (lef + rig) >> 1
		if (trainRel[mid]['r'] - mid + ll - 1 < tmp):
			lef = mid
		else:
			rig = mid
	return tmp + lef - ll + 1


def _find(h, t, r, inData):
	'''
	二分查找三元组
	'''

	tripleTotal = inData.tripleTotal
	tripleList = inData.tripleList

	lef = 0
	rig = tripleTotal - 1
	mid = -1
	while (lef + 1 < rig):
		mid = (lef + rig) >> 1

		# print('tripleTotal', len(tripleList))
		# print('mid', mid)
		# print('tripleList[mid]', tripleList[mid])

		if ((tripleList[mid]['h'] < h) or (tripleList[mid]['h'] == h and tripleList[mid]['r'] < r) or (tripleList[mid]['h'] == h and tripleList[mid]['r'] == r and tripleList[mid]['t'] < t)):
			lef = mid 
		else:
			rig = mid
 
	if (tripleList[lef]['h'] == h and tripleList[lef]['r'] == r and tripleList[lef]['t'] == t):
		return True
	if (tripleList[rig]['h'] == h and tripleList[rig]['r'] == r and tripleList[rig]['t'] == t):
		return True
	return False

def corrupt(h, r, inData):
	'''
	
	'''
	tail_lef, tail_rig, tail_type = inData.tail_lef, inData.tail_rig, inData.tail_type

	ll = tail_lef[r]
	rr = tail_rig[r]
	loop = 0
	t = -1
	while(1):
		tmp = rand(ll, rr)
		t = tail_type[tmp]
		if (not _find(h, t, r, inData)):
			return t
		else:
			loop += 1
			if (loop >= 1000):
				return corrupt_head(0, h, r, inData)

