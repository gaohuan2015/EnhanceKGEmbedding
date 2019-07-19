
import numpy as np
from .Corrupt import _find

# INT lastValidHead = 0
# INT lastValidTail = 0
	
# REAL l_valid_filter_tot = 0
# REAL r_valid_filter_tot = 0


def _validInit(inData):
	inData.lastValidHead = 0
	inData.lastValidTail = 0
	inData.l_valid_filter_tot = 0
	inData.r_valid_filter_tot = 0


def _getValidHeadBatch(inData):

	lastValidHead = inData.lastValidHead
	validList = inData.validList
	entityTotal = inData.entityTotal

	valid_h = np.zeros(entityTotal, dtype=np.int64)
	valid_t = np.zeros(entityTotal, dtype=np.int64)
	valid_r = np.zeros(entityTotal, dtype=np.int64)

	for i in range(entityTotal):
		valid_h[i] = i
		valid_t[i] = validList[lastValidHead]['t']
		valid_r[i] = validList[lastValidHead]['r']

	return valid_h, valid_t, valid_r
	


def _getValidTailBatch(inData):

	lastValidTail = inData.lastValidTail
	validList = inData.validList
	entityTotal = inData.entityTotal

	valid_h = np.zeros(entityTotal, dtype=np.int64)
	valid_t = np.zeros(entityTotal, dtype=np.int64)
	valid_r = np.zeros(entityTotal, dtype=np.int64)

	for i in range(entityTotal):
		valid_h[i] = validList[lastValidTail]['h']
		valid_t[i] = i
		valid_r[i] = validList[lastValidTail]['r']

	return valid_h, valid_t, valid_r


def _validHead(con, inData):

	validList = inData.validList
	lastValidHead = inData.lastValidHead
	l_valid_filter_tot = inData.l_valid_filter_tot
	entityTotal = inData.entityTotal

	h = validList[lastValidHead]['h']
	t = validList[lastValidHead]['t']
	r = validList[lastValidHead]['r']

	# 这里con应该是一个np array有待确定结构 TODO
	minimal = con[h]
	
	l_filter_s = 0
	for j in range(entityTotal):
		if j != h:
			value = con[j]
			if (value < minimal) and (not _find(j, t, r, inData)):
				l_filter_s += 1

	if (l_filter_s < 10):
		l_valid_filter_tot += 1
	lastValidHead += 1

	inData.lastValidHead = lastValidHead
	inData.l_valid_filter_tot = l_valid_filter_tot



def _validTail(con, inData):

	validList = inData.validList
	lastValidTail = inData.lastValidTail
	entityTotal = inData.entityTotal
	r_valid_filter_tot = inData.r_valid_filter_tot

	h = validList[lastValidTail]['h']
	t = validList[lastValidTail]['t']
	r = validList[lastValidTail]['r']

	minimal = con[t]
	r_filter_s = 0
	for j in range(entityTotal):
		if j != t:
			value = con[j]
			if (value < minimal) and (not _find(h, j, r, inData)):
				r_filter_s += 1

	if (r_filter_s < 10):
		r_valid_filter_tot += 1
	lastValidTail += 1

	inData.lastValidTail = lastValidTail
	inData.r_valid_filter_tot = r_valid_filter_tot



def _getValidHit10(inData):

	l_valid_filter_tot = inData.l_valid_filter_tot
	r_valid_filter_tot = inData.r_valid_filter_tot
	validTotal = inData.validTotal

	l_valid_filter_tot /= validTotal
	r_valid_filter_tot /= validTotal
	validHit10 = (l_valid_filter_tot + r_valid_filter_tot) / 2

	inData.l_valid_filter_tot = l_valid_filter_tot
	inData.r_valid_filter_tot = r_valid_filter_tot
	inData.validHit10 = validHit10
	
