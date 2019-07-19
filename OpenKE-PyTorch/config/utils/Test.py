import copy
import numpy as np
from .Corrupt import _find, corrupt

# =====================================================================================
# link prediction
# ======================================================================================*/

def _initTest():
    '''
    参数初始化

    这个功能应该可以不用
    这些参数会在DataRead初始化的时候进行初始化
    '''
    return 
    # lastHead = 0
    # lastTail = 0
    # l1_filter_tot = 0, l1_tot = 0, r1_tot = 0, r1_filter_tot = 0, l_tot = 0, r_tot = 0, l_filter_rank = 0, l_rank = 0, l_filter_reci_rank = 0, l_reci_rank = 0
    # l3_filter_tot = 0, l3_tot = 0, r3_tot = 0, r3_filter_tot = 0, l_filter_tot = 0, r_filter_tot = 0, r_filter_rank = 0, r_rank = 0, r_filter_reci_rank = 0, r_reci_rank = 0

    # l1_filter_tot_constrain = 0, l1_tot_constrain = 0, r1_tot_constrain = 0, r1_filter_tot_constrain = 0, l_tot_constrain = 0, r_tot_constrain = 0, l_filter_rank_constrain = 0, l_rank_constrain = 0, l_filter_reci_rank_constrain = 0, l_reci_rank_constrain = 0
    # l3_filter_tot_constrain = 0, l3_tot_constrain = 0, r3_tot_constrain = 0, r3_filter_tot_constrain = 0, l_filter_tot_constrain = 0, r_filter_tot_constrain = 0, r_filter_rank_constrain = 0, r_rank_constrain = 0, r_filter_reci_rank_constrain = 0, r_reci_rank_constrain = 0


def _getHeadBatch(inData):

    '''
    在config中仅有一个地方调用该函数，所以该函数的内容是可以写死的
    '''

    entityTotal = inData.entityTotal
    testList = inData.testList
    lastHead = inData.lastHead
    # print('lastHead', lastHead)
    test_h = np.zeros(entityTotal, dtype=np.int64)
    test_t = np.zeros(entityTotal, dtype=np.int64)
    test_r = np.zeros(entityTotal, dtype=np.int64)

    for i in range(entityTotal):
        test_h[i] = i
        test_t[i] = testList[lastHead]['t']
        test_r[i] = testList[lastHead]['r']

    return test_h, test_t, test_r


def _getTailBatch(inData):

    entityTotal = inData.entityTotal
    testList = inData.testList
    lastTail = inData.lastTail
    # print('lastTail', lastTail)
    test_h = np.zeros(entityTotal, dtype=np.int64)
    test_t = np.zeros(entityTotal, dtype=np.int64)
    test_r = np.zeros(entityTotal, dtype=np.int64)


    for i in range(entityTotal):

        test_h[i] = testList[lastTail]['h']
        test_t[i] = i
        test_r[i] = testList[lastTail]['r']

    return test_h, test_t, test_r



def _testHead(con, inData):

    testList = inData.testList
    lastHead = inData.lastHead
    head_type, head_lef, head_rig = inData.head_type, inData.head_lef, inData.head_rig
    entityTotal = inData.entityTotal

    l_tot, l_filter_tot = inData.l_tot, inData.l_filter_tot
    l_filter_rank, l_rank, l_filter_reci_rank, l_reci_rank = inData.l_filter_rank, inData.l_rank, inData.l_filter_reci_rank, inData.l_reci_rank 
    l_filter_tot_constrain, l_filter_rank_constrain, l_filter_reci_rank_constrain = inData.l_filter_tot_constrain, inData.l_filter_rank_constrain, inData.l_filter_reci_rank_constrain
    l_tot_constrain, l_rank_constrain, l_reci_rank_constrain = inData.l_tot_constrain, inData.l_rank_constrain, inData.l_reci_rank_constrain
    l1_tot, l1_filter_tot, l1_filter_tot_constrain, l1_tot_constrain = inData.l1_tot, inData.l1_filter_tot, inData.l1_filter_tot_constrain, inData.l1_tot_constrain   
    l3_tot, l3_filter_tot, l3_filter_tot_constrain, l3_tot_constrain = inData.l3_tot, inData.l3_filter_tot, inData.l3_filter_tot_constrain, inData.l3_tot_constrain

    # 此处被赋值的变量都是局部变量
    h = testList[lastHead]['h']
    t = testList[lastHead]['t']
    r = testList[lastHead]['r']
    lef = head_lef[r]
    rig = head_rig[r]

    minimal = con[h]
    # print('_testHead minimal', minimal)
    l_s = 0
    l_filter_s = 0
    l_s_constrain = 0
    l_filter_s_constrain = 0

    for j in range(entityTotal):
        if (j != h):
            value = con[j]
            if (value < minimal):
                l_s += 1
                if (not _find(j, t, r, inData)):
                    l_filter_s += 1

            while (lef < rig and head_type[lef] < j):
                lef += 1
            if (lef < rig and j == head_type[lef]):
                if (value < minimal):
                    l_s_constrain += 1
                    if (not _find(j, t, r, inData)):
                        l_filter_s_constrain += 1

    # print('testHead l_s', l_s, l_filter_s)
    if (l_filter_s < 10):
        l_filter_tot += 1
    if (l_s < 10):
        l_tot += 1
    if (l_filter_s < 3): 
        l3_filter_tot += 1
    if (l_s < 3):
        l3_tot += 1
    if (l_filter_s < 1): 
        l1_filter_tot += 1
    if (l_s < 1):
        l1_tot += 1

    if (l_filter_s_constrain < 10):
        l_filter_tot_constrain += 1
    if (l_s_constrain < 10):
        l_tot_constrain += 1
    if (l_filter_s_constrain < 3): 
        l3_filter_tot_constrain += 1
    if (l_s_constrain < 3):
     l3_tot_constrain += 1
    if (l_filter_s_constrain < 1): 
        l1_filter_tot_constrain += 1
    if (l_s_constrain < 1): 
        l1_tot_constrain += 1

    l_filter_rank += (l_filter_s + 1)
    l_rank += (1 + l_s)
    l_filter_reci_rank += 1.0 / (l_filter_s+1)
    l_reci_rank += 1.0 / (l_s + 1)

    l_filter_rank_constrain += (l_filter_s_constrain + 1)
    l_rank_constrain += (1 + l_s_constrain)
    l_filter_reci_rank_constrain += 1.0 / (l_filter_s_constrain + 1)
    l_reci_rank_constrain += 1.0 / (l_s_constrain + 1)

    lastHead += 1
    # print('test_head', l_tot, l3_tot, l1_tot)
    inData.lastHead = lastHead
    inData.l_tot, inData.l_filter_tot = l_tot, l_filter_tot
    inData.l_filter_rank, inData.l_rank, inData.l_filter_reci_rank, inData.l_reci_rank  = l_filter_rank, l_rank, l_filter_reci_rank, l_reci_rank 
    inData.l_filter_tot_constrain, inData.l_filter_rank_constrain, inData.l_filter_reci_rank_constrain = l_filter_tot_constrain, l_filter_rank_constrain, l_filter_reci_rank_constrain
    inData.l_tot_constrain, inData.l_rank_constrain, inData.l_reci_rank_constrain = l_tot_constrain, l_rank_constrain, l_reci_rank_constrain
    inData.l1_tot, inData.l1_filter_tot, inData.l1_filter_tot_constrain, inData.l1_tot_constrain = l1_tot, l1_filter_tot, l1_filter_tot_constrain, l1_tot_constrain   
    inData.l3_tot, inData.l3_filter_tot, inData.l3_filter_tot_constrain, inData.l3_tot_constrain = l3_tot, l3_filter_tot, l3_filter_tot_constrain, l3_tot_constrain




def _testTail(con, inData):

    testList = inData.testList
    lastTail = inData.lastTail
    tail_type, tail_lef, tail_rig = inData.tail_type, inData.tail_lef, inData.tail_rig
    entityTotal = inData.entityTotal


    r_tot, r_filter_tot = inData.r_tot, inData.r_filter_tot
    r_filter_rank, r_rank, r_filter_reci_rank, r_reci_rank = inData.r_filter_rank, inData.r_rank, inData.r_filter_reci_rank, inData.r_reci_rank 
    r_filter_tot_constrain, r_filter_rank_constrain, r_filter_reci_rank_constrain = inData.r_filter_tot_constrain, inData.r_filter_rank_constrain, inData.r_filter_reci_rank_constrain
    r_tot_constrain, r_rank_constrain, r_reci_rank_constrain = inData.r_tot_constrain, inData.r_rank_constrain, inData.r_reci_rank_constrain
    r1_tot, r1_filter_tot, r1_filter_tot_constrain, r1_tot_constrain = inData.r1_tot, inData.r1_filter_tot, inData.r1_filter_tot_constrain, inData.r1_tot_constrain   
    r3_tot, r3_filter_tot, r3_filter_tot_constrain, r3_tot_constrain = inData.r3_tot, inData.r3_filter_tot, inData.r3_filter_tot_constrain, inData.r3_tot_constrain


    h = testList[lastTail]['h']
    t = testList[lastTail]['t']
    r = testList[lastTail]['r']
    lef = tail_lef[r]
    rig = tail_rig[r]

    minimal = con[t] # true label的预测概率
    r_s = 0
    r_filter_s = 0
    r_s_constrain = 0
    r_filter_s_constrain = 0

    for j in range(entityTotal):
        if (j != t):
            value = con[j] # 这个应当是预测为结果的概率
            if (value < minimal):
                r_s += 1 
                if (not _find(h, j, r, inData)):
                    r_filter_s += 1

            while (lef < rig and tail_type[lef] < j):
                lef += 1
            if (lef < rig and j == tail_type[lef]):
                    if (value < minimal):
                        r_s_constrain += 1
                        if (not _find(h, j ,r, inData)):
                            r_filter_s_constrain += 1
    # print('testTail r_s', r_s, r_filter_s)
    # r链接预测结果统计
    if (r_filter_s < 10):
        r_filter_tot += 1 # filter r hit@10
    if (r_s < 10):
        r_tot += 1 # r hit@10
    if (r_filter_s < 3): 
        r3_filter_tot += 1 # filter r hit@3
    if (r_s < 3):
        r3_tot += 1
    if (r_filter_s < 1): 
        r1_filter_tot += 1
    if (r_s < 1):
        r1_tot += 1

    # print('testTail', r3_tot, r3_filter_tot)

    if (r_filter_s_constrain < 10):
        r_filter_tot_constrain += 1
    if (r_s_constrain < 10):
        r_tot_constrain += 1
    if (r_filter_s_constrain < 3): 
        r3_filter_tot_constrain += 1
    if (r_s_constrain < 3):
        r3_tot_constrain += 1
    if (r_filter_s_constrain < 1): 
        r1_filter_tot_constrain += 1
    if (r_s_constrain < 1): 
        r1_tot_constrain += 1

    r_filter_rank += (1 + r_filter_s)
    r_rank += (1 + r_s)
    r_filter_reci_rank += 1.0 / (1 + r_filter_s)
    r_reci_rank += 1.0 / (1 + r_s)

    r_filter_rank_constrain += (1 + r_filter_s_constrain)
    r_rank_constrain += (1 + r_s_constrain)
    r_filter_reci_rank_constrain += 1. / (1 + r_filter_s_constrain)
    r_reci_rank_constrain += 1.0 / (1 + r_s_constrain)

    lastTail += 1

    inData.lastTail = lastTail
    inData.r_tot, inData.r_filter_tot = r_tot, r_filter_tot
    inData.r_filter_rank, inData.r_rank, inData.r_filter_reci_rank, inData.r_reci_rank  = r_filter_rank, r_rank, r_filter_reci_rank, r_reci_rank 
    inData.r_filter_tot_constrain, inData.r_filter_rank_constrain, inData.r_filter_reci_rank_constrain = r_filter_tot_constrain, r_filter_rank_constrain, r_filter_reci_rank_constrain
    inData.r_tot_constrain, inData.r_rank_constrain, inData.r_reci_rank_constrain = r_tot_constrain, r_rank_constrain, r_reci_rank_constrain
    inData.r1_tot, inData.r1_filter_tot, inData.r1_filter_tot_constrain, inData.r1_tot_constrain = r1_tot, r1_filter_tot, r1_filter_tot_constrain, r1_tot_constrain   
    inData.r3_tot, inData.r3_filter_tot, inData.r3_filter_tot_constrain, inData.r3_tot_constrain = r3_tot, r3_filter_tot, r3_filter_tot_constrain, r3_tot_constrain



    # 以下部分为bad case收集
    item = {}
    item['h'] = h
    item['r'] = r
    item['t'] = t
    item['rank'] = r_s
    item['h_freq'] = inData.freqEnt[h]
    item['r_freq'] = inData.freqRel[r]
    item['t_freq'] = inData.freqEnt[t]
    return item







def _test_link_prediction(inData):

    testTotal = inData.testTotal

    l_tot, l_filter_tot = inData.l_tot, inData.l_filter_tot
    l_filter_rank, l_rank, l_filter_reci_rank, l_reci_rank = inData.l_filter_rank, inData.l_rank, inData.l_filter_reci_rank, inData.l_reci_rank 
    l_filter_tot_constrain, l_filter_rank_constrain, l_filter_reci_rank_constrain = inData.l_filter_tot_constrain, inData.l_filter_rank_constrain, inData.l_filter_reci_rank_constrain
    l_tot_constrain, l_rank_constrain, l_reci_rank_constrain = inData.l_tot_constrain, inData.l_rank_constrain, inData.l_reci_rank_constrain
    l1_tot, l1_filter_tot, l1_filter_tot_constrain, l1_tot_constrain = inData.l1_tot, inData.l1_filter_tot, inData.l1_filter_tot_constrain, inData.l1_tot_constrain   
    l3_tot, l3_filter_tot, l3_filter_tot_constrain, l3_tot_constrain = inData.l3_tot, inData.l3_filter_tot, inData.l3_filter_tot_constrain, inData.l3_tot_constrain

    r_tot, r_filter_tot = inData.r_tot, inData.r_filter_tot
    r_filter_rank, r_rank, r_filter_reci_rank, r_reci_rank = inData.r_filter_rank, inData.r_rank, inData.r_filter_reci_rank, inData.r_reci_rank 
    r_filter_tot_constrain, r_filter_rank_constrain, r_filter_reci_rank_constrain = inData.r_filter_tot_constrain, inData.r_filter_rank_constrain, inData.r_filter_reci_rank_constrain
    r_tot_constrain, r_rank_constrain, r_reci_rank_constrain = inData.r_tot_constrain, inData.r_rank_constrain, inData.r_reci_rank_constrain
    r1_tot, r1_filter_tot, r1_filter_tot_constrain, r1_tot_constrain = inData.r1_tot, inData.r1_filter_tot, inData.r1_filter_tot_constrain, inData.r1_tot_constrain   
    r3_tot, r3_filter_tot, r3_filter_tot_constrain, r3_tot_constrain = inData.r3_tot, inData.r3_filter_tot, inData.r3_filter_tot_constrain, inData.r3_tot_constrain

    l_rank /= testTotal
    r_rank /= testTotal
    l_reci_rank /= testTotal
    r_reci_rank /= testTotal
 
    # print('link_predic', l_tot, l3_tot, l1_tot)


    l_tot /= testTotal
    l3_tot /= testTotal
    l1_tot /= testTotal
 
    r_tot /= testTotal
    r3_tot /= testTotal
    r1_tot /= testTotal

    # with filter
    l_filter_rank /= testTotal
    r_filter_rank /= testTotal
    l_filter_reci_rank /= testTotal
    r_filter_reci_rank /= testTotal
 
    l_filter_tot /= testTotal
    l3_filter_tot /= testTotal
    l1_filter_tot /= testTotal
 
    r_filter_tot /= testTotal
    r3_filter_tot /= testTotal
    r1_filter_tot /= testTotal

    print("no type constraint results:\n")
    
    print("metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \n")
    print("l(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n" % (l_reci_rank, l_rank, l_tot, l3_tot, l1_tot))
    print("r(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n" % (r_reci_rank, r_rank, r_tot, r3_tot, r1_tot))
    print("averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \n" %
            ((l_reci_rank+r_reci_rank)/2, (l_rank+r_rank)/2, (l_tot+r_tot)/2, (l3_tot+r3_tot)/2, (l1_tot+r1_tot)/2))
    print("\n")
    print("l(filter):\t\t %f \t %f \t %f \t %f \t %f \n" % (l_filter_reci_rank, l_filter_rank, l_filter_tot, l3_filter_tot, l1_filter_tot))
    print("r(filter):\t\t %f \t %f \t %f \t %f \t %f \n" % (r_filter_reci_rank, r_filter_rank, r_filter_tot, r3_filter_tot, r1_filter_tot))
    print("averaged(filter):\t %f \t %f \t %f \t %f \t %f \n" % ((l_filter_reci_rank+r_filter_reci_rank)/2, (l_filter_rank+r_filter_rank)/2, (l_filter_tot+r_filter_tot)/2, (l3_filter_tot+r3_filter_tot)/2, (l1_filter_tot+r1_filter_tot)/2))

    # type constrain
    l_rank_constrain /= testTotal
    r_rank_constrain /= testTotal
    l_reci_rank_constrain /= testTotal
    r_reci_rank_constrain /= testTotal
 
    l_tot_constrain /= testTotal
    l3_tot_constrain /= testTotal
    l1_tot_constrain /= testTotal
 
    r_tot_constrain /= testTotal
    r3_tot_constrain /= testTotal
    r1_tot_constrain /= testTotal

    # with filter
    l_filter_rank_constrain /= testTotal
    r_filter_rank_constrain /= testTotal
    l_filter_reci_rank_constrain /= testTotal
    r_filter_reci_rank_constrain /= testTotal
 
    l_filter_tot_constrain /= testTotal
    l3_filter_tot_constrain /= testTotal
    l1_filter_tot_constrain /= testTotal
 
    r_filter_tot_constrain /= testTotal
    r3_filter_tot_constrain /= testTotal
    r1_filter_tot_constrain /= testTotal

    print("type constraint results:\n")
    
    print("metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \n")
    print("l(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n" % (l_reci_rank_constrain, l_rank_constrain, l_tot_constrain, l3_tot_constrain, l1_tot_constrain))
    print("r(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n" % (r_reci_rank_constrain, r_rank_constrain, r_tot_constrain, r3_tot_constrain, r1_tot_constrain))
    print("averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \n" % ((l_reci_rank_constrain+r_reci_rank_constrain)/2, (l_rank_constrain+r_rank_constrain)/2, (l_tot_constrain+r_tot_constrain)/2, (l3_tot_constrain+r3_tot_constrain)/2, (l1_tot_constrain+r1_tot_constrain)/2))
    print("\n")
    print("l(filter):\t\t %f \t %f \t %f \t %f \t %f \n" % (l_filter_reci_rank_constrain, l_filter_rank_constrain, l_filter_tot_constrain, l3_filter_tot_constrain, l1_filter_tot_constrain))
    print("r(filter):\t\t %f \t %f \t %f \t %f \t %f \n" % (r_filter_reci_rank_constrain, r_filter_rank_constrain, r_filter_tot_constrain, r3_filter_tot_constrain, r1_filter_tot_constrain))
    print("averaged(filter):\t %f \t %f \t %f \t %f \t %f \n" % ( (l_filter_reci_rank_constrain+r_filter_reci_rank_constrain)/2, (l_filter_rank_constrain+r_filter_rank_constrain)/2, (l_filter_tot_constrain+r_filter_tot_constrain)/2, (l3_filter_tot_constrain+r3_filter_tot_constrain)/2, (l1_filter_tot_constrain+r1_filter_tot_constrain)/2))



# =====================================================================================
# triple classification
# ======================================================================================*/


def _getNegTest(inData):

    negTestList = []
    testTotal = inData.testTotal
    testList = inData.testList
    # negTestList = [-1] * testTotal

    for i in range(testTotal):
        negTestList.append(copy.copy(testList[i]))
        negTestList[i]['t'] = corrupt(testList[i]['h'], testList[i]['r'], inData)
    inData.negTestList = negTestList


def _getNegValid(inData):

    negValidList = []
    validTotal = inData.validTotal
    validList = inData.validList

    # print('HERE', len(validList), validTotal)
    for i in range(validTotal):
        negValidList.append(copy.copy(validList[i]))
        negValidList[i]['t'] = corrupt(validList[i]['h'], validList[i]['r'], inData)

    inData.negValidList = negValidList


def _getTestBatch(inData):
    _getNegTest(inData)
    testTotal = inData.testTotal
    testList = inData.testList
    negTestList = inData.negTestList

    test_pos_h = np.zeros(testTotal, dtype=np.int64)
    test_pos_t = np.zeros(testTotal, dtype=np.int64)
    test_pos_r = np.zeros(testTotal, dtype=np.int64)
    test_neg_h = np.zeros(testTotal, dtype=np.int64)
    test_neg_t = np.zeros(testTotal, dtype=np.int64)
    test_neg_r = np.zeros(testTotal, dtype=np.int64)    

    for i in range(testTotal):
        test_pos_h[i] = testList[i]['h']
        test_pos_t[i] = testList[i]['t']
        test_pos_r[i] = testList[i]['r']
        test_neg_h[i] = negTestList[i]['h']
        test_neg_t[i] = negTestList[i]['t']
        test_neg_r[i] = negTestList[i]['r']
    

    return test_pos_h,test_pos_t,test_pos_r,test_neg_h,test_neg_t,test_neg_r


def _getValidBatch(inData):
    _getNegValid(inData)
    validTotal = inData.validTotal
    validList = inData.validList
    negValidList = inData.negValidList
    valid_pos_h = np.zeros(validTotal, dtype=np.int64)
    valid_pos_t = np.zeros(validTotal, dtype=np.int64)
    valid_pos_r = np.zeros(validTotal, dtype=np.int64)
    valid_neg_h = np.zeros(validTotal, dtype=np.int64)
    valid_neg_t = np.zeros(validTotal, dtype=np.int64)
    valid_neg_r = np.zeros(validTotal, dtype=np.int64)    

    for i in range(validTotal):
        valid_pos_h[i] = validList[i]['h']
        valid_pos_t[i] = validList[i]['t']
        valid_pos_r[i] = validList[i]['r']
        valid_neg_h[i] = negValidList[i]['h']
        valid_neg_t[i] = negValidList[i]['t']
        valid_neg_r[i] = negValidList[i]['r']
    
    return valid_pos_h,valid_pos_t,valid_pos_r,valid_neg_h,valid_neg_t,valid_neg_r


# REAL threshEntire

def _getBestThreshold(score_pos, score_neg, inData):

    relationTotal = inData.relationTotal
    validLef = inData.validLef
    validRig = inData.validRig

    relThresh = np.zeros(relationTotal, dtype=np.float32)

    interval = 0.01
    min_score, max_score, bestThresh, tmpThresh, bestAcc, tmpAcc = 0, 0, 0, 0, 0, 0
    n_interval, correct, total = 0, 0, 0

    for r in range(relationTotal):
        if (validLef[r] == -1):
            continue

        total = (validRig[r] - validLef[r] + 1) * 2
        min_score = score_pos[validLef[r]]

        if (score_neg[validLef[r]] < min_score):
            min_score = score_neg[validLef[r]]
        max_score = score_pos[validLef[r]]

        if (score_neg[validLef[r]] > max_score):
            max_score = score_neg[validLef[r]]

        for i in range(validLef[r] + 1, validRig[r] + 1):
            if(score_pos[i] < min_score):
                min_score = score_pos[i]
            if(score_pos[i] > max_score): 
                max_score = score_pos[i]
            if(score_neg[i] < min_score): 
                min_score = score_neg[i]
            if(score_neg[i] > max_score): 
                max_score = score_neg[i]
        
        n_interval = int((max_score - min_score) / interval)
        for i in range(n_interval+1):
            tmpThresh = min_score + i * interval
            correct = 0
            for j in range(validLef[r], validRig[r] + 1):
                if (score_pos[j] <= tmpThresh): 
                    correct += 1
                if (score_neg[j] > tmpThresh): 
                    correct += 1
            
            tmpAcc = 1.0 * correct / total
            if (i == 0):
                bestThresh = tmpThresh
                bestAcc = tmpAcc
            elif (tmpAcc > bestAcc):
                bestAcc = tmpAcc
                bestThresh = tmpThresh
            
        
        relThresh[r] = bestThresh
        
        inData.relThresh = relThresh


def _test_triple_classification(score_pos, score_neg, inData):

    relationTotal = inData.relationTotal
    relThresh = inData.relThresh
    validLef, validRig = inData.validLef, inData.validRig
    testLef, testRig = inData.testLef, inData.testRig

    testAcc = [-1] * relationTotal
    aveCorrect = 0
    aveTotal = 0
    aveAcc = 0

    for r in range(relationTotal):

        if (validLef[r] == -1 or testLef[r] ==-1):
            continue
        correct = 0
        total = 0
        for i in range(testLef[r], testRig[r] + 1):
            if (score_pos[i] <= relThresh[r]):
                correct += 1
            if (score_neg[i] > relThresh[r]):
                correct += 1
            total += 2
        
        testAcc[r] = 1.0 * correct / total
        aveCorrect += correct 
        aveTotal += total
    


    aveAcc = 1.0 * aveCorrect / aveTotal


    inData.testAcc = testAcc
    inData.aveAcc = aveAcc

    print("triple classification accuracy is %lf\n" % aveAcc)

