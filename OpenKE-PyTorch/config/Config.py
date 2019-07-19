# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
# from .utils import Base
from .DataReader import *

def to_var(x):
    # return Variable(torch.from_numpy(x).cuda())
    return Variable(torch.from_numpy(x))


class Config(object):
    def __init__(self):

        """set essential parameters"""
        self.in_path = "./"
        self.batch_size = 100
        self.bern = 0
        self.work_threads = 8
        self.hidden_size = 100
        self.negative_ent = 1
        self.negative_rel = 0
        self.ent_size = self.hidden_size
        self.rel_size = self.hidden_size
        self.margin = 1.0
        self.valid_steps = 5
        self.save_steps = 5
        self.opt_method = "SGD"
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.lmbda = 0.0
        self.alpah = 0.001
        self.early_stopping_patience = 10
        self.nbatches = 100
        self.p_norm = 1
        self.test_link = True
        self.test_triple = True
        self.model = None
        self.trainModel = None
        self.testModel = None
        self.pretrain_model = None

        self.DataReader = DataReader()


    def init(self):

        self.DataReader.setInPath(self.in_path)
        self.DataReader.setBern(self.bern)
        self.DataReader.setNegEnt(1)
        self.DataReader.setNegRel(0)

        # self.DataReader.setWorkThreads(1)
        self.DataReader.randReset()
        self.DataReader.importTrainFiles()
        self.DataReader.importTestFiles()
        self.DataReader.importTypeFiles()
        self.relTotal = self.DataReader.getRelationTotal()
        self.entTotal = self.DataReader.getEntityTotal()
        self.trainTotal = self.DataReader.getTrainTotal()
        self.testTotal = self.DataReader.getTestTotal()
        self.validTotal = self.DataReader.getValidTotal()

        self.batch_size = int(self.trainTotal / self.nbatches)

        self.DataReader.setBatchSize(int(self.trainTotal / self.nbatches))

        self.batch_seq_size = self.batch_size * (
            1 + self.negative_ent + self.negative_rel
        )


    def set_test_link(self, test_link):
        self.test_link = test_link

    def set_test_triple(self, test_triple):
        self.test_triple = test_triple

    def set_margin(self, margin):
        self.margin = margin

    def set_in_path(self, in_path):
        self.in_path = in_path

    def set_nbatches(self, nbatches):
        self.nbatches = nbatches

    def set_p_norm(self, p_norm):
        self.p_norm = p_norm

    def set_valid_steps(self, valid_steps):
        self.valid_steps = valid_steps

    def set_save_steps(self, save_steps):
        self.save_steps = save_steps

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def set_result_dir(self, result_dir):
        self.result_dir = result_dir

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lmbda(self, lmbda):
        self.lmbda = lmbda

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_bern(self, bern):
        self.bern = bern

    def set_dimension(self, dim):
        self.hidden_size = dim
        self.ent_size = dim
        self.rel_size = dim

    def set_ent_dimension(self, dim):
        self.ent_size = dim

    def set_rel_dimension(self, dim):
        self.rel_size = dim

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_work_threads(self, work_threads):
        self.work_threads = work_threads

    def set_ent_neg_rate(self, rate):
        self.negative_ent = rate

    def set_rel_neg_rate(self, rate):
        self.negative_rel = rate

    def set_early_stopping_patience(self, early_stopping_patience):
        self.early_stopping_patience = early_stopping_patience

    def set_pretrain_model(self, pretrain_model):
        self.pretrain_model = pretrain_model

    def get_parameters(self, param_dict, mode="numpy"):
        for param in param_dict:
            param_dict[param] = param_dict[param].cpu()
        res = {}
        for param in param_dict:
            if mode == "numpy":
                res[param] = param_dict[param].numpy()
            elif mode == "list":
                res[param] = param_dict[param].numpy().tolist()
            else:
                res[param] = param_dict[param]
        return res

    def save_embedding_matrix(self, best_model):
        path = os.path.join(self.result_dir, self.model.__name__ + ".json")
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters(best_model, "list")))
        f.close()

    def set_train_model(self, model):
        print("Initializing training model...")
        self.model = model
        self.trainModel = self.model(config=self)
        # self.trainModel.cuda()
        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.trainModel.parameters(),
                lr=self.alpha,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.trainModel.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.trainModel.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.SGD(
                self.trainModel.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        print("Finish initializing")

    def set_test_model(self, model, path=None):
        print("Initializing test model...")
        self.model = model
        self.testModel = self.model(config=self)
        if path == None:
            path = os.path.join(self.result_dir, self.model.__name__ + ".ckpt")
        self.testModel.load_state_dict(torch.load(path))
        # self.testModel.cuda()
        self.testModel.eval()
        print("Finish initializing")

    def sampling(self):

        self.DataReader.sampling()


    def save_checkpoint(self, model, epoch):
        path = os.path.join(
            self.checkpoint_dir, self.model.__name__ + "-" + str(epoch) + ".ckpt"
        )
        torch.save(model, path)

    def save_best_checkpoint(self, best_model):
        path = os.path.join(self.result_dir, self.model.__name__ + ".ckpt")
        torch.save(best_model, path)

    def train_one_step(self):
        self.trainModel.batch_h = to_var(self.DataReader.batch_h)
        self.trainModel.batch_t = to_var(self.DataReader.batch_t)
        self.trainModel.batch_r = to_var(self.DataReader.batch_r)
        self.trainModel.batch_y = to_var(self.DataReader.batch_y)
        self.optimizer.zero_grad()
        loss = self.trainModel()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test_one_step(self, model, test_h, test_t, test_r):
        model.batch_h = to_var(test_h)
        model.batch_t = to_var(test_t)
        model.batch_r = to_var(test_r)

        # 这里应该搞清楚返回的数据是什么结构
        return model.predict()

    def valid(self, model):

        # 初始化
        self.DataReader.validInit()

        for i in range(self.validTotal):
            sys.stdout.write("%d\r" % (i))
            sys.stdout.flush()
            # 获取验证数据
            self.DataReader.getValidHeadBatch()
            # print(self.DataReader.valid_h)
            # 放入模型进行验证
            res = self.test_one_step(model, self.DataReader.valid_h, self.DataReader.valid_t, self.DataReader.valid_r)

            self.DataReader.validHead(res)

            self.DataReader.getValidTailBatch()
            res = self.test_one_step(model, self.DataReader.valid_h, self.DataReader.valid_t, self.DataReader.valid_r)
            self.DataReader.validTail(res)
            
        return self.DataReader.getValidHit10()

    def train(self):
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        best_epoch = 0
        best_hit10 = 0.0
        best_model = None
        bad_counts = 0

        for epoch in range(self.train_times):

            start_time = time.time()
            # 训练
            res = 0.0
            for batch in range(self.nbatches):
                self.sampling()
                loss = self.train_one_step()
                res += loss
            end_time = time.time()
            print("Epoch %d | loss: %f | cost time: %f" % (epoch, res, end_time - start_time))

            # 保存模型
            if (epoch + 1) % self.save_steps == 0:
                print("Epoch %d has finished, saving..." % (epoch))
                self.save_checkpoint(self.trainModel.state_dict(), epoch)

            # 验证模型
            if (epoch + 1) % self.valid_steps == 0:
                print("Epoch %d has finished, validating..." % (epoch))
                hit10 = self.valid(self.trainModel)
                if hit10 > best_hit10:
                    best_hit10 = hit10
                    best_epoch = epoch
                    best_model = copy.deepcopy(self.trainModel.state_dict())
                    print("Best model | hit@10 of valid set is %f" % (best_hit10))
                    bad_counts = 0
                else:
                    print(
                        "Hit@10 of valid set is %f | bad count is %d"
                        % (hit10, bad_counts)
                    )
                    bad_counts += 1
                if bad_counts == self.early_stopping_patience:
                    print("Early stopping at epoch %d" % (epoch))
                    break

        # 确定最优模型
        if best_model == None:
            best_model = self.trainModel.state_dict()
            best_epoch = self.train_times - 1
            best_hit10 = self.valid(self.trainModel)
        print("Best epoch is %d | hit@10 of valid set is %f" % (best_epoch, best_hit10))
        print("Store checkpoint of best result at epoch %d..." % (best_epoch))
        if not os.path.isdir(self.result_dir):
            os.mkdir(self.result_dir)
        self.save_best_checkpoint(best_model)
        self.save_embedding_matrix(best_model)

        # 测试
        print("Finish storing")
        print("Testing...")
        self.set_test_model(self.model)
        self.test()
        print("Finish test")
        return best_model

    def triple_classification(self):
        self.DataReader.getValidBatch()
        res_pos = self.test_one_step(
            self.testModel, 
            self.DataReader.valid_pos_h, 
            self.DataReader.valid_pos_t, 
            self.DataReader.valid_pos_r
        )
        res_neg = self.test_one_step(
            self.testModel, 
            self.DataReader.valid_neg_h, 
            self.DataReader.valid_neg_t, 
            self.DataReader.valid_neg_r
        )
        self.DataReader.getBestThreshold(res_pos, res_neg)

        self.DataReader.getTestBatch()
        res_pos = self.test_one_step(
            self.testModel,
            self.DataReader.test_pos_h, 
            self.DataReader.test_pos_t, 
            self.DataReader.test_pos_r
        )
        res_neg = self.test_one_step(
            self.testModel, 
            self.DataReader.test_neg_h, 
            self.DataReader.test_neg_t, 
            self.DataReader.test_neg_r
        )
        self.DataReader.test_triple_classification(res_pos, res_neg)



    def link_prediction(self):
        print("The total of test triple is %d" % (self.testTotal))
        for i in range(self.testTotal):
            sys.stdout.write("%d\r" % (i))
            sys.stdout.flush()

            self.DataReader.getHeadBatch()
            res = self.test_one_step(
                self.testModel, 
                self.DataReader.test_h,
                self.DataReader.test_t, 
                self.DataReader.test_r
            )
            self.DataReader.testHead(res)

            self.DataReader.getTailBatch()
            res = self.test_one_step(
                self.testModel, 
                self.DataReader.test_h,
                self.DataReader.test_t, 
                self.DataReader.test_r
            )
            self.DataReader.testTail(res)

        self.DataReader.test_link_prediction()



    # ================================数据分析部分==================================




    def analysis_link_prediction(self):
        print("The total of test triple is %d" % (self.testTotal))
        bad_list = []

        # self.testTotal = 200 # 临时设定

        for i in range(self.testTotal):
            sys.stdout.write("%d\r" % (i))
            sys.stdout.flush()

            self.DataReader.getHeadBatch()
            res = self.test_one_step(
                self.testModel, 
                self.DataReader.test_h,
                self.DataReader.test_t, 
                self.DataReader.test_r
            )
            self.DataReader.testHead(res)

            self.DataReader.getTailBatch()
            res = self.test_one_step(
                self.testModel, 
                self.DataReader.test_h,
                self.DataReader.test_t, 
                self.DataReader.test_r
            )


            # 数据处理,添加F范数
            bad = self.DataReader.testTail(res)
            h_embedding = self.testModel.getEntEmbedding(to_var(np.array([bad['h']], dtype=np.int64))).detach().numpy()
            r_embedding = self.testModel.getRelEmbedding(to_var(np.array([bad['r']], dtype=np.int64))).detach().numpy()
            t_embedding = self.testModel.getEntEmbedding(to_var(np.array([bad['t']], dtype=np.int64))).detach().numpy()
            bad['h_norm_F'] = np.sqrt(np.sum(h_embedding * h_embedding))
            bad['r_norm_F'] = np.sqrt(np.sum(r_embedding * r_embedding))
            bad['t_norm_F'] = np.sqrt(np.sum(t_embedding * t_embedding))
            bad_list.append(bad)

        
        bad_list.sort(key = lambda k : (k.get('h', 0), k.get('r', 0), k.get('t', 0)))
        self.writeBad('/Users/zhangjiatao/Documents/EnhanceKGEmbedding/OpenKE-PyTorch/result.txt', bad_list)
        self.DataReader.test_link_prediction()

    def writeBad(self, file, bad_list):

        with open(file, 'w') as f:

            f.write('h h_freq h_norm_F r r_freq r_norm_F t t_freq t_norm_F rank\n')
            for bad in bad_list:
                line = str(bad['h']) + ' ' + str(bad['h_freq']) + ' ' + str(bad['h_norm_F']) + ' ' + str(bad['r']) + ' ' + str(bad['r_freq']) + ' ' + str(bad['r_norm_F']) + ' ' + str(bad['t']) + ' ' + str(bad['t_freq']) + ' ' + str(bad['t_norm_F']) + ' ' + str(bad['rank']) + '\n'
                f.write(line)
        f.close()



    def test(self):
        if self.test_link:
            self.analysis_link_prediction()
        if self.test_triple:
            self.triple_classification()
