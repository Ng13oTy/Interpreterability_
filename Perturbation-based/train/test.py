import pickle as pkl
import torch
from tqdm import tqdm
import csv
import numpy as np
import copy
from until import create_data_list
from gensim.models.word2vec import Word2Vec
from model import GraphGNN
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, confusion_matrix, recall_score


a = np.load('result/gin/gin_correct_idx_0.npy')
print(11)
# a = target = torch.empty(3).random_(2)
# a = torch.nn.functional.one_hot(torch.tensor(0, dtype=torch.long).cuda(),
#                                 num_classes=8).float()

# lr = 0.001
# epochs = 100
# clip_max = 2
# batch_size = 128
# max_patience = 5
# num_class = 2
# in_features_dim = 58  # 单词嵌入维度50， 结点语义嵌入50，结点类型维度8.结点嵌入维度58
# out_features_dim = 100  # 图神经网络训练后结点向量维度大小
#
#
# with open('data/pro_data/test.pkl', 'rb') as f:
#     test_adjs, test_init_feas, test_labels, _ = pkl.load(f)
#     f.close()
#
#
# test_set = create_data_list(test_adjs, test_init_feas, test_labels)
#
# test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
#
# w2v = Word2Vec.load('data/w2v.model')
#
# # model_types = ['gcn', 'gat', 'ggnn', 'gin']
# m_type = 'gin'
# model = GraphGNN(in_features_dim, out_features_dim, num_class, w2v.wv.vectors, m_type)
# checkpoints = torch.load('result/gin/best_model_0')
# model.load_state_dict(checkpoints['model_state_dict'])
# model.cuda()
# model.eval()
#
# with torch.no_grad():
#     test_data = next(iter(test_loader))
#     test_data.cuda()
#     out = model(test_data.x, test_data.edge_index, test_data.batch)
#     predictions = out.argmax(dim=1).detach().cpu().numpy().tolist()
#     targets = test_data.y.detach().cpu().numpy().tolist()
#     f1 = f1_score(targets, predictions)
#     best_acc = accuracy_score(targets, predictions)
#     best_pre = precision_score(targets, predictions)
#     best_recall = recall_score(targets, predictions)
#     con_m = confusion_matrix(targets, predictions)
#     tn, fp, fn, tp = con_m.ravel()
#     best_fpr = fp / (fp + tn)
#     best_fnr = fn / (tp + fn)
#     print(best_pre, best_acc, best_recall, f1, best_fpr, best_fnr)
