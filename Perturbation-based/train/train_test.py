import pickle as pkl
import torch
import numpy as np
from tqdm import tqdm
import csv
import copy
from until import create_data_list
from gensim.models.word2vec import Word2Vec
from model import GraphGNN
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, confusion_matrix, recall_score

word2vec = Word2Vec.load('data/w2v.model').wv.vectors

lr = 0.001
epochs = 100
clip_max = 2
batch_size = 128
max_patience = 5
num_class = 2
in_features_dim = 58  # 单词嵌入维度50， 结点语义嵌入50，结点类型维度8.结点嵌入维度58
out_features_dim = 100  # 图神经网络训练后结点向量维度大小

with open('data/pro_data/train.pkl', 'rb') as f:
    train_adjs, train_init_feas, train_labels = pkl.load(f)
    f.close()

with open('data/pro_data/test.pkl', 'rb') as f:
    test_adjs, test_init_feas, test_labels, _ = pkl.load(f)
    f.close()

train_set = create_data_list(train_adjs, train_init_feas, train_labels)
test_set = create_data_list(test_adjs, test_init_feas, test_labels)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

# model_types = ['gcn', 'gat', 'ggnn', 'gin']
model_types = ['gat']

# 每种训练十个模型
for m_type in model_types:
    for i in range(3, 10):
        model = GraphGNN(in_features_dim, out_features_dim, num_class, word2vec, m_type)
        # model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        patience_counter = 0

        best_f1 = 0.0
        best_acc = 0.0
        best_pre = 0.0
        best_recall = 0.0
        best_fnr = 0.0
        best_fpr = 0.0

        best_model_checkpoints = {}

        correct_idx = []

        save_dir = 'result' + '/' + m_type + '/'

        for epoch in range(epochs):
            model.train()
            for data in tqdm(train_loader):
                # data.cuda()
                optimizer.zero_grad()

                out = model(data.x, data.edge_index, data.batch)
                loss = criterion(out, data.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                test_data = next(iter(test_loader))
                # test_data.cuda()

                out = model(test_data.x, test_data.edge_index, test_data.batch)
                predictions = out.argmax(dim=1).detach().cpu().numpy().tolist()
                targets = test_data.y.detach().cpu().numpy().tolist()

                f1 = f1_score(targets, predictions)

                print(f"Epoch: {epoch}, test_f1: {f1:.4f}")
                if f1 > best_f1:

                    print("val improved")
                    best_f1 = f1
                    best_acc = accuracy_score(targets, predictions)
                    best_pre = precision_score(targets, predictions)
                    best_recall = recall_score(targets, predictions)
                    con_m = confusion_matrix(targets, predictions)
                    tn, fp, fn, tp = con_m.ravel()
                    best_fpr = fp / (fp + tn)
                    best_fnr = fn / (tp + fn)

                    patience_counter = 0
                    best_model_checkpoints = {'model_state_dict': copy.deepcopy(model.state_dict())}

                    correct_idx = []
                    for idx in range(len(predictions)):
                        if predictions[idx] == targets[idx]:
                            correct_idx.append(idx)
                else:
                    patience_counter += 1
                if patience_counter == max_patience:
                    break

        # record
        with open(save_dir + '{}_prediction_result.csv'.format(m_type), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([best_pre, best_acc, best_recall, best_f1, best_fpr, best_fnr])
            f.close()

        torch.save(best_model_checkpoints, save_dir + '{}_best_model_{}'.format(m_type, i))
        np.save(save_dir + '{}_correct_idx_{}.npy'.format(m_type, i), correct_idx)
        del model


