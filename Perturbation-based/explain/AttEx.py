import torch
import pickle as pkl
import csv
from gensim.models.word2vec import Word2Vec
from train.model import GraphGNN
import numpy as np
from My_Explainer.AttExplainer import AttExplainer
from tqdm import tqdm

with open('../train/data/pro_data/test.pkl', 'rb') as f:
    explain_data = pkl.load(f)
    f.close()

graphs_ = explain_data[0]
feas = torch.tensor(explain_data[1])
labels = torch.tensor(explain_data[2])
graphs_ = [torch.tensor(g) for g in graphs_]
key_nodes = explain_data[3]

word2vec = Word2Vec.load('../train/data/w2v.model').wv.vectors
num_class = 2
in_features_dim = 58  # 单词嵌入维度50， 结点语义嵌入50，结点类型维度8.结点嵌入维度58
out_features_dim = 100  # 图神经网络训练后结点向量维度大小
model_types = ['gat']
for m_type in model_types:
    # 解释十个模型
    for i in range(3):
        # # 每种解释方法解释十次
        # for exp_time in range(3):
        print('current model:{}'.format(i))
        trained_model_path = '../train/result/{}/{}_best_model_{}'.format(m_type, m_type, i)
        model = GraphGNN(in_features_dim, out_features_dim, num_class, word2vec, m_type)
        checkpoints = torch.load(trained_model_path)
        model.load_state_dict(checkpoints['model_state_dict'])

        model.eval()

        explainer = AttExplainer('mean')
        correct_idx = np.load('../train/result/{}/{}_correct_idx_{}.npy'.format(m_type, m_type, i))


        fixed_p50, fixed_p30, fixed_3, fixed_1, fixed_total = 0, 0, 0, 0, 0
        b_total = 0
        b_sesk_p50, b_sesk_p30, b_sesk_3, b_sesk_1 = 0, 0, 0, 0
        b_se_p50, b_se_p30, b_se_3, b_se_1 = 0, 0, 0, 0
        b_sk_p50, b_sk_p30, b_sk_3, b_sk_1 = 0, 0, 0, 0
        b_avg_p50, b_avg_p30 = 0, 0

        for idx in tqdm(correct_idx):

            # 得到解释
            node_id_1 = []
            node_id_3 = []
            node_id_30 = []
            node_id_50 = []

            gt_label = labels[idx].argmax(dim=-1).unsqueeze(0).detach()
            edge_mask = explainer.explain(model, feas[idx].detach(), graphs_[idx].detach(), gt_label).tolist()

            sorted_edge_id = sorted(range(len(edge_mask)), key=lambda k: edge_mask[k], reverse=True)
            cur_graph = graphs_[idx].T.tolist()
            e_num = len(sorted_edge_id)

            node_id_1.extend(cur_graph[sorted_edge_id[1]])

            for top_k in range(3):
                node_id_3.extend(cur_graph[sorted_edge_id[top_k]])

            for top_k in range(int(e_num * 0.3)):
                node_id_30.extend(cur_graph[sorted_edge_id[top_k]])

            for top_k in range(int(e_num * 0.5)):
                node_id_50.extend(cur_graph[sorted_edge_id[top_k]])

            if gt_label.item():
                b_total += 1
                bad_source = key_nodes[idx][0]
                bad_sink = key_nodes[idx][1]

                if bad_source in node_id_50:
                    b_se_p50 += 1
                if bad_source in node_id_30:
                    b_se_p30 += 1
                if bad_source in node_id_3:
                    b_se_3 += 1
                if bad_source in node_id_1:
                    b_se_1 += 1

                if bad_sink in node_id_50:
                    b_sk_p50 += 1
                if bad_sink in node_id_30:
                    b_sk_p30 += 1
                if bad_sink in node_id_3:
                    b_sk_3 += 1
                if bad_sink in node_id_1:
                    b_sk_1 += 1

                if bad_source in node_id_50 or bad_sink in node_id_50:
                    b_sesk_p50 += 1
                if bad_source in node_id_30 or bad_sink in node_id_30:
                    b_sesk_p30 += 1
                if bad_source in node_id_3 or bad_sink in node_id_3:
                    b_sesk_3 += 1
                if bad_source in node_id_1 or bad_sink in node_id_1:
                    b_sesk_1 += 1

                if bad_source in node_id_50 and bad_sink in node_id_50:
                    b_avg_p50 += 1
                if bad_source in node_id_30 and bad_sink in node_id_30:
                    b_avg_p30 += 1
            else:
                fixed_total += 1
                fixed = key_nodes[idx][0]
                if fixed in node_id_50:
                    fixed_p50 += 1
                if fixed in node_id_30:
                    fixed_p30 += 1
                if fixed in node_id_3:
                    fixed_3 += 1
                if fixed in node_id_1:
                    fixed_1 += 1

        with open('result/{}/{}_best_model_{}_att.csv'.format(m_type, m_type, i),
                  'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['hit_type', 'hit50%', 'hit30%', 'hit3', 'hit1'])
            writer.writerow(['fixed', fixed_p50 / fixed_total, fixed_p30 / fixed_total, fixed_3 / fixed_total,
                             fixed_1 / fixed_total])
            writer.writerow(['b_se', b_se_p50 / b_total, b_se_p30 / b_total, b_se_3 / b_total, b_se_1 / b_total])
            writer.writerow(['b_sk', b_sk_p50 / b_total, b_sk_p30 / b_total, b_sk_3 / b_total, b_sk_1 / b_total])
            writer.writerow(
                ['b_sesk', b_sesk_p50 / b_total, b_sesk_p30 / b_total, b_sesk_3 / b_total, b_sesk_1 / b_total])
            writer.writerow(
                ['b_avg', b_avg_p50 / b_total, b_avg_p30 / b_total, '-', '-'])
            f.close()
