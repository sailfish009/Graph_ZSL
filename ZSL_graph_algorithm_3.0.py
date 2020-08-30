from node2vec import Node2Vec
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import os
import argparse


class GraphImporter(object):
    def __init__(self, data_name):
        self.data_name = data_name

    def import_imdb_multi_graph(self):
        path = os.path.join(self.data_name, 'IMDb_multi_graph.gpickle')
        if os.path.exists(path):
            multi_gnx = nx.read_gpickle(path)
        else:
            from IMDb_data_preparation import main
            multi_gnx = main()
            nx.write_gpickle(multi_gnx, path)
        return multi_gnx

    def import_graph(self):
        graph = nx.MultiGraph()
        data_path = self.data_name + '.txt'
        path = os.path.join(self.data_name, data_path)
        with open(path, 'r') as f:
            for line in f:
                items = line.strip().split()
                att1 = str(items[0][0])
                att2 = str(items[1][0])
                graph.add_node(items[0], key=att1)
                graph.add_node(items[1], key=att2)
                sort_att = np.array([att1, att2])
                sort_att = sorted(sort_att)
                graph.add_edge(items[0], items[1], key=str(sort_att[0]) + str(sort_att[1]))
        return graph


class EmbeddingCreator(object):
    def __init__(self, data_name=None, graph=None):
        self.data_name = data_name
        self.graph = graph

    def create_node2vec_embeddings(self):
        path1 = os.path.join(self.data_name, 'Node2Vec_embedding.pickle')
        path2 = os.path.join(self.data_name, 'Node2Vec_embedding.csv')
        if os.path.exists(path1):
            with open(path1, 'rb') as handle:
                dict_embeddings = pickle.load(handle)
        elif os.path.exists(path2):
            embedding_df = pd.read_csv(path2)
            dict_embeddings = embedding_df.to_dict(orient='list')
            with open(path2, 'wb') as handle:
                pickle.dump(dict_embeddings, handle, protocol=3)
        else:
            node2vec = Node2Vec(self.graph, dimensions=16, walk_length=30, num_walks=200, workers=1)
            model = node2vec.fit()
            nodes = list(self.graph.nodes())
            dict_embeddings = {}
            for i in range(len(nodes)):
                dict_embeddings.update({nodes[i]: np.asarray(model.wv.get_vector(nodes[i]))})
            with open(path1, 'wb') as handle:
                pickle.dump(dict_embeddings, handle, protocol=3)
        return dict_embeddings

    def create_event2vec_embeddings(self):
        data_path = self.data_name + '_e2v_embeddings.txt'
        path = os.path.join(self.data_name, data_path)
        cond = 0
        dict_embeddings = {}
        with open(path, 'r') as f:
            for line in f:
                if cond == 1:
                    items = line.strip().split()
                    dict_embeddings[items[0]] = items[1:]
                cond = 1
        return dict_embeddings



"""
Link prediction task for evaluation, as explained in the pdf file. Initialize the methods you want to compare in
static_embeddings.py file, full explanation in their. if initial_method == "hope" and method == "directed_node2vec",
link prediction will be applied on these 2 embedding methods and will compare between them. Each combination is legal.
"""

try:
    import cPickle as pickle
except:
    import pickle
from numpy import linalg as LA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import model_selection as sk_ms
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier as oneVr
from sklearn.linear_model import LogisticRegression as lr
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

# from state_of_the_art.state_of_the_art_embedding import *


# for plots that will come later
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 14


def read_file(X, G):
    """
    Read a txt file of embedding and get the embedding
    :param X: The file after np.loadtxt
    :param G: The graph
    :return: A dictionary where keys==nodes and values==emmbedings
    """
    nodes = list(G.nodes())
    my_dict = {}
    for i in range(len(nodes)):
        my_dict.update({nodes[i]: X[i]})
    return my_dict


def make_true_edges(G, data_name):
    """
    Randomly choose a fixed number of existing edges
    :param edges: The graph's edges
    :param K: Fixed number of edges to choose
    :return: A list of K true edges
    """
    data_path = data_name + '_true_edges.pickle'
    if os.path.exists(os.path.join(data_name, data_path)):
        with open(os.path.join(data_name, data_path), 'rb') as handle:
            true_edges = pickle.load(handle)
    else:
        nodes = list(G.nodes)
        true_edges = []
        for node in nodes:
            info = G._adj[node]
            neighs = list(info.keys())
            for neigh in neighs:
                if info[neigh][0]['key'] == 'labels_edges':
                    true_edges.append([node, neigh])
        try:
            with open(os.path.join(data_name, data_path), 'wb') as handle:
                pickle.dump(true_edges, handle, protocol=3)
        except:
            pass
    return true_edges


def choose_true_edges(true_edges, times, K):
    if K > len(true_edges):
        K = len(true_edges)
    relevant_true_edges = []
    for j in range(times):
        relevant_true_edges_j = []
        indexes = np.arange(j * K, (j + 1) * K)
        for i in indexes:
            relevant_true_edges_j.append([true_edges[i][0], true_edges[i][1]])
        relevant_true_edges.append(relevant_true_edges_j)
    return relevant_true_edges


def true_edges_classes_ordered(true_edges, data_name):
    ordered_true_edges = []
    dict_class_label_edge = {}
    for edge in true_edges:
        if edge[0][0] == 'c':
            label = edge[0]
        else:
            label = edge[1]
        if dict_class_label_edge.get(label) is not None:
            edges = dict_class_label_edge[label]
            edges.append(edge)
            dict_class_label_edge[label] = edges
        else:
            dict_class_label_edge.update({label: [edge]})
    classes = list(dict_class_label_edge.keys())
    for c in classes:
        ordered_true_edges.append(c)
    return dict_class_label_edge, ordered_true_edges


def make_false_edges(data_name, dict_class_label_edge, false_per_true):
    """
    Randomly choose a fixed number of non-existing edges
    :param G: Our graph
    :param K: Fixed number of edges to choose
    :return: A list of K false edges
    """
    data_path = data_name + '_false_edges_balanced_{}.pickle'.format(false_per_true)
    if os.path.exists(os.path.join(data_name, data_path)):
        with open(os.path.join(data_name, data_path), 'rb') as handle:
            false_edges = pickle.load(handle)
        # with open('pkl_e2v/dblp_non_edges_old.pickle', 'wb') as handle:
        #     pickle.dump(non_edges, handle, protocol=3)
    else:
        false_edges = {}
        # path = os.path.join(data_name, 'final_labels_data.pkl')
        # if os.path.exists(path):
        #     final_labels_data = pd.read_pickle(path)
        #     labels = set(final_labels_data['name_label'])
        labels = list(dict_class_label_edge.keys())
        false_labels = []
        for label in labels:
            for edge in dict_class_label_edge[label]:
                if edge[0][0] == 'c':
                    label = edge[0]
                    movie = edge[1]
                else:
                    label = edge[1]
                    movie = edge[0]
                if len(false_labels) < false_per_true+1:
                    false_labels = list(set(labels) - set(label))
                else:
                    false_labels = list(set(false_labels)-set(label))
                indexes = random.sample(range(1, len(false_labels)), false_per_true)
                for i, index in enumerate(indexes):
                    if false_edges.get(label) is None:
                        false_edges[label] = [[movie, false_labels[index]]]
                    else:
                        edges = false_edges[label]
                        edges.append([movie, false_labels[index]])
                        false_edges[label] = edges
                false_labels = list(np.delete(np.array(false_labels), indexes))
        try:
            with open(os.path.join(data_name, data_path), 'wb') as handle:
                pickle.dump(false_edges, handle, protocol=3)
        except:
            pass
    return false_edges


def choose_false_edges(false_edges, times, K):
    relevant_false_edges = []
    for j in range(times):
        relevant_false_edges_j = []
        indexes = np.arange(j * K, (j+1) * K)
        for i in indexes:
            relevant_false_edges_j.append([false_edges[i][0], false_edges[i][1]])
        relevant_false_edges.append(relevant_false_edges_j)
    return relevant_false_edges
# def choose_true_edges(G, K, times):
#     """
#     Randomly choose a fixed number of existing edges
#     :param edges: The graph's edges
#     :param K: Fixed number of edges to choose
#     :return: A list of K true edges
#     """
#     edges = list(G.edges)
#     random_ind = random.sample(range(1, len(edges)), times * K)
#     true_edges = []
#     for j in range(times):
#         true_edges_j = []
#         indexes = np.arange(j * K, (j + 1) * K)
#         for i in indexes:
#             true_edges_j.append([edges[random_ind[i]][0], edges[random_ind[i]][1]])
#         true_edges.append(true_edges_j)
#     return true_edges

# def choose_false_edges(G, K, data_name, times, in_prob):
#     """
#     Randomly choose a fixed number of non-existing edges
#     :param G: Our graph
#     :param K: Fixed number of edges to choose
#     :return: A list of K false edges
#     """
#     data_path = data_name + '_non_edges.pickle'
#     if os.path.exists(os.path.join(data_name, data_path)):
#         with open(os.path.join(data_name, data_path), 'rb') as handle:
#             non_edges = pickle.load(handle)
#         # with open('pkl_e2v/dblp_non_edges_old.pickle', 'wb') as handle:
#         #     pickle.dump(non_edges, handle, protocol=3)
#     else:
#         if in_prob is True:
#             # times = int(1/(1-ratio)) * K
#             relevant_edges = list(G.edges)
#             false_edges = []
#             for time in range(times):
#                 is_edge = True
#                 while is_edge:
#                     indexes = random.sample(range(1, len(relevant_edges)), 2)
#                     node_1 = relevant_edges[indexes[0]][0]
#                     node_2 = relevant_edges[indexes[1]][1]
#                     if node_2 != relevant_edges[indexes[0]][1]:
#                         is_edge = False
#                         false_edges.append([node_1, node_2])
#             non_edges = false_edges
#         else:
#             non_edges = list(nx.non_edges(G))
#             false_edges = []
#             indexes = random.sample(range(1, len(non_edges)), times * K)
#             for j in indexes:
#                 false_edges.append([non_edges[j][0], non_edges[j][1]])
#         try:
#             with open(os.path.join(data_name, data_path), 'wb') as handle:
#                 pickle.dump(false_edges, handle, protocol=3)
#         except:
#             pass
#     # indexes = random.sample(range(1, len(non_edges)), K)
#     # false_edges = []
#     # for i in indexes:
#     #     false_edges.append([non_edges[i][0], non_edges[i][1]])
#     false_edges = []
#     for j in range(times):
#         false_edges_j = []
#         indexes = np.arange(j * K, (j+1) * K)
#         for i in indexes:
#             false_edges_j.append([non_edges[i][0], non_edges[i][1]])
#         false_edges.append(false_edges_j)
#     return false_edges


def calculate_classifier_value(dict_projections, true_edges, false_edges, K, norma):
    """
    Create X and Y for Logistic Regression Classifier.
    :param dict_projections: A dictionary of all nodes emnbeddings, where keys==nodes and values==embeddings
    :param true_edges: A list of K false edges
    :param false_edges: A list of K false edges
    :param K: Fixed number of edges to choose
    :return: X - The feature matrix for logistic regression classifier. Its size is 2K,1 and the the i'th row is the
                norm score calculated for each edge, as explained in the attached pdf file.
            Y - The edges labels, 0 for true, 1 for false
    """
    X = np.zeros(shape=(len(true_edges)+len(false_edges), 1))
    Y = np.zeros(shape=(len(true_edges)+len(false_edges), 4)).astype(int).astype(str)
    my_dict = {}
    count = 0
    for edge in true_edges:
        embd1 = np.array(dict_projections[edge[0]]).astype(float)
        embd2 = np.array(dict_projections[edge[1]]).astype(float)
        if norma == set('L1 Norm'):
            norm = LA.norm(np.subtract(embd1, embd2), 1)
        elif norma == set('L2 Norm'):
            norm = LA.norm(np.subtract(embd1, embd2), 1)
        elif norma == set('cosine'):
            norm = cosine_similarity(embd1.reshape(1, -1), embd2.reshape(1, -1))[0]
        X[count, 0] = norm
        Y[count, 2] = edge[0]
        Y[count, 3] = edge[1]
        Y[count, 0] = str(1)
        # my_dict.update({edge: [count, norm, int(0)]})
        count += 1
    for edge in false_edges:
        embd1 = np.array(dict_projections[edge[0]]).astype(float)
        embd2 = np.array(dict_projections[edge[1]]).astype(float)
        if norma == set('L1 Norm'):
            norm = LA.norm(np.subtract(embd1, embd2), 1)
        elif norma == set('L2 Norm'):
            norm = LA.norm(np.subtract(embd1, embd2), 1)
        elif norma == set('cosine'):
            norm = cosine_similarity(embd1.reshape(1, -1), embd2.reshape(1, -1))[0]
        X[count, 0] = norm
        Y[count, 2] = edge[0]
        Y[count, 3] = edge[1]
        Y[count, 1] = str(1)
        # my_dict.update({edge: [count, norm, int(1)]})
        count += 1
    return my_dict, X, Y


class TopKRanker(oneVr):
    """
    Linear regression with one-vs-rest classifier
    """

    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = super(TopKRanker, self).predict_proba(X)
        # probs = np.asarray()
        prediction = np.zeros((X.shape[0], self.classes_.shape[0]))
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-int(k):]].tolist()
            for label in labels:
                prediction[i, int(label)] = 1
        return prediction, probs


def evaluate_edge_classification(prediction, Y_test):
    """
    Predictions of nodes' labels.
    :param X: The features' graph- norm
    :param Y: The edges labels- 0 for true, 1 for false
    :param test_ratio: To determine how to split the data into train and test
    :return: Scores- F1-macro, F1-micro accuracy and auc
    """
    accuracy = accuracy_score(Y_test, prediction)
    # micro = f1_score(Y_test, prediction, average='micro')
    # macro = f1_score(Y_test, prediction, average='macro')
    # auc = roc_auc_score(Y_test, prediction)
    precision = precision_score(Y_test, prediction, average='weighted')
    return accuracy, precision


def train_edge_classification(X_train, Y_train):
    """
    Predictions of nodes' labels.
    :param X: The features' graph- norm
    :param Y: The edges labels- 0 for true, 1 for false
    :param test_ratio: To determine how to split the data into train and test
    :return: Scores- F1-macro, F1-micro accuracy and auc
    """
    classif2 = TopKRanker(lr())
    classif2.fit(X_train, Y_train)
    return classif2


def predict_edge_classification(classif2, X_test, Y_test):
    # try:
    #     top_k_list = list(Y_test.toarray().sum(axis=1))
    # except:
    #     top_k_list = list(Y_test.sum(axis=1))
    top_k_list = list(np.ones(len(X_test)).astype(int))
    prediction, probs = classif2.predict(X_test, top_k_list)
    return prediction, probs


def exp_lp(X, Y, test_ratio_arr, rounds):
    """
    The final node classification task as explained in our git.
    :param X: The features' graph- norm
    :param Y: The edges labels- 0 for true, 1 for false
    :param test_ratio_arr: To determine how to split the data into train and test. This an array
                with multiple options of how to split.
    :param rounds: How many times we're doing the mission. Scores will be the average.
    :return: Scores for all splits and all splits- F1-micro, F1-macro accuracy and auc
    """
    micro = [None] * rounds
    macro = [None] * rounds
    acc = [None] * rounds
    auc = [None] * rounds

    for round_id in range(rounds):
        micro_round = [None] * len(test_ratio_arr)
        macro_round = [None] * len(test_ratio_arr)
        acc_round = [None] * len(test_ratio_arr)
        auc_round = [None] * len(test_ratio_arr)
        for i, test_ratio in enumerate(test_ratio_arr):
            micro_round[i], macro_round[i], acc_round[i], auc_round[i] = predict_edge_classification(X, Y, test_ratio)

        micro[round_id] = micro_round
        macro[round_id] = macro_round
        acc[round_id] = acc_round
        auc[round_id] = auc_round

    micro = np.asarray(micro)
    macro = np.asarray(macro)
    acc = np.asarray(acc)
    auc = np.asarray(auc)

    return micro, macro, acc, auc


# def compute_precision_curve(Y, Y_test, true_digraph, k):
#     precision_scores = []
#     delta_factors = []
#     correct_edge = 0
#     for i in range(k):
#         if true_digraph.has_edge(sorted_edges[i][0], sorted_edges[i][1]):
#             correct_edge += 1
#             delta_factors.append(1.0)
#         else:
#             delta_factors.append(0.0)
#         precision_scores.append(1.0 * correct_edge / (i + 1))
#     return precision_scores, delta_factors


def calculate_avg_score(score, rounds):
    """
    Given the lists of scores for every round of every split, calculate the average score of every split.
    :param score: F1-micro / F1-macro / Accuracy / Auc
    :param rounds: How many times the experiment has been applied for each split.
    :return: Average score for every split
    """
    all_avg_scores = []
    for i in range(score.shape[1]):
        avg_score = (np.sum(score[:, i])) / rounds
        all_avg_scores.append(avg_score)
    return all_avg_scores


def calculate_all_avg_scores(micro, macro, acc, auc, rounds):
    """
    For all scores calculate the average score for every split. The function returns list for every
    score type- 1 for cheap node2vec and 2 for regular node2vec.
    """
    all_avg_micro = calculate_avg_score(micro, rounds)
    all_avg_macro = calculate_avg_score(macro, rounds)
    all_avg_acc = calculate_avg_score(acc, rounds)
    all_avg_auc = calculate_avg_score(auc, rounds)
    return all_avg_micro, all_avg_macro, all_avg_acc, all_avg_auc


def do_graph_split(avg_score1, avg_score2, test_ratio_arr, top, bottom, score, i):
    """
    Plot a graph of the score as a function of the test split value.
    :param avg_score1: list of average scores for every test ratio, 1 for cheap node2vec.
    :param avg_score2: list of average scores for every test ratio, 2 for regular node2vec.
    :param test_ratio_arr: list of the splits' values
    :param top: top limit of y axis
    :param bottom: bottom limit of y axis
    :param score: type of score (F1-micro / F1-macro / accuracy/ auc)
    :return: plot as explained above
    """
    fig = plt.figure(i)
    plt.plot(test_ratio_arr, avg_score1, '-ok', color='blue')
    plt.plot(test_ratio_arr, avg_score2, '-ok', color='red')
    plt.legend(['cheap node2vec', 'regular node2vec'], loc='upper left')
    plt.ylim(bottom=bottom, top=top)
    # plt.title("Pubmed2 dataset")
    plt.xlabel("test ratio")
    plt.ylabel(score)
    return fig


def split_vs_score(avg_micro1, avg_macro1, avg_micro2, avg_macro2, avg_acc1, avg_acc2, avg_auc1, avg_auc2,
                   test_ratio_arr):
    """
    For every type of score plot the graph as explained above.
    """
    # you can change borders as you like
    fig1 = do_graph_split(avg_micro1, avg_micro2, test_ratio_arr, 1, 0, "micro-F1 score", 1)
    fig2 = do_graph_split(avg_macro1, avg_macro2, test_ratio_arr, 1, 0, "macro-F1 score", 2)
    fig3 = do_graph_split(avg_acc1, avg_acc2, test_ratio_arr, 1, 0, "accuracy", 3)
    fig4 = do_graph_split(avg_auc1, avg_auc2, test_ratio_arr, 1, 0, "auc", 4)
    return fig1, fig2, fig3, fig4


def edges_to_predict(multi_graph):
    edges = list(multi_graph.edges.data(keys=True))
    start = False
    relevant_edges = None
    for edge in edges:
        if edge[3]['key'] == 'labels_edges':
            if edge[0][0] == 't':
                node_1 = edge[0]
                node_2 = edge[1]
            else:
                node_1 = edge[1]
                node_2 = edge[0]
            if start:
                relevant_edges = np.append(relevant_edges, np.array([[node_1, node_2]]), axis=0)
            else:
                relevant_edges = np.array([[node_1, node_2]])
            start = True
    return relevant_edges


#
# def choose_true_edges(relevant_edges, K):
#     """
#     Randomly choose a fixed number of existing edges
#     :param edges: The graph's edges
#     :param K: Fixed number of edges to choose
#     :return: A list of K true edges
#     """
#     indexes = random.sample(range(1, len(relevant_edges)), K)
#     true_edges = []
#     for i in indexes:
#         true_edges.append([relevant_edges[i][0], relevant_edges[i][1]])
#     return true_edges
#
#
# def choose_false_edges(G, relevant_edges, K):
#     """
#     Randomly choose a fixed number of non-existing edges
#     :param G: Our graph
#     :param K: Fixed number of edges to choose
#     :return: A list of K false edges
#     """
#     times = 5 * K
#     false_edges = []
#     for time in range(times):
#         is_edge = True
#         while is_edge:
#             indexes = random.sample(range(1, len(relevant_edges)), 2)
#             node_1 = relevant_edges[indexes[0]][0]
#             node_2 = relevant_edges[indexes[1]][1]
#             if node_2 != relevant_edges[indexes[0]][1]:
#                 is_edge = False
#                 false_edges.append([node_1, node_2])
#     return false_edges
def compute_final_measures(true_edges, false_edges, dict_embeddings, ratio, number, times, norm):
    dict_measures = {'acc': {}, 'precision': {}}
    dict_class_measures = {}
    dict_class_x_train, dict_class_x_test, dict_class_y_train, dict_class_y_test = {}, {}, {}, {}
    X_train_all, Y_train_all = np.array([]), np.array([])
    classes = list(true_edges.keys())
    seen_classes = classes[:int(0.8*len(classes))]
    unseen_classes = classes[int(0.8*len(classes)):]
    for j in range(len(ratio)):
        for c in seen_classes:
            my_dict, X, Y = calculate_classifier_value(dict_embeddings, true_edges[c], false_edges[c], number, norm)
            X_train, X_test, Y_train, Y_test = sk_ms.train_test_split(X, Y, test_size=1 - ratio[j])
            dict_class_x_train.update({c: X_train})
            dict_class_x_test.update({c: X_test})
            dict_class_y_train.update({c: Y_train})
            dict_class_y_test.update({c: Y_test})
            if len(X_train_all) > 0:
                X_train_all = np.concatenate((X_train_all, X_train), axis=0)
                Y_train_all = np.concatenate((Y_train_all, Y_train), axis=0)
            else:
                X_train_all = X_train
                Y_train_all = Y_train
        for c in unseen_classes:
            my_dict, X, Y = calculate_classifier_value(dict_embeddings, true_edges[c], false_edges[c], number, norm)
            dict_class_x_train.update({c: []})
            dict_class_x_test.update({c: X})
            dict_class_y_train.update({c: []})
            dict_class_y_test.update({c: Y})
        classif2 = train_edge_classification(np.array(X_train_all), np.array(Y_train_all))
        for c in classes:
            prediction, probs = predict_edge_classification(classif2, dict_class_x_test[c], dict_class_y_test[c])
            acc, precision = evaluate_edge_classification(prediction, dict_class_y_test[c])
            # dict_measures['micro'] = micro
            # dict_measures['macro'] = macro
            dict_measures['acc'] = acc
            # dict_measures['auc'] = auc
            dict_measures['precision'] = precision
            dict_class_measures[c] = dict_measures.copy()
    return dict_class_measures


def train(true_edges, false_edges, dict_embeddings, ratio, number, times, norma, data_name):
    path1 = os.path.join(data_name, 'train/classifier_n2v_cosine.pkl')
    path2 = os.path.join(data_name,'train/dict_c_t_n2v_cosine.pkl')
    if os.path.exists(path1) and os.path.exists(path2):
        with open(path1, 'rb') as handle:
            classif2 = pickle.load(handle)
        with open(path2, 'rb') as handle:
            dict_c_t = pickle.load(handle)
    else:
        classes = list(true_edges.keys())
        # path='train/X_train_all'
        for i, k in enumerate(sorted(true_edges, key=lambda k: len(true_edges[k]), reverse=True)):
            classes[i] = k
        num_classes = len(classes)
        dict_measures = {'acc': {}, 'precision': {}}
        dict_class_measures = {}
        dict_c_t = {}
        dict_class_x_train, dict_class_x_test, dict_class_y_train, dict_class_y_test = {}, {}, {}, {}
        X_train_all, Y_train_all = np.array([]), np.array([])
        seen_classes = classes[:int(0.8 * len(classes))]
        unseen_classes = classes[int(0.8 * len(classes)):]
        for j in range(len(ratio)):
            # if os.path.exists(os.path.join(data_name, path)):
            #     with open(os.path.join(data_name, path), 'rb') as handle:
            #         true_edges = pickle.load(handle)
            for c in seen_classes:
                dict_t_edge = {}
                _, X_true, Y_true = calculate_classifier_value(dict_embeddings, true_edges[c], [], number, norma)
                _, X_false, Y_false = calculate_classifier_value(dict_embeddings, [], false_edges[c], number, norma)
                # X , Y = np.concatenate((X_true,X_false), axis=0), np.concatenate((Y_true,Y_false), axis=0)
                X_train_true, X_test_true, Y_train_true, Y_test_true = sk_ms.train_test_split(X_true, Y_true,
                                                                                                    test_size=1 - ratio[j])
                X_train_false, X_test_false, Y_train_false, Y_test_false = sk_ms.train_test_split(X_false, Y_false,
                                                                                                    test_size=1 - ratio[j])
                X_train, X_test, Y_train, Y_test = np.concatenate((X_train_true, X_train_false), axis=0),\
                                                         np.concatenate((X_test_true, X_test_false), axis=0), \
                                                         np.concatenate((Y_train_true, Y_train_false), axis=0),\
                                                         np.concatenate((Y_test_true, Y_test_false), axis=0)
                true_edges_test_source = Y_test_true.T[2].reshape(-1, 1)
                true_edges_test_target = Y_test_true.T[3].reshape(-1, 1)
                Y_train = np.array([Y_train.T[0].reshape(-1, 1), Y_train.T[1].reshape(-1, 1)]).T.reshape(-1, 2).astype(int)
                true_edges_test = np.array([true_edges_test_source, true_edges_test_target]).T[0]
                for edge in true_edges_test:
                    if edge[0][0] == 't':
                        movie = edge[0]
                    else:
                        movie = edge[1]
                    dict_t_edge[movie] = set(edge)
                dict_c_t[c] = dict_t_edge.copy()
                dict_class_x_train.update({c: X_train})
                dict_class_x_test.update({c: X_test})
                dict_class_y_train.update({c: Y_train})
                dict_class_y_test.update({c: Y_test})
                if len(X_train_all) > 0:
                    X_train_all = np.concatenate((X_train_all, X_train), axis=0)
                    Y_train_all = np.concatenate((Y_train_all, Y_train), axis=0)
                else:
                    X_train_all = X_train
                    Y_train_all = Y_train
            for c in unseen_classes:
                dict_t_edge = {}
                _, X_true, Y_true = calculate_classifier_value(dict_embeddings, true_edges[c], [], number, norma)
                _, X_false, Y_false = calculate_classifier_value(dict_embeddings, [], false_edges[c], number, norma)
                X, Y = np.concatenate((X_true, X_false), axis=0), np.concatenate((Y_true, Y_false), axis=0)
                true_edges_test_source = Y_true.T[2].reshape(-1, 1)
                true_edges_test_target = Y_true.T[3].reshape(-1, 1)
                Y = np.array([Y.T[0].reshape(-1, 1), Y.T[1].reshape(-1, 1)]).T.reshape(-1, 2).astype(int)
                # Y = Y.T[0].reshape(-1, 1)
                # true_edges_test_source = X_true.T[1].reshape(-1, 1)
                # true_edges_test_target = X_true.T[2].reshape(-1, 1)
                true_edges_test = np.array([true_edges_test_source, true_edges_test_target]).T[0]
                for edge in true_edges_test:
                    if edge[0][0] == 't':
                        movie = edge[0]
                    else:
                        movie = edge[1]
                    dict_t_edge[movie] = edge
                dict_c_t[c] = dict_t_edge.copy()
                dict_class_x_train.update({c: []})
                dict_class_x_test.update({c: X})
                dict_class_y_train.update({c: []})
                dict_class_y_test.update({c: Y})
            shuff = np.c_[X_train_all.reshape(len(X_train_all), -1), Y_train_all.reshape(len(Y_train_all), -1)]
            np.random.shuffle(shuff)
            X_train_all = shuff.T[0].reshape(-1,1)
            Y_train_all = np.array([shuff.T[1].reshape(-1, 1), shuff.T[2].reshape(-1, 1)]).T.reshape(-1, 2).astype(int)
            classif2 = train_edge_classification(np.array(X_train_all), np.array(Y_train_all))
            with open(path1, 'wb') as fid:
                pickle.dump(classif2, fid)
            with open(path2, 'wb') as fid:
                pickle.dump(dict_c_t, fid)
    return classif2, dict_c_t

def evaluate(classif2, dict_c_t, true_edges, dict_embeddings, norma, data_name):
    # evaluate
    classes = list(true_edges.keys())
    pred_true = []
    pred = []
    # path='train/X_train_all'
    for i, k in enumerate(sorted(true_edges, key=lambda k: len(true_edges[k]), reverse=True)):
        classes[i] = k
    num_classes = len(classes)
    dict_measures = {'acc': {}, 'precision': {}}
    dict_class_measures = {}
    for c in classes:
        class_movies = list(dict_c_t[c].keys())
        count = 0
        for m in class_movies:
            edges = np.array([np.repeat(m, num_classes), classes]).T
            class_test = np.zeros(shape=(len(edges), 1))
            for i, edge in enumerate(edges):
                embd1 = np.array(dict_embeddings[edge[0]]).astype(float)
                embd2 = np.array(dict_embeddings[edge[1]]).astype(float)
                if norma == set('L1 Norm'):
                    norm = LA.norm(np.subtract(embd1, embd2), 1)
                elif norma == set('L2 Norm'):
                    norm = LA.norm(np.subtract(embd1, embd2), 1)
                elif norma == set('cosine'):
                    norm = cosine_similarity(embd1.reshape(1, -1), embd2.reshape(1, -1))[0]
                class_test[i, 0] = norm
            _, probs = predict_edge_classification(classif2, class_test, None)
            pred_index = np.argmax(probs.T[0])
            prediction = edges[pred_index]
            real_edge = list(dict_c_t[c][m])
            # dict_class_pred = dict.fromkeys(c, 0)
            # if prediction[0][0] == 'c':
            #     dict_class_pred[prediction[0]] += 1
            # else:
            #     dict_class_pred[prediction[1]] += 1
            pred_true.append(c)
            if prediction[0][0] == 'c':
                pred.append(prediction[0])
            else:
                pred.append(prediction[1])

            if prediction[0] == real_edge[0]:
                if prediction[1] == real_edge[1]:
                    count += 1
            elif prediction[1] == real_edge[0]:
                if prediction[0] == real_edge[1]:
                    count += 1
        acc = count / len(class_movies)
        dict_measures['acc'] = acc
        dict_class_measures[c] = dict_measures.copy()
    with open(os.path.join(data_name, 'dict_class_measures_n2v_cosine.pkl'), 'wb') as handle:
        pickle.dump(dict_class_measures, handle, protocol=3)
    # TODO dict class measures for every ratio
    return dict_class_measures, pred, pred_true

def confision_matrix_maker(dict_class_measures, pred, pred_true):
    conf_matrix = confusion_matrix(pred_true, pred, labels=list(dict_class_measures.keys()))
    true_count = 0
    count = 0
    for i in range(len(conf_matrix)):
        true_count += conf_matrix[i][i]
        for j in range(len(conf_matrix)):
            count += conf_matrix[i][j]
    print(f'acc_all: {true_count / count}')
    plt.figure(1)
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['axes.titlesize'] = 20
    mpl.rcParams['axes.labelsize'] = 16
    plt.title('Confusion Matrix, ZSL OUR_IMDB')
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")
    plt.imshow(conf_matrix, cmap='gist_gray', vmin=0, vmax=2)
    plt.colorbar()
    plt.savefig('our_imdb/plots/confiusion_matrix_n2v_cosine')

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams["font.family"] = "Times New Roman"


def plots_maker(dict_measures, ratio_arr, measure, data_name, number, norm):
    x_axis = np.array(ratio_arr)
    task = 'Link Prediction'
    bottom = 0.5
    top = 0.95
    keys = list(dict_measures.keys())
    plt.figure(figsize=(7, 6))
    for j in range(len(keys)):
        if 'event2vec' in keys[j]:
            color = 'red'
            marker = 'o'
            markersize = 8
            linestyle = 'solid'
            y_axis = dict_measures[keys[j]][measure]
        elif "node2vec" in keys[j]:
            color = 'green'
            marker = 's'
            markersize = 6
            linestyle = 'solid'
            y_axis = dict_measures[keys[j]][measure]
        plt.plot(x_axis, y_axis, marker=marker, linestyle=linestyle, markersize=markersize, color=color)
    plt.plot(x_axis, [0.58, 0.63, 0.74, 0.8, 0.83, 0.86, 0.88, 0.9, 0.93], marker='o', linestyle='dashed', markersize=8, color='red')
    plt.plot(x_axis, [0.53, 0.54, 0.58, 0.61, 0.64, 0.66, 0.67, 0.68, 0.81], marker='s', linestyle='dashed', markersize=6, color='green')
    keys = ['our_node2vec', 'our_event2vec', 'event2vec', 'node2vec']
    plt.ylim(bottom=bottom, top=top)
    plt.legend(keys, loc='best', ncol=3, fontsize='large')
    plt.title("{} Dataset \n {} Task - {} Score".format(data_name, task, measure))
    plt.xlabel("Percentage")
    plt.ylabel("{} ({})".format(measure, norm))
    plt.tight_layout()
    plt.savefig(os.path.join(data_name, "plots", "{} {} {} {} {}.png".format(data_name, task, measure, norm, number)))
    plt.show()


def main():
    """
    Main Function for link prediction task.
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='our_imdb')
    parser.add_argument('--norm', default='cosine')  # cosine / L2 Norm / L1 Norm
    args = parser.parse_args()
    norm = set(args.norm)
    all_micro = []
    all_macro = []
    all_acc = []
    all_auc = []
    number = 1000
    times = 2
    false_per_true = 10
    # ratio_arr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ratio_arr = [0.8]
    graph_maker = GraphImporter(args.data_name)
    graph = graph_maker.import_imdb_multi_graph()
    # nodes = graph.nodes()
    # indexes = np.linspace(0, len(nodes)-1, 5000)
    # indexes = indexes.astype(int)
    # relevant_nodes = np.array(nodes)[indexes]
    # graph = nx.subgraph(graph, relevant_nodes)
    embeddings_maker = EmbeddingCreator(args.data_name, graph)
    # dict_embeddings_event2vec = embeddings_maker.create_event2vec_embeddings()
    dict_embeddings_node2vec = embeddings_maker.create_node2vec_embeddings()
    # dict_event2vec_embeddings = embedding_model.create_event2vec_embeddings()
    # nodes = list(dict_event2vec_embeddings.keys())
    # relevant_edges = edges_to_predict(multi_graph)
    # true_edges = choose_true_edges(relevant_edges, number)
    # false_edges = choose_false_edges(multi_graph, relevant_edges, number)
    true_edges = make_true_edges(graph, args.data_name)
    dict_true_edges, true_edges = true_edges_classes_ordered(true_edges, args.data_name)
    dict_false_edges = make_false_edges(args.data_name, dict_true_edges, false_per_true)
    # dict_false_edges = choose_false_edges(false_edges, len(dict_true_edges), number)
    # dict_measures_event2vec = compute_final_measures(true_edges, false_edges, dict_embeddings_event2vec, ratio_arr, number, times, norm)
    classif, dict_c_t = train(dict_true_edges, dict_false_edges, dict_embeddings_node2vec, ratio_arr, number, times, norm, args.data_name)
    dict_class_measures_node2vec, pred, pred_true = evaluate(classif, dict_c_t, dict_true_edges, dict_embeddings_node2vec, norm, args.data_name)
    confision_matrix_maker(dict_class_measures_node2vec, pred, pred_true)
    with open(os.path.join('our_imdb', 'dict_class_measures_n2v_cosine.pkl'), 'rb') as handle:
        dict_class_measures = pickle.load(handle)
    keys = list(dict_class_measures.keys())
    count = 0
    for key in keys:
        count += dict_class_measures[key]['acc']
    avg_acc = count / len(keys)
    print(avg_acc)
    # dict_measures = {}
    # dict_measures['node2vec'] = dict_measures_node2vec
    # dict_measures['event2vec'] = dict_measures_event2vec
    # plots_maker(dict_measures, ratio_arr, 'AUC', args.data_name.upper(), number, args.norm)
    # print('avg acc e2v: ', dict_measures_event2vec['Accuracy'])
    # print('avg auc e2v: ', dict_measures_event2vec['AUC'])
    # print('avg micro e2v: ', dict_measures_event2vec['Micro-f1'])
    # print('avg macro e2v: ', dict_measures_event2vec['Macro-f1'])
    # print('std acc e2v: ', dict_measures_event2vec['std_acc'])
    # print('std auc e2v: ', dict_measures_event2vec['std_auc'])
    # print('std micro e2v: ', dict_measures_event2vec['std_micro'])
    # print('std macro e2v: ', dict_measures_event2vec['std_macro'])
    # print('avg acc n2v: ', dict_measures_node2vec['Accuracy'])
    # print('avg auc n2v: ', dict_measures_node2vec['AUC'])
    # print('avg micro n2v: ', dict_measures_node2vec['Micro-f1'])
    # print('avg macro n2v: ', dict_measures_node2vec['Macro-f1'])
    # print('std acc n2v: ', dict_measures_node2vec['std_acc'])
    # print('std auc n2v: ', dict_measures_node2vec['std_auc'])
    # print('std micro n2v: ', dict_measures_node2vec['std_micro'])
    # print('std macro n2v: ', dict_measures_node2vec['std_macro'])
    # dict_embeddings = embeddings_maker.create_node2vec_embeddings()
    # micro, macro, acc, auc = exp_lp(X, Y, ratio_arr, 3)
    # avg_micro, avg_macro, avg_acc, avg_auc = calculate_all_avg_scores(micro, macro, acc, auc, 3)
    # all_micro.append(avg_micro)
    # all_macro.append(avg_macro)
    # all_acc.append(avg_acc)
    # all_auc.append(avg_auc)
    # fig1, fig2, fig3, fig4 = split_vs_score(all_micro[0], all_macro[0], all_micro[1], all_macro[1], all_acc[0],
    #                                         all_acc[1], all_auc[0], all_auc[1], ratio_arr)
    # plt.show()


main()