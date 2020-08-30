"""
Implantation of three state-of-the-art static embedding algorithms: Node2Vec, Graph Factorization and HOPE.
Implementations where taken from GEM package.
"""

import networkx as nx
import numpy as np
from time import time
# import scipy.sparse.linalg as lg
from node2vec import Node2Vec
import os
from scipy.sparse import identity
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import svds


class StaticGraphEmbedding:
    def __init__(self, d, method_name, graph):
        """
        Initialize the Embedding class
        :param d: dimension of embedding
        """
        self._d = d
        self._method_name = method_name
        self._graph = graph

    @staticmethod
    def get_method_name(self):
        """
        Returns the name for the embedding method
        :param self:
        :return: The name of embedding
        """
        return self._method_name

    def learn_embedding(self):
        """
        Learning the graph embedding from the adjacency matrix.
        :param graph: the graph to embed in networkx DiGraph format
        :return:
        """
        pass

    @staticmethod
    def get_embedding(self):
        """
        Returns the learnt embedding
        :return: A numpy array of size #nodes * d
        """
        pass


class GraphFactorization(StaticGraphEmbedding):
    """
    Graph Factorization factorizes the adjacency matrix with regularization.
    Args: hyper_dict (object): Hyper parameters.
    """

    def __init__(self, params, method_name, graph):
        super(GraphFactorization, self).__init__(params["dimension"], method_name, graph)
        """
        Initialize the GraphFactorization class
        Args: params:
            d: dimension of the embedding
            eta: learning rate of sgd
            regu: regularization coefficient of magnitude of weights
            max_iter: max iterations in sgd
            print_step: #iterations to log the prgoress (step%print_step)
        """
        self._eta = params["eta"]
        self._regu = params["regularization"]
        self._max_iter = params["max_iter"]
        self._print_step = params["print_step"]
        self._X = np.zeros(shape=(len(list(self._graph.nodes())), self._d))

    def get_f_value(self):
        """
        Get the value of f- the optimization function
        """
        nodes = list(self._graph.nodes())
        new_names = list(np.arange(0, len(nodes)))
        mapping = {}
        for i in new_names:
            mapping.update({nodes[i]: str(i)})
        H = nx.relabel.relabel_nodes(self._graph, mapping)
        f1 = 0
        for i, j, w in H.edges(data='weight', default=1):
            f1 += (w - np.dot(self._X[int(i), :], self._X[int(j), :])) ** 2
        f2 = self._regu * (np.linalg.norm(self._X) ** 2)
        return H, [f1, f2, f1 + f2]

    def learn_embedding(self):
        """
        Apply graph factorization embedding
        """
        t1 = time()
        node_num = len(list(self._graph.nodes()))
        self._X = 0.01 * np.random.randn(node_num, self._d)
        for iter_id in range(self._max_iter):
            my_f = self.get_f_value()
            count = 0
            if not iter_id % self._print_step:
                H, [f1, f2, f] = self.get_f_value()
                print('\t\tIter id: %d, Objective: %g, f1: %g, f2: %g' % (
                    iter_id,
                    f,
                    f1,
                    f2
                ))
            for i, j, w in H.edges(data='weight', default=1):
                if j <= i:
                    continue
                term1 = -(w - np.dot(self._X[int(i), :], self._X[int(j), :])) * self._X[int(j), :]
                term2 = self._regu * self._X[int(i), :]
                delPhi = term1 + term2
                self._X[int(i), :] -= self._eta * delPhi
            if count > 30:
                break
        t2 = time()
        projections = {}
        nodes = list(self._graph.nodes())
        new_nodes = list(H.nodes())
        for j in range(len(nodes)):
            projections.update({nodes[j]: self._X[int(new_nodes[j]), :]})
        # X is the embedding matrix and projections are the embedding dictionary
        return self._X, (t2 - t1), projections

    def get_embedding(self):
        return self._X


class HOPE(StaticGraphEmbedding):
    def __init__(self, params, method_name, graph):
        super(HOPE, self).__init__(params["dimension"], method_name, graph)
        """
        Initialize the HOPE class
        Args:
            d: dimension of the embedding
            beta: higher order coefficient
        """
        self._beta = params["beta"]

    def learn_embedding(self):
        """
        Apply HOPE embedding
        """
        A = nx.to_scipy_sparse_matrix(self._graph, format='csc')
        I = identity(self._graph.number_of_nodes(), format='csc')
        M_g = I - - self._beta * A
        M_l = self._beta * A
        # A = nx.to_numpy_matrix(self._graph)
        # M_g = np.eye(len(self._graph.nodes())) - self._beta * A
        # M_l = self._beta * A
        # S = inv(M_g).dot(M_l)
        S = np.dot(inv(M_g), M_l)

        u, s, vt = svds(S, k=self._d // 2)
        X1 = np.dot(u, np.diag(np.sqrt(s)))
        X2 = np.dot(vt.T, np.diag(np.sqrt(s)))
        self._X = np.concatenate((X1, X2), axis=1)

        p_d_p_t = np.dot(u, np.dot(np.diag(s), vt))
        eig_err = np.linalg.norm(p_d_p_t - S)
        print('SVD error (low rank): %f' % eig_err)

        # create dictionary of nodes
        nodes = list(self._graph.nodes())
        projections = {}
        for i in range(len(nodes)):
            y = self._X[i]
            y = np.reshape(y, newshape=(1, self._d))
            projections.update({nodes[i]: y[0]})
        # X is the embedding matrix, S is the similarity, projections is the embedding dictionary
        return projections, S, self._X, X1, X2

    def get_embedding(self):
        return self._X


class NODE2VEC(StaticGraphEmbedding):
    """
    Nod2Vec Embedding using random walks
    """
    def __init__(self, params, method_name, graph):
        super(NODE2VEC, self).__init__(params["dimension"], method_name, graph)
        """
        parameters:
        "walk_length" - Length of each random walk
        "num_walks" - Number of random walks from each source nodes
        "workers" - How many times repeat this process
        """
        self._walk_length = params["walk_length"]
        self._num_walks = params["num_walks"]
        self._workers = params["workers"]

    def learn_embedding(self):
        """
        Apply Node2Vec embedding
        """
        node2vec = Node2Vec(self._graph, dimensions=self._d, walk_length=self._walk_length,
                            num_walks=self._num_walks, workers=self._workers)
        model = node2vec.fit()
        nodes = list(self._graph.nodes())
        self._my_dict = {}
        for node in nodes:
            self._my_dict.update({node: np.asarray(model.wv.get_vector(node))})
        self._X = np.zeros((len(nodes), self._d))
        for i in range(len(nodes)):
             self._X[i, :] = np.asarray(model.wv.get_vector(nodes[i]))
        # X is the embedding matrix and projections are the embedding dictionary
        return self._X, self._my_dict

    def get_embedding(self):
        return self._X, self._my_dict


def final(G, method_name, params):
    """
    Final function to apply state-of-the-art embedding methods
    :param G: Graph to embed
    :param method_name: state-of-the-art embedding algorithm
    :param params: Parameters dictionary according to the embedding method
    :return:
    """
    if method_name == "HOPE":
        t = time()
        embedding = HOPE(params, method_name, G)
        projections, S, _, X1, X2 = embedding.learn_embedding()
        X = embedding.get_embedding()
        elapsed_time = time() - t
        return X, projections, elapsed_time
    elif method_name == "node2vec":
        t = time()
        embedding = NODE2VEC(params, method_name, G)
        embedding.learn_embedding()
        X, projections = embedding.get_embedding()
        elapsed_time = time() - t
        return X, projections, elapsed_time
    elif method_name == "GF":
        t = time()
        embedding = GraphFactorization(params, method_name, G)
        _, _, projections = embedding.learn_embedding()
        X = embedding.get_embedding()
        elapsed_time = time() - t
        return X, projections, elapsed_time
    else:
        print("Method is not valid. Valid methods are: node2vec, hope, graph_factorization")
        return None, None, None


def main():
    params = {"dimension": 10, "beta": 0.01}
    method = "hope"
    datasets_path = os.path.join("..", "datasets")
    G = nx.read_edgelist(os.path.join(datasets_path, "Email_Eu.txt"), create_using=nx.DiGraph(), delimiter=',')
    X, projections = final(G, method, params)


# main()