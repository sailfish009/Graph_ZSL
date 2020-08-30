"""
Implementation of Hope_Loss approach, as explained in the pdf file which is in our github page.
"""


import networkx as nx
import numpy as np
from time import time
import time
from numpy.linalg import inv
import heapq


def calculate_s(g2, beta):
    """
    Calculate S - similarity matrix using Katz Index.
    :param g2: Our graph
    :param beta: Hyperparameter
    :return: similarity matrix S
    """
    A = nx.to_numpy_matrix(g2)
    M_g = np.eye(A.shape[0]) - beta * A
    M_l = beta * A
    S = np.dot(np.linalg.inv(M_g), M_l)
    return S


def calculate_c_u(S, V, d):
    """
    As explained in the pdf file, find C to calculate U.
    :param S: Similarity matrix
    :param V: Target embeddings
    :param d: Embedding Dimension
    :return: C
    """
    m = int(d/2)
    # find c in the equation
    c = np.zeros(shape=(1, m))
    for l in range(m):
        sum1 = 0
        for j in range(V.shape[1]):
            sum1 += S[0, j]*V[l, j]
        c[0, l] = sum1
    c_t = np.transpose(c)
    return c_t


def calculate_c_v(S, U, d):
    """
    As explained in the pdf file, find C to calculate V.
    :param S: Similarity matrix
    :param U: Source embeddings
    :param d: Embedding Dimension
    :return: C
    """
    m = int(d / 2)
    # find c in the equation
    c = np.zeros(shape=(1, m))
    for l in range(m):
        sum1 = 0
        for i in range(U.shape[0]):
            sum1 += S[i, 0] * U[i, l]
        c[0, l] = sum1
    c_t = np.transpose(c)
    return c_t


def calculate_b_u(V, d):
    """
    As explained in the pdf file, find B to calculate U.
    :param V: Target embeddings
    :param d: Embedding Dimension
    :return: B
    """
    m = int(d / 2)
    B = np.zeros(shape=(m, m))
    for l in range(m):
        for k in range(m):
            sum1 = 0
            for j in range(V.shape[1]):
                sum1 += V[k, j] * V[l, j]
            B[l, k] = sum1
    return B


def calculate_b_v(U, d):
    """
    As explained in the pdf file, find B to calculate V.
    :param U: Source embeddings
    :param d: Embedding Dimension
    :return: B
    """
    m = int(d / 2)
    B = np.zeros(shape=(m, m))
    for l in range(m):
        for k in range(m):
            sum1 = 0
            for i in range(U.shape[0]):
                sum1 += U[i, k] * U[i, l]
            B[l, k] = sum1
    return B


def calculate_embedding_u(S, V, d):
    """
    Calculate Source embedding matrix, U
    :param S: Similarity matrix
    :param V: Target Embedding
    :param d: Embedding Dimension
    :return: U- source embedding
    """
    C_u = calculate_c_u(S, V, d)
    B_u = calculate_b_u(V, d)
    B_inv_u = inv(B_u)
    U = np.matmul(B_inv_u, C_u)
    U = U.T
    return U


def calculate_embedding_v(S, U, d):
    """
    Calculate Target embedding matrix, V
    :param S: Similarity matrix
    :param U: Source Embedding
    :param d: Embedding Dimension
    :return: V- Target embedding
    """
    C_v = calculate_c_v(S, U, d)
    B_v = calculate_b_v(U, d)
    B_inv_v = inv(B_v)
    V = np.matmul(B_inv_v, C_v)
    V = V.T
    return V


def hope_loss(node, S, U, V, dict_index, d):
    """
    Calculate the given node's embedding according to the formula that in the pfd file.
    :param node: Current node its embedding needs to be calculated
    :param S: Similarity matrix
    :param U: Source Embedding
    :param V: Target embedding matrix
    :param dict_index:
    :param d: Embedding Dimension
    :return: Final embedding of the node
    """
    S_j = S[dict_index[node], :]
    S_i = S[:, dict_index[node]]
    t = time.time()
    u = calculate_embedding_u(S_j, V, d)
    v = calculate_embedding_v(S_i, U, d)
    embd = np.concatenate((u, v), axis=1)
    t = time.time() - t
    print(t)
    return embd


def create_dict_index(g):
    """
    Create a dictionary where keys==nodes and values==their indices
    :param g:
    :return:
    """
    nodes = list(g.nodes())
    dict_index = {}
    for i in range(len(nodes)):
        dict_index.update({nodes[i]: i})
    return dict_index


def one_iteration(dict_index, S, U, V, dict_enode_proj, dict_node_enode, dict_node_node, dict_enode_enode, set_n_e, current_node, dim):
    """
    a function that does one iteration over a given batch
    """
    condition = 1
    # get the neighbors in projection of node i
    embd_neigh = dict_node_enode[current_node]
    # the final projection of the node
    # final_proj = calculate_projection(embd_neigh, dict_enode_proj, dim, dict_enode_enode)
    final_proj = hope_loss(current_node, S, U, V, dict_index, dim)
    # add the node and its projection to the dict of projections
    dict_enode_proj.update({current_node: final_proj})
    # add our node to the dict of proj to proj and delete it from node_enode because now it's in the projection
    dict_enode_enode.update({current_node: embd_neigh})
    dict_node_enode.pop(current_node)
    # get the non embd neighbors of the node
    relevant_n_e = dict_node_node[current_node]
    # delete because now it is in the projection
    dict_node_node.pop(current_node)
    embd_neigh = list(embd_neigh)
    for i in range(len(embd_neigh)):
        f = dict_enode_enode.get(embd_neigh[i])
        if f is not None:
            dict_enode_enode[embd_neigh[i]].update([current_node])
        else:
            dict_enode_enode.update({embd_neigh[i]: set([current_node])})
    # check if num of non embd neighbors of our node bigger then zero
    if len(relevant_n_e) > 0:
        # loop of non embd neighbors
        relevant_n_e1 = list(relevant_n_e)
        for j in range(len(relevant_n_e)):
            tmp_append_n_n = dict_node_node.get(relevant_n_e1[j])
            if tmp_append_n_n is not None:
                # if relevant_n_e1[j] in dict_node_node:
                tmp_append_n_n = tmp_append_n_n-set([current_node])
                dict_node_node[relevant_n_e1[j]] = tmp_append_n_n
            tmp_append = dict_node_enode.get(relevant_n_e1[j])
            if tmp_append is not None:
                # add our node to the set cause now our node is in embd
                tmp_append.update(set([current_node]))
                dict_node_enode[relevant_n_e1[j]] = tmp_append
            else:
                dict_node_enode.update({relevant_n_e1[j]: set([current_node])})
    set_n_e.remove(current_node)
    return condition, dict_enode_proj, dict_node_enode, dict_node_node,dict_enode_enode, set_n_e


def final_function_hope(g, S, U, V, dict_enode_proj, dict_node_enode, dict_node_node, dict_enode_enode, set_n_e, batch_precent, dim):
    """
    the final function that iteratively divided the dictionary of nodes without embedding into number of batches
    determined by batch_precent. It does by building a heap every iteration so that we enter the nodes to the
    projection from the nodes which have the most neighbors in the embedding to the least. This way the projection
    gets more accurate.
    """
    dict_index = create_dict_index(g)
    condition = 1
    k = 0
    set_n_e2 = set_n_e.copy()
    while condition > 0:
        condition = 0
        k += 1
        print(k)
        batch_size = int(batch_precent * len(set_n_e2))
        # loop over node are not in the embedding
        if batch_size > len(set_n_e):
            num_times = len(set_n_e)
        else:
            num_times = batch_size
        list_n_e = list(set_n_e)
        heap = []
        for i in range(len(list_n_e)):
            my_node = list_n_e[i]
            a = dict_node_enode.get(my_node)
            if a is not None:
                num_neighbors = len(dict_node_enode[my_node])
            else:
                num_neighbors = 0
            heapq.heappush(heap, [-num_neighbors, my_node])
        for i in range(len(set_n_e))[:num_times]:
            # look on node number i in the loop
            current_node = heapq.heappop(heap)[1]
            if dict_node_enode.get(current_node) is not None:
                condition, dict_enode_proj, dict_node_enode, dict_node_node, dict_enode_enode, set_n_e = one_iteration(dict_index, S, U, V,
                                                                                                            dict_enode_proj, dict_node_enode, dict_node_node,
                                                                                                            dict_enode_enode, set_n_e, current_node, dim)
    return dict_enode_proj, set_n_e