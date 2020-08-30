"""
Implementation of Graph_Factorization_Loss approach, as explained in the pdf file which is in our github page.
"""

import numpy as np
import heapq
from numpy.linalg import inv


def gf_loss(node, dict_node_enode, dict_proj, a, d):
    """
    Calculation of every new node's embedding according to the formula: Z=(a*I+B)^-1C where a is a hyperparameter
    and -1 means the inverse matrix.
    :param node: Current node its embedding needs to be calculated
    :param dict_node_enode: key == nodes not in projection, value == set of outdoing nodes in projection (i.e
                    there is a directed edge (i,j) when i is the key node and j is in the projection)
    :param dict_proj: key==nodes, value==embeddings
    :param a: hyperparameter to the final formula
    :param d: Embedding Dimension
    :return: The new node's embedding
    """
    neighbors = list(dict_node_enode[node])

    # define I in the equation
    I = np.identity(d)

    # find c in the equation
    c = np.zeros(shape=(1, d))
    for l in range(d):
        sum1 = 0
        for n in neighbors:
            p = np.reshape(dict_proj[n], (1, d))[0, l]
            sum1 += p
        c[0, l] = sum1
    c_t = np.transpose(c)

    # find B in the equation
    B = np.zeros(shape=(d, d))
    for l in range(d):
        for k in range(d):
            sum2 = 0
            for n in neighbors:
                p = np.reshape(dict_proj[n], (1, d))
                sum2 += p[0, k] * p[0, l]
            B[l, k] = sum2

    # calculate embedding with given equation
    z = np.matmul(inv(a * I - B), c_t)
    z_t = np.transpose(z)
    return z_t


def one_iteration(dict_enode_proj, dict_node_enode, dict_node_node, dict_enode_enode, set_n_e, current_node, dim):
    """
    a function that does one iteration over a given batch
    """
    condition = 1
    # get the neighbors in projection of node i
    embd_neigh = dict_node_enode[current_node]
    # the final projection of the node
    final_proj = gf_loss(current_node, dict_node_enode, dict_enode_proj, 0.01, dim)
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


def final_function_LGF(dict_enode_proj, dict_node_enode, dict_node_node, dict_enode_enode, set_n_e, batch_precent, dim):
    """
    the final function that iteratively divided the dictionary of nodes without embedding into number of batches
    determined by batch_precent. It does by building a heap every iteration so that we enter the nodes to the
    projection from the nodes which have the most neighbors in the embedding to the least. This way the projection
    gets more accurate.
    """
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
                condition, dict_enode_proj, dict_node_enode, dict_node_node, dict_enode_enode, set_n_e = one_iteration(dict_enode_proj,
                                                                                                   dict_node_enode,
                                                                                                   dict_node_node, dict_enode_enode,
                                                                                                   set_n_e,
                                                                                                   current_node, dim)
    return dict_enode_proj, set_n_e