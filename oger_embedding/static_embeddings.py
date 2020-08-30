from oger_embedding.utils import *
from oger_embedding.LGF import *
from oger_embedding.hope_loss import *
from oger_embedding.OGRE import *
from oger_embedding.D_W_OGRE import main_D_W_OGRE
from oger_embedding.state_of_the_art_embedding import *
import time


def main_static(method, initial_method, G, initial, dim, params, choose="degrees", regu_val=0, weighted_reg=False):
    """
    Main function to run our different static embedding methods- OGRE, DOGRE, WOGRE, LGF.
    :param method: One of our methods - OGRE, DOGRE, WOGRE, LGF (string)
    :param initial_method: state-of-the-art algorithm for initial embedding - node2vec, HOPE, GF (string)
    :param G: The graph to embed
    :param initial: A list of different sizes of initial embedding (in any length, sizes must be integers).
    :param dim: Embedding dimension
    :param params: Dictionary of parameters for thr initial algorithm
    :param choose: How to choose the nodes in the initial embedding - if == "degrees" nodes with highest degrees are
    chosen, if == "k-core" nodes with highest k-core score are chosen.
    :param regu_val: If DOGRE/WOGRE method is applied, one can have a regression with regularization, this is the value
    of the regularization coefficient. Default is 0 (no regularization).
    :param weighted_reg: If DOGRE/WOGRE method is applied, one can have a weighted regression. True for weighted regression,
    else False. Default is False.
    :return: - The applied method and initial method (as strings)
             - list of embedding dictionaries, each connected to a different initial embedding size. The keys of the
             dictionary are the nodes that in the final embedding, values are the embedding vectors.
             - the grapj
             - the parameters dictionary
             - set of nodes that aren't in the final embedding
             - list of times - each member is the running time of the embedding method, corresponding to the matching
             size of initial embedding.

    """
    if method == "DOGRE":
        list_dicts, set_n_e, times = main_D_W_OGRE(G, initial_method, method, initial, dim, params, choose, regu_val,
                                                   weighted_reg)
    elif method == "WOGRE":
        list_dicts, set_n_e, times = main_D_W_OGRE(G, initial_method, method, initial, dim, params, choose, regu_val,
                                                   weighted_reg)

    else:
        user_wish = True

        # choose number of nodes in initial projection. These value corresponds to 116 nodes
        list_dicts = []

        times = []
        for l in initial:
            t = time.time()
            # get the initial projection by set and list to help us later
            if choose == "degrees":
                initial_proj_nodes = get_initial_proj_nodes_by_degrees(G, l)
            else:
                initial_proj_nodes = get_initial_proj_nodes_by_k_core(G, l)
            user_print("number of nodes in initial projection is: " + str(len(initial_proj_nodes)), user_wish)
            n = G.number_of_nodes()
            e = G.number_of_edges()
            user_print("number of nodes in graph is: " + str(n), user_wish)
            user_print("number of edges in graph is: " + str(e), user_wish)
            # the nodes of our graph
            G_nodes = list(G.nodes())
            set_G_nodes = set(G_nodes)
            set_proj_nodes = set(initial_proj_nodes)
            # convert the graph to undirected
            H = G.to_undirected()
            # calculate neighbours dictionary
            neighbors_dict = create_dict_neighbors(H)
            # making all lists to set (to help us later in the code)
            set_nodes_no_proj = set_G_nodes - set_proj_nodes
            # create dicts of connections
            dict_node_node, dict_node_enode, dict_enode_enode = create_dicts_of_connections(set_proj_nodes,
                                                                                            set_nodes_no_proj,
                                                                                            neighbors_dict)
            # creating sub_G to do node2vec on it later
            sub_G = create_sub_G(initial_proj_nodes, G)
            user_print("calculate the projection of the sub graph with {}...".format(initial_method), user_wish)
            if method == "HOPE":
                embedding = HOPE(params, initial_method, sub_G)
                dict_projections, S, X, X1, X2 = embedding.learn_embedding()
                final_dict_enode_proj, set_n_e = final_function_hope(G, S, X1, X2, dict_projections, dict_node_enode,
                                                   dict_node_node, dict_enode_enode, set_nodes_no_proj, 0.01, dim)
            else:
                if initial_method == "GF":
                    my_iter = params["max_iter"]
                    params["max_iter"] = 1500
                    _, dict_projections, _ = final(sub_G, initial_method, params)
                    params["max_iter"] = my_iter
                else:
                    _, dict_projections, _ = final(sub_G, initial_method, params)
                if method == "LGF":
                    final_dict_enode_proj, set_n_e = final_function_LGF(dict_projections, dict_node_enode, dict_node_node,
                                                                       dict_enode_enode, set_nodes_no_proj, 0.01, dim)
                elif method == "OGRE":
                    final_dict_enode_proj, set_n_e = final_function_OGRE(dict_projections, dict_node_enode,
                                                        dict_node_node, dict_enode_enode, set_nodes_no_proj, 0.01, dim, H)
                else:
                    print("non-valid embedding method")
                    break
            elapsed_time = time.time() - t
            times.append(elapsed_time)
            print("running time: ", elapsed_time)
            print("The number of nodes that aren't in the final projection:", len(set_n_e))
            print("The number of nodes that are in the final projection:", len(final_dict_enode_proj))
            list_dicts.append(final_dict_enode_proj)
    return method, initial_method, list_dicts, G, params, set_n_e, times



