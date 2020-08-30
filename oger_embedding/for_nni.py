from oger_embedding.static_embeddings import *


def add_weights(G):
    """
    If the graph is not weighted, add weights equal to 1.
    :param G: The graph
    :return: The weighted version of the graph
    """
    edges = list(G.edges())
    for e in edges:
        G[e[0]][e[1]] = {"weight": 1}
    return G


class StaticEmbeddings:
    """
    Class to run one of our suggested static embedding methods.
    """
    def __init__(self, name, graph, is_weighted=False, initial_method="node2vec", method="OGRE",
                 initial_size=1000, dim=128, choose="degrees"):
        """
        Init function to initialize the class
        :param name: Name of the graph/dataset
        :param graph_path: Path to where the dataset is
        :param is_weighted: True if the graph is weighted, else False
        :param initial_method: Initial state-of-the-art embedding algorithm for the initial embedding. Options are
                "node2vec" , "gf" or "HOPE". Default is "node2vec".
        :param method: One of our suggested static embedding methods. Options are "OGRE", "DOGRE" or "WOGRE". Default
                is "OGRE".
        :param initial_size: Size of initial embedding (integer that is less or equal to the number of nodes in the
                graph). Default value is 1000.
        :param dim: Embedding dimension. Default is 128.
        :param choose: Weather to choose the nodes of the initial embedding by highest degree or highest k-core score.
                Options are "degrees" for the first and "k-core" for the second.
        """
        self.name = name
        # The graph which needs to be embed. If you have a different format change to your own loader.
        # self.graph = self.load_graph(graph_path, name, is_weighted)
        self.graph = graph
        self.initial_method = initial_method
        self.embedding_method = method
        self.initial_size = [initial_size]
        self.dim = dim
        # dictionary of parameters for state-of-the-art method
        self.params_dict = self.define_params_for_initial_method()
        self.choose = choose
        # calculate the given graph embedding and return a dictionary of nodes as keys and embedding vectors as values,
        self.dict_embedding = self.calculate_embedding()

    @ staticmethod
    def load_graph(path, name, is_weighted):
        """
        Data loader assuming the format is a text file with columns of : target source (e.g. 1 2) or target source weight
        (e.g. 1 2 0.34). If you have a different format, you may want to create your own data loader.
        :param path: The path to the edgelist file
        :param name: The name of te dataset
        :param is_weighted: True if the graph is weighted, False otherwise.
        :return: A Directed networkx graph with an attribute of "weight" for each edge.
        """
        if is_weighted is True:
            # where the file is in the format : source target weight
            G = nx.read_weighted_edgelist(os.path.join(path, name + ".txt"), create_using=nx.DiGraph())
        else:
            # where the file is in the format : source target , so we put weight=1 for each edge
            G = nx.read_edgelist(os.path.join(path, name + ".txt"), create_using=nx.DiGraph(), delimiter=",")
            G = add_weights(G)
        return G

    def define_params_for_initial_method(self):
        """
        According to the initial state-of-the-art embedding method, create the dictionary of parameters.
        :return: Parameters dictionary
        """
        if self.initial_method == "node2vec":
            params_dict = {"dimension": self.dim, "walk_length": 80, "num_walks": 16, "workers": 2}
        elif self.initial_method == "gf":
            params_dict ={"dimension": self.dim, "eta": 0.1, "regularization": 0.1, "max_iter": 3000, "print_step": 100}
        elif self.initial_method == "HOPE":
            params_dict = {"dimension": self.dim, "beta": 0.1}
        return params_dict

    def calculate_embedding(self):
        """
        Calculate the graph embedding.
        :return: An embedding dictionary where keys are the nodes that are in the final embedding and values are
                their embedding vectors.
        """
        _, _, list_dicts, _, _, set_n_e, times = main_static(self.embedding_method, self.initial_method, self.graph,
                                                             self.initial_size, self.dim, self.params_dict, self.choose,
                                                             regu_val=0.1, weighted_reg=False)
        return list_dicts[0]

    def save_embedding(self, path):
        """
        Save the calculated embedding in a .npy file.
        :param path: Path to where to save the embedding
        :return: The file name
        """
        file_name = self.name + " + " + self.initial_method + " + " + self.embedding_method + " + " \
                    + str(self.initial_size[0])
        np.save(os.path.join(path, '{}.npy'.format(file_name)), self.dict_embedding)
        return file_name

    def load_embedding(self, path, file_name):
        """
        Given a .npy file - embedding of a given graph. return the embedding dictionary
        :param path: Where this file is saved.
        :param file_name: The name of the file
        :return: Embedding dictionary
        """
        data = np.load(os.path.join(path, '{}.npy'.format(file_name)), allow_pickle=True)
        dict_embedding = data.item()
        return dict_embedding


# name = "Pubmed"
# dataset_path = os.path.join("..", "datasets")
# initial_method = "HOPE"  # node2vec, "HOPE", "GF"
# method = "DOGRE"  # OGRE, DOGRE, WOGRE
# initial_size = 1000
# dim = 100
# choose = "k-core"  # degrees, k-core
# SE = StaticEmbeddings(name, dataset_path, initial_method=initial_method, method=method, initial_size=initial_size, dim=dim)
# dict_embedding = SE.dict_embedding
# file_name = SE.save_embedding(dataset_path)
# same_dict_embedding = load_embedding(dataset_path, file_name)