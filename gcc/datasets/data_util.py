import os
import utils
import json
from collections import defaultdict, namedtuple

import dgl
import numpy as np
import scipy.sparse as sparse
import sklearn.preprocessing as preprocessing
import torch
import torch.nn.functional as F
from scipy.sparse import linalg
import networkx as nx
from torch_geometric.utils import from_networkx
import pickle as pkl
from ogb.nodeproppred import PygNodePropPredDataset

link_root = "/root/data/link_prediction/"


# /home/srtpgroup/tfgcc/data/link_prediction
def batcher():
    def batcher_dev(batch):
        graph_q, graph_k = zip(*batch)
        graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
        return graph_q, graph_k

    return batcher_dev


def labeled_batcher():
    def batcher_dev(batch):
        graph_q, label = zip(*batch)
        graph_q = dgl.batch(graph_q)
        return graph_q, torch.LongTensor(label)

    return batcher_dev


def idx_labeled_batcher():
    def batcher_dev(batch):
        graph_q, label, idx = zip(*batch)
        graph_q = dgl.batch(graph_q)
        return graph_q, torch.LongTensor(label), idx

    return batcher_dev


def continual_batcher():
    def batcher_dev(batch):
        graph_q, graph_k, label = zip(*batch)
        graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
        return graph_q, graph_k, torch.LongTensor(label)

    return batcher_dev


def continual_uncertain_batcher():
    def batcher_dev(batch):
        graph_q, graph_k, label, uncertain = zip(*batch)
        graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
        return graph_q, graph_k, torch.LongTensor(label), torch.LongTensor(uncertain)

    return batcher_dev


Data = namedtuple("Data", ["x", "edge_index", "y"])


class Edgelist(object):
    def __init__(self, root, name):
        self.name = name
        edge_list_path = os.path.join(root, name + ".edgelist")
        node_label_path = os.path.join(root, name + ".nodelabel")
        edge_index, y, self.node2id = self._preprocess(edge_list_path, node_label_path)
        self.data = Data(x=None, edge_index=edge_index, y=y)
        self.transform = None

    def get(self, idx):
        assert idx == 0
        return self.data

    def _preprocess(self, edge_list_path, node_label_path):
        with open(edge_list_path) as f:
            edge_list = []
            node2id = defaultdict(int)
            for line in f:
                x, y = list(map(int, line.split()))
                # Reindex
                if x not in node2id:
                    node2id[x] = len(node2id)
                if y not in node2id:
                    node2id[y] = len(node2id)
                edge_list.append([node2id[x], node2id[y]])
                edge_list.append([node2id[y], node2id[x]])

        num_nodes = len(node2id)
        with open(node_label_path) as f:
            nodes = []
            labels = []
            label2id = defaultdict(int)
            for line in f:
                x, label = list(map(int, line.split()))
                if label not in label2id:
                    label2id[label] = len(label2id)
                nodes.append(node2id[x])
                if "hindex" in self.name:
                    labels.append(label)
                else:
                    labels.append(label2id[label])
            if "hindex" in self.name:
                median = np.median(labels)
                labels = [int(label > median) for label in labels]
        assert num_nodes == len(set(nodes))
        y = torch.zeros(num_nodes, len(label2id))
        y[nodes, labels] = 1
        return torch.LongTensor(edge_list).t(), y, node2id


class NewDataEdgelist(object):
    def __init__(self, name):
        print("using new data")
        self.name = name
        datasets = json.load(open('./dataset.json'))
        print("####" + name + "####")
        dataset = name
        dataset_str = datasets[dataset]['dataset']
        dataset_path = datasets[dataset]['dataset_path'][0]
        val_size = datasets[dataset]['val_size']

        dataset = utils.PlanetoidData(dataset_str=dataset_str, dataset_path=dataset_path, val_size=val_size)

        adj = dataset._sparse_data["sparse_adj"]
        features = dataset._sparse_data["features"]
        labels = dataset._dense_data["y_all"]

        G = nx.from_scipy_sparse_matrix(adj)
        pyg_G = from_networkx(G)
        edge_index = torch.tensor(pyg_G.edge_index, dtype=torch.long)
        y = torch.Tensor(labels)

        self.data = Data(x=None, edge_index=edge_index, y=y)
        self.transform = None

    def get(self, idx):
        assert idx == 0
        return self.data


class DDDataEdgelist(object):
    def __init__(self, name):
        print("using new data")
        self.name = name

        graph_name = name
        g = nx.Graph()
        label_list = []
        with open("./data/" + graph_name + "/" + graph_name + '.node_labels', 'r') as f:
            for line in f.readlines():
                if graph_name in ["DD68", "DD242", "DD497", "DD687"]:
                    k = line.split(" ")
                else:
                    k = line.split(",")
                g.add_node(int(k[0]) - 1)
                label_list.append(int(k[1]) - 1)

        with open('./data/' + graph_name + "/" + graph_name + '.edges', 'r') as f:
            for line in f.readlines():
                if graph_name in ["DD68", "DD242", "DD497", "DD687"]:
                    k = line.split(" ")
                else:
                    k = line.split(",")
                g.add_edge(int(k[0]) - 1, int(k[1]) - 1)

        pyg_G = from_networkx(g)
        edge_index = torch.tensor(pyg_G.edge_index, dtype=torch.long)
        label = torch.LongTensor(label_list)
        label = label.unsqueeze(dim=-1)
        tmp = torch.unique(label)
        class_num = tmp.shape[0]
        y = torch.zeros(label.shape[0], class_num).scatter_(1, label, 1)

        self.data = Data(x=None, edge_index=edge_index, y=y)
        self.transform = None

    def get(self, idx):
        assert idx == 0
        return self.data

    def _preprocess(self, edge_list_path, node_label_path):
        with open(edge_list_path) as f:
            edge_list = []
            node2id = defaultdict(int)
            for line in f:
                x, y = list(map(int, line.split()))
                # Reindex
                if x not in node2id:
                    node2id[x] = len(node2id)
                if y not in node2id:
                    node2id[y] = len(node2id)
                edge_list.append([node2id[x], node2id[y]])
                edge_list.append([node2id[y], node2id[x]])

        num_nodes = len(node2id)
        with open(node_label_path) as f:
            nodes = []
            labels = []
            label2id = defaultdict(int)
            for line in f:
                x, label = list(map(int, line.split()))
                if label not in label2id:
                    label2id[label] = len(label2id)
                nodes.append(node2id[x])
                if "hindex" in self.name:
                    labels.append(label)
                else:
                    labels.append(label2id[label])
            if "hindex" in self.name:
                median = np.median(labels)
                labels = [int(label > median) for label in labels]
        assert num_nodes == len(set(nodes))
        y = torch.zeros(num_nodes, len(label2id))
        y[nodes, labels] = 1
        return torch.LongTensor(edge_list).t(), y, node2id


class OGBEdgelist(object):
    def __init__(self, name):
        print("using new data")
        self.name = name
        self.name = 'ogbn-' + self.name

        dataset = PygNodePropPredDataset(name=self.name, root="~/nips-rebuttal/dataset")
        data = dataset[0]

        edge_index = data.edge_index
        if self.name != 'ogbn-proteins':
            label = data.y
        else:
            label = data.y[:, 0].unsqueeze(dim=1)
        tmp = torch.unique(label)
        class_num = tmp.shape[0]

        y = torch.zeros(label.shape[0], class_num).scatter_(1, label, 1)
        print(type(y))
        print(y.shape)
        self.data = Data(x=None, edge_index=edge_index, y=y)
        self.transform = None

    def get(self, idx):
        assert idx == 0
        return self.data

    def _preprocess(self, edge_list_path, node_label_path):
        pass


def create_node_classification_dataset(dataset_name):
    if "airport" in dataset_name:
        return Edgelist(
            "data/struc2vec/",
            {
                "usa_airport": "usa-airports",
                "brazil_airport": "brazil-airports",
                "europe_airport": "europe-airports",
            }[dataset_name],
        )
    elif "h-index" in dataset_name:
        return Edgelist(
            "data/hindex/",
            {
                "h-index-rand-1": "aminer_hindex_rand1_5000",
                "h-index-top-1": "aminer_hindex_top1_5000",
                "h-index": "aminer_hindex_rand20intop200_5000",
            }[dataset_name],
        )
    elif dataset_name in ["cora", "texas", "cornell", "wisconsin", ]:
        return NewDataEdgelist(dataset_name, )
    elif dataset_name in ["DD242", "DD68", "DD687"]:
        return DDDataEdgelist(dataset_name, )
    elif dataset_name in ["arxiv", "products", "proteins"]:
        return OGBEdgelist(dataset_name, )
    else:
        raise NotImplementedError


def _rwr_trace_to_dgl_graph(
        g, seed, trace, positional_embedding_size, entire_graph=False
):
    subv = torch.unique(torch.cat(trace)).tolist()
    try:
        subv.remove(seed)
    except ValueError:
        pass
    subv = [seed] + subv
    if entire_graph:
        subg = g.subgraph(g.nodes())
    else:
        subg = g.subgraph(subv)

    subg = _add_undirected_graph_positional_embedding(subg, positional_embedding_size)

    subg.ndata["seed"] = torch.zeros(subg.number_of_nodes(), dtype=torch.long)
    if entire_graph:
        subg.ndata["seed"][seed] = 1
    else:
        subg.ndata["seed"][0] = 1
    return subg


def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return torch.zeros(n, hidden_size)
    laplacian = laplacian.astype("float64")
    ncv = min(n, max(2 * k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype("float64")
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
        except sparse.linalg.eigen.arpack.ArpackError:
            # print("arpack error, retry=", i)
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = torch.zeros(n, k)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    x = torch.from_numpy(x.astype("float32"))
    x = F.pad(x, (0, hidden_size - k), "constant", 0)
    return x


def _add_undirected_graph_positional_embedding(g, hidden_size, retry=10):
    n = g.number_of_nodes()
    adj = g.adjacency_matrix_scipy(transpose=False, return_edge_ids=False).astype(float)
    norm = sparse.diags(
        dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float
    )
    laplacian = norm * adj * norm
    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    g.ndata["pos_undirected"] = x.float()
    return g


class Linklist(object):
    def __init__(self, root, name, node2id_path=None, suffix=".edgelabel"):
        self.name = name
        edge_list_path = os.path.join(root, name + ".edgelist")
        edge_label_path = os.path.join(root, name + suffix)

        edge_index, self.edges, self.edge_labels, self.node2id = self._preprocess(edge_list_path, edge_label_path,
                                                                                  node2id_path)
        self.data = Data(x=None, edge_index=edge_index, y=None)
        self.transform = None

    def _preprocess(self, edge_list_path, edge_label_path, node2id_path):
        node2id = dict()
        if node2id_path is not None:
            node2id = pkl.load(open(node2id_path, 'rb'))
        with open(edge_list_path) as f:
            edge_list = []
            for line in f:
                x, y = list(map(int, line.split()))
                # Reindex
                if x not in node2id:
                    assert node2id_path is None, "node2id dict is wrong"
                    node2id[x] = len(node2id)
                if y not in node2id:
                    assert node2id_path is None, "node2id dict is wrong"
                    node2id[y] = len(node2id)
                edge_list.append([node2id[x], node2id[y]])
                edge_list.append([node2id[y], node2id[x]])
        edge_list = torch.tensor(edge_list, dtype=torch.int64).t()

        with open(edge_label_path) as f:
            edges = []
            labels = []
            for line in f:
                x, y, label = list(map(int, line.split()))
                src_node, target_node = node2id[x], node2id[y]
                assert label in (0, 1)
                edges.append([src_node, target_node])
                labels.append(label)
        edges = torch.tensor(edges, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)

        return edge_list, edges, labels, node2id


def create_link_prediction_dataset(dataset_name, split_num, is_train):
    if "airport" in dataset_name:
        if is_train:
            return Linklist(f'{link_root}/{dataset_name}',
                            str(split_num) + "_train",
                            f"{link_root}/node2id_dicts/{dataset_name}_node2id.pkl")
        else:
            return Linklist(f'{link_root}/{dataset_name}',
                            str(split_num) + "_test",
                            f"{link_root}/node2id_dicts/{dataset_name}_node2id.pkl")
    else:
        if is_train:
            return Linklist(f'{link_root}/{dataset_name}',
                            str(split_num) + "_train",
                            f"{link_root}/node2id_dicts/{dataset_name}_node2id.pkl")
        else:
            return Linklist(f'{link_root}/{dataset_name}',
                            str(split_num) + "_test",
                            f"{link_root}/node2id_dicts/{dataset_name}_node2id.pkl")


def link_prediction_batcher():
    def batcher_dev(batch):
        graph_q, graph_k, label = zip(*batch)
        graph_q, graph_k = dgl.batch(graph_q), dgl.batch(graph_k)
        return graph_q, graph_k, torch.LongTensor(label)

    return batcher_dev
