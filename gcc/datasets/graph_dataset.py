#!/usr/bin/env python
# encoding: utf-8
# File Name: graph_dataset.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/11 12:17
# TODO:

import math
import operator
from sklearn.model_selection import StratifiedKFold
from sklearn.semi_supervised import LabelPropagation
import dgl
import dgl.data
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from dgl.data import AmazonCoBuy, Coauthor
from dgl.nodeflow import NodeFlow

from gcc.datasets import data_util
import torch_geometric as ptgeom


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.graphs, _ = dgl.data.utils.load_graphs(
        dataset.dgl_graphs_file, dataset.jobs[worker_id]
    )
    dataset.length = sum([g.number_of_nodes() for g in dataset.graphs])
    np.random.seed(worker_info.seed % (2 ** 32))


class LoadBalanceGraphDataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            rw_hops=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
            num_workers=1,
            dgl_graphs_file="./data/small.bin",
            num_samples=10000,
            num_copies=1,
            graph_transform=None,
            aug="rwr",
            num_neighbors=5,
    ):
        super(LoadBalanceGraphDataset).__init__()
        self.rw_hops = rw_hops
        self.num_neighbors = num_neighbors
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        self.num_samples = num_samples
        assert sum(step_dist) == 1.0
        assert positional_embedding_size > 1
        self.dgl_graphs_file = dgl_graphs_file
        graph_sizes = dgl.data.utils.load_labels(dgl_graphs_file)[
            "graph_sizes"
        ].tolist()

        print("load graph done")

        # a simple greedy algorithm for load balance
        # sorted graphs w.r.t its size in decreasing order
        # for each graph, assign it to the worker with least workload
        assert num_workers % num_copies == 0
        jobs = [list() for i in range(num_workers // num_copies)]
        workloads = [0] * (num_workers // num_copies)
        graph_sizes = sorted(
            enumerate(graph_sizes), key=operator.itemgetter(1), reverse=True
        )
        for idx, size in graph_sizes:
            argmin = workloads.index(min(workloads))
            workloads[argmin] += size
            jobs[argmin].append(idx)
        self.jobs = jobs * num_copies
        self.total = self.num_samples * num_workers
        self.graph_transform = graph_transform
        assert aug in ("rwr", "ns")
        self.aug = aug

    def __len__(self):
        return self.num_samples * num_workers

    def __iter__(self):
        degrees = torch.cat([g.in_degrees().double() ** 0.75 for g in self.graphs])
        prob = degrees / torch.sum(degrees)
        samples = np.random.choice(
            self.length, size=self.num_samples, replace=True, p=prob.numpy()
        )
        # self.length和之后的self.graphs是哪里来的？
        for idx in samples:
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()

        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx
        else:
            other_node_idx = dgl.contrib.sampling.random_walk(
                g=self.graphs[graph_idx], seeds=[node_idx], num_traces=1, num_hops=step
            )[0][0][-1].item()

        if self.aug == "rwr":
            max_nodes_per_seed = max(
                self.rw_hops,
                int(
                    (
                            (self.graphs[graph_idx].in_degree(node_idx) ** 0.75)
                            * math.e
                            / (math.e - 1)
                            / self.restart_prob
                    )
                    + 0.5
                ),
            )
            traces = dgl.contrib.sampling.random_walk_with_restart(
                self.graphs[graph_idx],
                seeds=[node_idx, other_node_idx],
                restart_prob=self.restart_prob,
                max_nodes_per_seed=max_nodes_per_seed,
            )
        elif self.aug == "ns":
            prob = dgl.backend.tensor([], dgl.backend.float32)
            prob = dgl.backend.zerocopy_to_dgl_ndarray(prob)
            nf1 = dgl.contrib.sampling.sampler._CAPI_NeighborSampling(
                self.graphs[graph_idx]._graph,
                dgl.utils.toindex([node_idx]).todgltensor(),
                0,  # batch_start_id
                1,  # batch_size
                1,  # workers
                self.num_neighbors,  # expand_factor
                self.rw_hops,  # num_hops
                "out",
                False,
                prob,
            )[0]
            nf1 = NodeFlow(self.graphs[graph_idx], nf1)
            trace1 = [nf1.layer_parent_nid(i) for i in range(nf1.num_layers)]
            nf2 = dgl.contrib.sampling.sampler._CAPI_NeighborSampling(
                self.graphs[graph_idx]._graph,
                dgl.utils.toindex([other_node_idx]).todgltensor(),
                0,  # batch_start_id
                1,  # batch_size
                1,  # workers
                self.num_neighbors,  # expand_factor
                self.rw_hops,  # num_hops
                "out",
                False,
                prob,
            )[0]
            nf2 = NodeFlow(self.graphs[graph_idx], nf2)
            trace2 = [nf2.layer_parent_nid(i) for i in range(nf2.num_layers)]
            traces = [trace1, trace2]

        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
        )
        graph_k = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=traces[1],
            positional_embedding_size=self.positional_embedding_size,
        )
        if self.graph_transform:
            graph_q = self.graph_transform(graph_q)
            graph_k = self.graph_transform(graph_k)
        return graph_q, graph_k


class GraphDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            rw_hops=64,
            subgraph_size=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
    ):
        super(GraphDataset).__init__()
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert sum(step_dist) == 1.0
        assert positional_embedding_size > 1
        #  graphs = []
        graphs, _ = dgl.data.utils.load_graphs(
            "data_bin/dgl/lscc_graphs.bin", [0, 1, 2]
        )
        for name in ["cs", "physics"]:
            g = Coauthor(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            g.readonly()
            graphs.append(g)
        for name in ["computers", "photo"]:
            g = AmazonCoBuy(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            g.readonly()
            graphs.append(g)
        # more graphs are comming ...
        print("load graph done")
        self.graphs = graphs
        self.length = sum([g.number_of_nodes() for g in self.graphs])

    def __len__(self):
        return self.length

    def _convert_idx(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()
        return graph_idx, node_idx

    def __getitem__(self, idx):
        graph_idx, node_idx = self._convert_idx(idx)

        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx
        else:
            other_node_idx = dgl.contrib.sampling.random_walk(
                g=self.graphs[graph_idx], seeds=[node_idx], num_traces=1, num_hops=step
            )[0][0][-1].item()

        max_nodes_per_seed = max(
            self.rw_hops,
            int(
                (
                        self.graphs[graph_idx].out_degree(node_idx)
                        * math.e
                        / (math.e - 1)
                        / self.restart_prob
                )
                + 0.5
            ),
        )
        # max_nodes_per_seed = 16
        traces = dgl.contrib.sampling.random_walk_with_restart(
            self.graphs[graph_idx],
            seeds=[node_idx, other_node_idx],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=max_nodes_per_seed,
        )

        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
        )
        graph_k = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=traces[1],
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
        )
        return graph_q, graph_k


class PesudoGraphDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            rw_hops=64,
            subgraph_size=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
    ):
        super(PesudoGraphDataset).__init__()
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert sum(step_dist) == 1.0
        assert positional_embedding_size > 1
        #  graphs = []
        graphs, _ = dgl.data.utils.load_graphs(
            "data_bin/dgl/lscc_graphs.bin", [0, 1, 2]
        )
        for name in ["cs", "physics"]:
            g = Coauthor(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            g.readonly()
            graphs.append(g)
        for name in ["computers", "photo"]:
            g = AmazonCoBuy(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            g.readonly()
            graphs.append(g)
        # more graphs are comming ...
        print("load graph done")
        self.graphs = graphs
        self.length = sum([g.number_of_nodes() for g in self.graphs])

    def __len__(self):
        return self.length

    def _convert_idx(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()
        return graph_idx, node_idx

    def __getitem__(self, idx):
        graph_idx, node_idx = self._convert_idx(idx)

        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx
        else:
            other_node_idx = dgl.contrib.sampling.random_walk(
                g=self.graphs[graph_idx], seeds=[node_idx], num_traces=1, num_hops=step
            )[0][0][-1].item()

        max_nodes_per_seed = max(
            self.rw_hops,
            int(
                (
                        self.graphs[graph_idx].out_degree(node_idx)
                        * math.e
                        / (math.e - 1)
                        / self.restart_prob
                )
                + 0.5
            ),
        )
        # max_nodes_per_seed = 16
        traces = dgl.contrib.sampling.random_walk_with_restart(
            self.graphs[graph_idx],
            seeds=[node_idx, other_node_idx],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=max_nodes_per_seed,
        )

        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
        )
        graph_k = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=traces[1],
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
        )
        return graph_q, graph_k


class NodeClassificationDataset(GraphDataset):
    def __init__(
            self,
            dataset,
            rw_hops=64,
            subgraph_size=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
    ):
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert positional_embedding_size > 1

        self.data = data_util.create_node_classification_dataset(dataset).data
        self.graphs = [self._create_dgl_graph(self.data)]
        self.length = sum([g.number_of_nodes() for g in self.graphs])
        self.total = self.length

    def _create_dgl_graph(self, data):
        graph = dgl.DGLGraph()
        src, dst = data.edge_index.tolist()
        num_nodes = data.edge_index.max() + 1
        graph.add_nodes(num_nodes)
        graph.add_edges(src, dst)
        graph.add_edges(dst, src)
        graph.readonly()
        return graph


class PesudoNodeClassificationDataset(PesudoGraphDataset):
    def __init__(
            self,
            dataset,
            rw_hops=64,
            subgraph_size=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
    ):
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert positional_embedding_size > 1

        self.data = data_util.create_node_classification_dataset(dataset).data
        self.graphs = [self._create_dgl_graph(self.data)]
        self.length = sum([g.number_of_nodes() for g in self.graphs])
        self.total = self.length

    def _create_dgl_graph(self, data):
        graph = dgl.DGLGraph()
        src, dst = data.edge_index.tolist()
        num_nodes = data.edge_index.max() + 1
        graph.add_nodes(num_nodes)
        graph.add_edges(src, dst)
        graph.add_edges(dst, src)
        graph.readonly()
        return graph


class NodeClassificationDatasetLabeled(NodeClassificationDataset):
    def __init__(
            self,
            dataset,
            rw_hops=64,
            subgraph_size=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
            cat_prone=False,
    ):
        super(NodeClassificationDatasetLabeled, self).__init__(
            dataset,
            rw_hops,
            subgraph_size,
            restart_prob,
            positional_embedding_size,
            step_dist,
        )
        assert len(self.graphs) == 1
        self.num_classes = self.data.y.shape[1]

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()

        traces = dgl.contrib.sampling.random_walk_with_restart(
            self.graphs[graph_idx],
            seeds=[node_idx],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=self.rw_hops,
        )

        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
        )
        return graph_q, self.data.y[idx].argmax().item()


class PesudoNodeClassificationDatasetLabeled(NodeClassificationDataset):
    def __init__(
            self,
            dataset,
            rw_hops=64,
            subgraph_size=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
            cat_prone=False,
    ):
        super(PesudoNodeClassificationDatasetLabeled, self).__init__(
            dataset,
            rw_hops,
            subgraph_size,
            restart_prob,
            positional_embedding_size,
            step_dist,
        )
        assert len(self.graphs) == 1
        self.num_classes = self.data.y.shape[1]

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()

        traces = dgl.contrib.sampling.random_walk_with_restart(
            self.graphs[graph_idx],
            seeds=[node_idx],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=self.rw_hops,
        )

        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
        )
        return graph_q, self.data.y[idx].argmax().item(), node_idx


class ContinualLearningDataset(NodeClassificationDataset):
    def __init__(
            self,
            dataset,
            rw_hops=64,
            subgraph_size=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
            cat_prone=False,
    ):
        super(ContinualLearningDataset, self).__init__(
            dataset,
            rw_hops,
            subgraph_size,
            restart_prob,
            positional_embedding_size,
            step_dist,
        )
        assert len(self.graphs) == 1
        self.num_classes = self.data.y.shape[1]
        self.mode = 'bridge'
        self.uncertain = [1] * self.data.y.shape[0]

    def __getitem__(self, idx):

        if self.mode == 'bridge':
            graph_idx, node_idx = self._convert_idx(idx)
            step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
            if step == 0:
                other_node_idx = node_idx
            else:
                other_node_idx = dgl.contrib.sampling.random_walk(
                    g=self.graphs[graph_idx], seeds=[node_idx], num_traces=1, num_hops=step
                )[0][0][-1].item()

            max_nodes_per_seed = max(
                self.rw_hops,
                int(
                    (
                            self.graphs[graph_idx].out_degree(node_idx)
                            * math.e
                            / (math.e - 1)
                            / self.restart_prob
                    )
                    + 0.5
                ),
            )
            traces = dgl.contrib.sampling.random_walk_with_restart(
                self.graphs[graph_idx],
                seeds=[node_idx, other_node_idx],
                restart_prob=self.restart_prob,
                max_nodes_per_seed=max_nodes_per_seed,
            )

            graph_q = data_util._rwr_trace_to_dgl_graph(
                g=self.graphs[graph_idx],
                seed=node_idx,
                trace=traces[0],
                positional_embedding_size=self.positional_embedding_size,
                entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
            )
            graph_k = data_util._rwr_trace_to_dgl_graph(
                g=self.graphs[graph_idx],
                seed=other_node_idx,
                trace=traces[1],
                positional_embedding_size=self.positional_embedding_size,
                entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
            )
            return graph_q, graph_k, self.data.y[idx].argmax().item()
        else:
            graph_idx = 0
            node_idx = idx

            traces = dgl.contrib.sampling.random_walk_with_restart(
                self.graphs[graph_idx],
                seeds=[node_idx],
                restart_prob=self.restart_prob,
                max_nodes_per_seed=self.rw_hops,
            )

            graph_q = data_util._rwr_trace_to_dgl_graph(
                g=self.graphs[graph_idx],
                seed=node_idx,
                trace=traces[0],
                positional_embedding_size=self.positional_embedding_size,
            )
            return graph_q, graph_q, self.data.y[idx].argmax().item()



class PesudoContinualLearningDataset(PesudoNodeClassificationDataset):
    def __init__(
            self,
            dataset,
            rw_hops=64,
            subgraph_size=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
            labels=None,
            test_id=None,
            cat_prone=False,
    ):
        super(PesudoContinualLearningDataset, self).__init__(
            dataset,
            rw_hops,
            subgraph_size,
            restart_prob,
            positional_embedding_size,
            step_dist,
        )
        assert len(self.graphs) == 1
        self.label = labels
        self.num_classes = self.data.y.shape[1]
        self.uncertain = [1] * self.data.y.shape[0]
        self.mode = 'bridge'

    def update_label(self, index_list, label_list, uncertrain_list=None):
        labels = self.data.y.argmax(dim=1).tolist()
        if index_list == None:
            return
        count = 0
        self.uncertain = [1] * len(labels)
        for i in range(len(index_list)):
            for j in range(len(index_list[i])):
                idx = index_list[i][j]
                if self.label[idx] != label_list[i][j]:
                    count = count + 1
                self.label[idx] = label_list[i][j]
                self.uncertain[idx] = uncertrain_list[i][j]
        print("change", count)

        # self.label = labels

    def __getitem__(self, idx):
        if self.mode == 'bridge':
            graph_idx, node_idx = self._convert_idx(idx)
            step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
            if step == 0:
                other_node_idx = node_idx
            else:
                other_node_idx = dgl.contrib.sampling.random_walk(
                    g=self.graphs[graph_idx], seeds=[node_idx], num_traces=1, num_hops=step
                )[0][0][-1].item()

            max_nodes_per_seed = max(
                self.rw_hops,
                int(
                    (
                            self.graphs[graph_idx].out_degree(node_idx)
                            * math.e
                            / (math.e - 1)
                            / self.restart_prob
                    )
                    + 0.5
                ),
            )
            traces = dgl.contrib.sampling.random_walk_with_restart(
                self.graphs[graph_idx],
                seeds=[node_idx, other_node_idx],
                restart_prob=self.restart_prob,
                max_nodes_per_seed=max_nodes_per_seed,
            )

            graph_q = data_util._rwr_trace_to_dgl_graph(
                g=self.graphs[graph_idx],
                seed=node_idx,
                trace=traces[0],
                positional_embedding_size=self.positional_embedding_size,
                entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
            )
            graph_k = data_util._rwr_trace_to_dgl_graph(
                g=self.graphs[graph_idx],
                seed=other_node_idx,
                trace=traces[1],
                positional_embedding_size=self.positional_embedding_size,
                entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
            )
            return graph_q, graph_k, self.data.y[idx].argmax().item(), self.uncertain[idx]
        else:
            graph_idx = 0
            node_idx = idx

            traces = dgl.contrib.sampling.random_walk_with_restart(
                self.graphs[graph_idx],
                seeds=[node_idx],
                restart_prob=self.restart_prob,
                max_nodes_per_seed=self.rw_hops,
            )

            graph_q = data_util._rwr_trace_to_dgl_graph(
                g=self.graphs[graph_idx],
                seed=node_idx,
                trace=traces[0],
                positional_embedding_size=self.positional_embedding_size,
            )
            return graph_q, graph_q, self.data.y[idx].argmax().item(), self.uncertain[idx]


class NodeClassificationDatasetLabeledIdx(NodeClassificationDataset):
    def __init__(
            self,
            dataset,
            rw_hops=64,
            subgraph_size=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
            cat_prone=False,
    ):
        super(NodeClassificationDatasetLabeledIdx, self).__init__(
            dataset,
            rw_hops,
            subgraph_size,
            restart_prob,
            positional_embedding_size,
            step_dist,
        )
        assert len(self.graphs) == 1
        self.num_classes = self.data.y.shape[1]

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()

        traces = dgl.contrib.sampling.random_walk_with_restart(
            self.graphs[graph_idx],
            seeds=[node_idx],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=self.rw_hops,
        )

        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
        )
        return graph_q, self.data.y[idx].argmax().item(), idx


class LinkPredictionDatasetLabeled(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset,
            is_train,
            rw_hops=64,
            subgraph_size=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
            split_num=0,
    ):
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert positional_embedding_size > 1

        self.data = data_util.create_link_prediction_dataset(dataset, split_num=split_num, is_train=is_train)
        self.dataset = dataset

        from torch_geometric.data import Data
        from torch_geometric.utils.convert import to_networkx
        pyg = Data(edge_index=self.data.edges.t())
        nx_g = to_networkx(pyg).to_undirected()
        count = 0
        edge_count = 0
        for sub_g in nx.connected_components(nx_g):
            sub_g = nx_g.subgraph(sub_g)
            g_node = sub_g.nodes()
            count = count + 1
            edge_count = edge_count + len(sub_g.edges())
        print(count)
        print(edge_count)

        self.graphs = [self._create_dgl_graph(self.data.data)]
        self.length = self.data.edges.shape[0]
        self.total = self.length

    def _create_dgl_graph(self, data):
        graph = dgl.DGLGraph()
        is_undirected = ptgeom.utils.is_undirected(data.edge_index)
        src, dst = data.edge_index.tolist()
        num_nodes = data.edge_index.max() + 1
        graph.add_nodes(num_nodes)
        graph.add_edges(src, dst)
        if not is_undirected:
            graph.add_edges(dst, src)
        graph.readonly()
        return graph

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        graph_idx, edge_idx = 0, idx
        edge = self.data.edges[edge_idx]
        src_node_idx, tgt_node_idx = edge

        max_nodes_per_seed = max(
            self.rw_hops,
            int(
                (
                        self.graphs[graph_idx].out_degree(src_node_idx)
                        * math.e
                        / (math.e - 1)
                        / self.restart_prob
                )
                + 0.5
            ),
            int(
                (
                        self.graphs[graph_idx].out_degree(tgt_node_idx)
                        * math.e
                        / (math.e - 1)
                        / self.restart_prob
                )
                + 0.5
            ),

        )

        if self.dataset == 'arxiv':
            max_nodes_per_seed = 16

        traces = dgl.contrib.sampling.random_walk_with_restart(
            self.graphs[graph_idx],
            seeds=[src_node_idx, tgt_node_idx],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=max_nodes_per_seed,
        )

        graph_src = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=src_node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
        )

        graph_tgt = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=tgt_node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
        )

        return graph_src, graph_tgt, self.data.edge_labels[edge_idx].item()
