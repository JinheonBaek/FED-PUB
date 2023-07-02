import torch
import random
import numpy as np

import metispy as metis

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from utils import get_data, split_train, torch_save

data_path = '../../../datasets'
ratio_train = 0.2
seed = 1234
clients = [5, 10, 20]

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def generate_data(dataset, n_clients):
    data = split_train(get_data(dataset, data_path), dataset, data_path, ratio_train, 'disjoint', n_clients)
    split_subgraphs(n_clients, data, dataset)

def split_subgraphs(n_clients, data, dataset):
    G = torch_geometric.utils.to_networkx(data)
    n_cuts, membership = metis.part_graph(G, n_clients)
    assert len(list(set(membership))) == n_clients
    print(f'graph partition done, metis, n_partitions: {len(list(set(membership)))}, n_lost_edges: {n_cuts}')
        
    adj = to_dense_adj(data.edge_index)[0]
    for client_id in range(n_clients):
        client_indices = np.where(np.array(membership) == client_id)[0]
        client_indices = list(client_indices)
        client_num_nodes = len(client_indices)

        client_edge_index = []
        client_adj = adj[client_indices][:, client_indices]
        client_edge_index, _ = dense_to_sparse(client_adj)
        client_edge_index = client_edge_index.T.tolist()
        client_num_edges = len(client_edge_index)

        client_edge_index = torch.tensor(client_edge_index, dtype=torch.long)
        client_x = data.x[client_indices]
        client_y = data.y[client_indices]
        client_train_mask = data.train_mask[client_indices]
        client_val_mask = data.val_mask[client_indices]
        client_test_mask = data.test_mask[client_indices]

        client_data = Data(
            x = client_x,
            y = client_y,
            edge_index = client_edge_index.t().contiguous(),
            train_mask = client_train_mask,
            val_mask = client_val_mask,
            test_mask = client_test_mask
        )
        assert torch.sum(client_train_mask).item() > 0

        torch_save(data_path, f'{dataset}_disjoint/{n_clients}/partition_{client_id}.pt', {
            'client_data': client_data,
            'client_id': client_id
        })
        print(f'client_id: {client_id}, iid, n_train_node: {client_num_nodes}, n_train_edge: {client_num_edges}')

for n_clients in clients:
    generate_data(dataset='Cora', n_clients=n_clients)
