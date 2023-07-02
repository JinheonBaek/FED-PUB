import os
import glob
import json
import random
import numpy as np

from collections import defaultdict, OrderedDict
from misc.forked_pdb import ForkedPdb

import torch
from torch import Tensor

def str2bool(v):
  return v.lower() in ['true', 't']

def torch_save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    fpath = os.path.join(base_dir, filename)    
    torch.save(data, fpath)

def torch_load(base_dir, filename):
    fpath = os.path.join(base_dir, filename)    
    return torch.load(fpath, map_location=torch.device('cpu'))

def shuffle(seed, x, y):
    idx = np.arange(len(x))
    random.seed(seed)
    random.shuffle(idx)
    return [x[i] for i in idx], [y[i] for i in idx]

def save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, filename), 'w+') as outfile:
        json.dump(data, outfile)

def exists(base_dir, filename):
    return os.path.exists(os.path.join(base_dir, filename))

def join_glob(base_dir, filename):
    return glob.glob(os.path.join(base_dir, filename))

def remove_if_exist(base_dir, filename):
    targets = join_glob(base_dir, filename)
    if len(targets)>0:
        for t in targets:
            os.remove(t)

def debugger():
    ForkedPdb().set_trace()

def get_state_dict(model):
    state_dict = convert_tensor_to_np(model.state_dict())
    return state_dict

def set_state_dict(model, state_dict, gpu_id, skip_stat=False, skip_mask=False):
    state_dict = convert_np_to_tensor(state_dict, gpu_id, skip_stat=skip_stat, skip_mask=skip_mask, model=model.state_dict())
    model.load_state_dict(state_dict)
    
def convert_tensor_to_np(state_dict):
    return OrderedDict([(k,v.clone().detach().cpu().numpy()) for k,v in state_dict.items()])

def convert_np_to_tensor(state_dict, gpu_id, skip_stat=False, skip_mask=False, model=None):
    _state_dict = OrderedDict()
    for k,v in state_dict.items():
        if skip_stat:
            if 'running' in k or 'tracked' in k:
                _state_dict[k] = model[k]
                continue
        if skip_mask:
            if 'mask' in k or 'pre' in k or 'pos' in k:
                _state_dict[k] = model[k]
                continue

        if len(np.shape(v)) == 0:
            _state_dict[k] = torch.tensor(v).cuda(gpu_id)
        else:
            _state_dict[k] = torch.tensor(v).requires_grad_().cuda(gpu_id)
    return _state_dict

def convert_np_to_tensor_cpu(state_dict):
    _state_dict = OrderedDict()
    for k,v in state_dict.items():
        _state_dict[k] = torch.tensor(v)
    return _state_dict

def from_networkx(G, group_node_attrs=None, group_edge_attrs=None):
    import networkx as nx
    from torch_geometric.data import Data

    G = G.to_directed() if not nx.is_directed(G) else G

    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
    for i, (src, dst) in enumerate(G.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]

    data = defaultdict(list)

    if G.number_of_nodes() > 0:
        node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
    else:
        node_attrs = {}

    if G.number_of_edges() > 0:
        edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
    else:
        edge_attrs = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        if set(feat_dict.keys()) != set(node_attrs):
            raise ValueError('Not all nodes contain the same attributes')
        for key, value in feat_dict.items():
            data[str(key)].append(value)

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        if set(feat_dict.keys()) != set(edge_attrs):
            raise ValueError('Not all edges contain the same attributes')
        for key, value in feat_dict.items():
            key = f'edge_{key}' if key in node_attrs else key
            data[str(key)].append(value)

    for key, value in G.graph.items():
        if key == 'node_default' or key == 'edge_default':
            continue  # Do not load default attributes.
        key = f'graph_{key}' if key in node_attrs else key
        data[str(key)] = value

    for key, value in data.items():
        if isinstance(value, (tuple, list)) and isinstance(value[0], Tensor):
            data[key] = torch.stack(value, dim=0)
        else:
            try:
                data[key] = torch.tensor(value)
            except (ValueError, TypeError, RuntimeError):
                pass

    data['edge_index'] = edge_index.view(2, -1)
    data = Data.from_dict(data)

    if group_node_attrs is all:
        group_node_attrs = list(node_attrs)
    if group_node_attrs is not None:
        xs = []
        for key in group_node_attrs:
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.x = torch.cat(xs, dim=-1)

    if group_edge_attrs is all:
        group_edge_attrs = list(edge_attrs)
    if group_edge_attrs is not None:
        xs = []
        for key in group_edge_attrs:
            key = f'edge_{key}' if key in node_attrs else key
            x = data[key]
            x = x.view(-1, 1) if x.dim() <= 1 else x
            xs.append(x)
            del data[key]
        data.edge_attr = torch.cat(xs, dim=-1)

    if data.x is None and data.pos is None:
        data.num_nodes = G.number_of_nodes()

    return data
