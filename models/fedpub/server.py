import time
import numpy as np

from scipy.spatial.distance import cosine

from misc.utils import *
from models.nets import *
from modules.federated import ServerModule

class Server(ServerModule):
    def __init__(self, args, sd, gpu_server):
        super(Server, self).__init__(args, sd, gpu_server)
        self.model = MaskedGCN(self.args.n_feat, self.args.n_dims, self.args.n_clss, self.args.l1, self.args).cuda(self.gpu_id)
        self.sd['proxy'] = self.get_proxy_data(args.n_feat)
        self.update_lists = []
        self.sim_matrices = []

    def get_proxy_data(self, n_feat):
        import networkx as nx

        num_graphs, num_nodes = self.args.n_proxy, 100
        data = from_networkx(nx.random_partition_graph([num_nodes] * num_graphs, p_in=0.1, p_out=0, seed=self.args.seed))
        data.x = torch.normal(mean=0, std=1, size=(num_nodes * num_graphs, n_feat))
        return data

    def on_round_begin(self, curr_rnd):
        self.round_begin = time.time()
        self.curr_rnd = curr_rnd
        self.sd['global'] = self.get_weights()

    def on_round_complete(self, updated):
        self.update(updated)
        self.save_state()

    def update(self, updated):
        st = time.time()
        local_weights = []
        local_functional_embeddings = []
        local_train_sizes = []
        for c_id in updated:
            local_weights.append(self.sd[c_id]['model'].copy())
            local_functional_embeddings.append(self.sd[c_id]['functional_embedding'])
            local_train_sizes.append(self.sd[c_id]['train_size'])
            del self.sd[c_id]
        self.logger.print(f'all clients have been uploaded ({time.time()-st:.2f}s)')

        n_connected = round(self.args.n_clients*self.args.frac)
        assert n_connected == len(local_functional_embeddings)
        sim_matrix = np.empty(shape=(n_connected, n_connected))
        for i in range(n_connected):
            for j in range(n_connected):
                sim_matrix[i, j] = 1 - cosine(local_functional_embeddings[i], local_functional_embeddings[j])

        if self.args.agg_norm == 'exp':
            sim_matrix = np.exp(self.args.norm_scale * sim_matrix)
        
        row_sums = sim_matrix.sum(axis=1)
        sim_matrix = sim_matrix / row_sums[:, np.newaxis]

        st = time.time()
        ratio = (np.array(local_train_sizes)/np.sum(local_train_sizes)).tolist()
        self.set_weights(self.model, self.aggregate(local_weights, ratio))
        self.logger.print(f'global model has been updated ({time.time()-st:.2f}s)')

        st = time.time()
        for i, c_id in enumerate(updated):
            aggr_local_model_weights = self.aggregate(local_weights, sim_matrix[i, :])
            if f'personalized_{c_id}' in self.sd: del self.sd[f'personalized_{c_id}']
            self.sd[f'personalized_{c_id}'] = {'model': aggr_local_model_weights}
        self.update_lists.append(updated)
        self.sim_matrices.append(sim_matrix)
        self.logger.print(f'local model has been updated ({time.time()-st:.2f}s)')

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict, self.gpu_id)

    def get_weights(self):
        return {
            'model': get_state_dict(self.model),
        }

    def save_state(self):
        torch_save(self.args.checkpt_path, 'server_state.pt', {
            'model': get_state_dict(self.model),
            'sim_matrices': self.sim_matrices,
            'update_lists': self.update_lists
        })
