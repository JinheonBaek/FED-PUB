import time
import numpy as np

from misc.utils import *
from models.nets import *
from modules.federated import ServerModule

class Server(ServerModule):
    def __init__(self, args, sd, gpu_server):
        super(Server, self).__init__(args, sd, gpu_server)
        self.model = GCN(self.args.n_feat, self.args.n_dims, self.args.n_clss, self.args).cuda(self.gpu_id)

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
        local_train_sizes = []
        for c_id in updated:
            local_weights.append(self.sd[c_id]['model'].copy())
            local_train_sizes.append(self.sd[c_id]['train_size'])
            del self.sd[c_id]
        self.logger.print(f'all clients have been uploaded ({time.time()-st:.2f}s)')

        st = time.time()
        ratio = (np.array(local_train_sizes)/np.sum(local_train_sizes)).tolist()
        self.set_weights(self.model, self.aggregate(local_weights, ratio))
        self.logger.print(f'global model has been updated ({time.time()-st:.2f}s)')

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict, self.gpu_id)

    def get_weights(self):
        return {
            'model': get_state_dict(self.model)
        }

    def save_state(self):
        torch_save(self.args.checkpt_path, 'server_state.pt', {
            'model': get_state_dict(self.model),
        })





