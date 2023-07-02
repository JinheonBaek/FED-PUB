import time
import torch.nn.functional as F

from misc.utils import *
from data.loader import DataLoader
from modules.logger import Logger

class ServerModule:
    def __init__(self, args, sd, gpu_server):
        self.args = args
        self._args = vars(self.args)
        self.gpu_id = gpu_server
        self.sd = sd
        self.logger = Logger(self.args, self.gpu_id, is_server=True)

    def get_active(self, mask):
        active = np.absolute(mask) >= self.args.l1
        return active.astype(float)

    def aggregate(self, local_weights, ratio=None):
        aggr_theta = OrderedDict([(k,None) for k in local_weights[0].keys()])
        if ratio is not None:
            for name, params in aggr_theta.items():
                aggr_theta[name] = np.sum([theta[name]*ratio[j] for j, theta in enumerate(local_weights)], 0)
        else:
            ratio = 1/len(local_weights)
            for name, params in aggr_theta.items():
                aggr_theta[name] = np.sum([theta[name] * ratio for j, theta in enumerate(local_weights)], 0)
        return aggr_theta

class ClientModule:
    def __init__(self, args, w_id, g_id, sd):
        self.sd = sd
        self.gpu_id = g_id
        self.worker_id = w_id
        self.args = args 
        self._args = vars(self.args)
        self.loader = DataLoader(self.args)
        self.logger = Logger(self.args, self.gpu_id)
       
    def switch_state(self, client_id):
        self.client_id = client_id
        self.loader.switch(client_id)
        self.logger.switch(client_id)
        if self.is_initialized():
            time.sleep(0.1)
            self.load_state()
        else:
            self.init_state()

    def is_initialized(self):
        return os.path.exists(os.path.join(self.args.checkpt_path, f'{self.client_id}_state.pt'))

    @property
    def init_state(self):
        raise NotImplementedError()

    @property
    def save_state(self):
        raise NotImplementedError()

    @property
    def load_state(self):
        raise NotImplementedError()

    @torch.no_grad()
    def validate(self, mode='test'):
        loader = self.loader.pa_loader

        with torch.no_grad():
            target, pred, loss = [], [], []
            for _, batch in enumerate(loader):
                batch = batch.cuda(self.gpu_id)
                mask = batch.test_mask if mode == 'test' else batch.val_mask
                y_hat, lss = self.validation_step(batch, mask)
                pred.append(y_hat[mask])
                target.append(batch.y[mask])
                loss.append(lss)
            acc = self.accuracy(torch.stack(pred).view(-1, self.args.n_clss), torch.stack(target).view(-1))
        return acc, np.mean(loss)

    @torch.no_grad()
    def validation_step(self, batch, mask=None):
        self.model.eval()
        y_hat = self.model(batch)
        if torch.sum(mask).item() == 0: return y_hat, 0.0
        lss = F.cross_entropy(y_hat[mask], batch.y[mask])
        return y_hat, lss.item()

    @torch.no_grad()
    def accuracy(self, preds, targets):
        if targets.size(0) == 0: return 1.0
        with torch.no_grad():
            preds = preds.max(1)[1]
            acc = preds.eq(targets).sum().item() / targets.size(0)
        return acc

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def save_log(self):
        save(self.args.log_path, f'client_{self.client_id}.txt', {
            'args': self._args,
            'log': self.log
        })

    def get_optimizer_state(self, optimizer):
        state = {}
        for param_key, param_values in optimizer.state_dict()['state'].items():
            state[param_key] = {}
            for name, value in param_values.items():
                if torch.is_tensor(value) == False: continue
                state[param_key][name] = value.clone().detach().cpu().numpy()
        return state
