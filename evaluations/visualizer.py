import os 
import json
import glob
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_acc_over_round(models, upto, opt): 
    #############################################################################
    processed = get_acc_over_round(models, upto, eval_type='test')
    #############################################################################
    matplotlib.rcParams.update({'font.size': opt['plt_font_size']})
    matplotlib.rcParams['axes.linewidth']=opt['plt_line_width']
    plt.figure(figsize=opt['plt_fig_size'])
    plt.title(opt['plt_title'], fontsize=opt['plt_font_size'])
    plt.ylabel(opt['plt_y_label'], fontsize=opt['plt_font_size'])
    plt.xlabel(opt['plt_x_label'], fontsize=opt['plt_font_size'])
    if opt['plt_background_grid']:
        plt.grid(linestyle='-.', linewidth=0.5)
    for name, proc in processed.items():
        if 'plt_y_interval' in opt.keys():
            proc['x'] = proc['x'][proc['y'] > opt['plt_y_interval'][0]]
            proc['y'] = proc['y'][proc['y'] > opt['plt_y_interval'][0]]
        plt.plot(proc['x'], proc['y'], 
            label=name, color=proc['viz']['color'], 
            linewidth=proc['viz']['linewidth'], linestyle=proc['viz']['linestyle'],
            marker=proc['viz']['marker'] if 'marker' in proc['viz'] else None, 
            markevery=proc['viz']['markevery']  if 'markevery' in proc['viz'] else None,
            markersize=proc['viz']['markersize']  if 'markersize' in proc['viz'] else None,
            clip_on=False)
    plt.xlim([1,upto+1])
    plt.xticks(np.concatenate([[1],np.arange(0,upto+1,opt['plt_x_interval'])[1:]], 0))
    if 'plt_y_interval' in opt.keys():
        plt.yticks(opt['plt_y_interval'])
    plt.tight_layout()    
    plt.legend(**opt['plt_legend_opt'])
    plt.savefig(opt['plt_save'], dpi=300) 
    plt.show()

def summary(models, upto, target_acc, target_rnd):
    
    RND_AT_LOC_PERF_ON_PART = f'Rnd @ Acc {target_acc}'
    LOC_PERF_ON_PART_AT_RND = f'Acc @ Rnd {target_rnd}'
    LOC_PERF_ON_PART_AT_BEST_VAL = f'Acc @ Best Val'
    LOC_PERF_ON_PART_AT_BEST_VAL_ALL = f'Acc @ Best Val All'
    LOC_PERF_ON_PART_AT_BEST_VAL_STD = f'Std @ Best Val'
    LOC_PERF_ON_PART_AT_BEST_VAL_ALL_STD = f'Std @ Best Val All'

    processed = {
        'model': [],
        RND_AT_LOC_PERF_ON_PART: [],
        LOC_PERF_ON_PART_AT_RND: [],
        LOC_PERF_ON_PART_AT_BEST_VAL: [],
        LOC_PERF_ON_PART_AT_BEST_VAL_STD: [],
        LOC_PERF_ON_PART_AT_BEST_VAL_ALL: [],
        LOC_PERF_ON_PART_AT_BEST_VAL_ALL_STD: [],
        'Processed Rnds': []
    }
    
    ltest = get_acc_over_round(models, upto, eval_type='test')
    lval = get_acc_over_round(models, upto, eval_type='val')
    
    for model in models:
        local_test_acc = ltest[model['name']]['y']
        local_val_acc = lval[model['name']]['y']
        local_val_acc_all = lval[model['name']]['y_all']
        local_test_acc_all = ltest[model['name']]['y_all']
        
        for rnd, acc in enumerate(local_test_acc):
            if acc>=target_acc:
                processed[RND_AT_LOC_PERF_ON_PART].append(rnd+1)
                break
            if rnd+1 == len(local_test_acc):
                processed[RND_AT_LOC_PERF_ON_PART].append('N/A')

        processed[LOC_PERF_ON_PART_AT_RND].append(local_test_acc[target_rnd-1])
        
        idx = np.argmax(local_val_acc)
        processed[LOC_PERF_ON_PART_AT_BEST_VAL].append(local_test_acc[idx])
        processed[LOC_PERF_ON_PART_AT_BEST_VAL_STD].append(np.round(np.std(np.array(local_test_acc_all)[:, idx]), 2))

        idx_all = np.argmax(local_val_acc_all, 1)
        _local_test_acc_all = []
        for _i, _idx in enumerate(idx_all):
            _local_test_acc_all.append(local_test_acc_all[_i][_idx])
        processed[LOC_PERF_ON_PART_AT_BEST_VAL_ALL].append(np.round(np.mean(_local_test_acc_all), 2))
        processed[LOC_PERF_ON_PART_AT_BEST_VAL_ALL_STD].append(np.round(np.std(_local_test_acc_all), 2))
        processed['model'].append(model['name'])
        processed['Processed Rnds'].append(len(ltest[model['name']]['y']))
    
    pd.options.display.max_columns = None
    df = pd.DataFrame(data=processed)
    return df

def get_acc_over_round(models, upto=-1,eval_type='test'): 
    data = {}
    for model in models:
        y_trials = []
        for log in model['logs']:
            min_n_rnds = 9999
            clients = {}
            for i, client in enumerate(glob.glob(os.path.join(log, 'client*.txt'))):
                with open(client) as f:
                    client =json.loads(f.read())
                    n_rnds = len(client['log'][f'rnd_local_{eval_type}_acc'][:upto]) 
                    if n_rnds < min_n_rnds:
                        min_n_rnds = n_rnds
                    clients[i] = client['log'][f'rnd_local_{eval_type}_acc']
            y_trial = np.round(np.mean([client[:min_n_rnds] for i, client in clients.items()], 0)*100, 2)
            y_trials.append(y_trial)
        min_n_rnds = np.min([len(y) for y in y_trials])
        y = [y[:min_n_rnds] for y in y_trials]
        y_avg = np.round(np.mean(y, 0), 2)
        y_std = np.round(np.std(y, 0), 2)
        data[model['name']] = {
            'x': np.arange(len(y_avg))+1,
            'y': y_avg,
            'std': y_std,
            'y_all': y,
            'process_rnd': min_n_rnds,
            'viz': model['viz']
        }
    return data











