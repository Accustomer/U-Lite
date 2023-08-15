import os
import torch
import shutil
import numpy as np
from models import createULite
from matplotlib import pyplot as plt

plt.style.use('seaborn-darkgrid')


def check_savedir(path):
    if os.path.exists(path):
        print("Save directory exists:", path)
        print("Cover the results in current directory? (1: YES, 0: NO):")
        ret = input()
        return bool(int(ret))
    else:
        os.makedirs(path)
        return True
    
    
def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

    
def load_model(path, device, cfg):
    model = createULite(cfg)
    if path:
        ckpt = torch.load(path, map_location=device)
        if 'state_dict' in ckpt:
            csd = intersect_dicts(ckpt['state_dict'], model.state_dict())
        else:
            csd = intersect_dicts(ckpt, model.state_dict())
        model.load_state_dict(csd, strict=False)
    print(model)
    return model


def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def plot_results(log_file, show_file):
    with open(log_file, 'r') as f:
        lines = f.read().strip().splitlines()
    titles = lines[0].split(',')
    npics = len(titles)
    y = np.array([[float(v) for v in line.split(',')] for line in lines[1:]])
    x = range(len(y))
    fig, ax = plt.subplots(2, npics // 2, figsize=(npics // 2 * 4, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(npics):
        ax[i].plot(x, y[:, i], marker='.', linewidth=2, markersize=8)
        ax[i].set_title(titles[i])
    fig.savefig(show_file, dpi=200)
    
    
def plot_pr_curve(rec, pre, save_path, names=()):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    rec, pre = np.flip(rec), np.flip(pre)
    nstep, nclass = rec.shape
    px = np.linspace(0, 1, nstep)
    py = []
    ap = np.zeros(nclass)
    for i in range(nclass):
        py.append(np.interp(px, rec[:, i], pre[:, i]))  # precision at mAP@0.5
        ap[i] = np.trapz(py[-1], px)
    py = np.stack(py, axis=1)
        
    if 0 < nclass < 21:  # display per-class legend if < 21 classes
        for i in range(nclass):
            ax.plot(px, py[:, i], linewidth=1, label=f'{names[i]} {ap[i]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py[:, 0], linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(axis=1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap.mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(save_path, dpi=250)

        
def plot_mc_curve(px, py, save_path='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    nclass = len(names)
    assert nclass == py.shape[1]
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < nclass < 21:  # display per-class legend if < 21 classes
        for i in range(nclass):
            ax.plot(px, py[:, i], linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.flatten(), linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(axis=1)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(save_path, dpi=250)
