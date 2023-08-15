import torch
from torch import Tensor
# from sklearn.metrics import precision_recall_curve


def calc_mask_iou(pred: Tensor, target: Tensor):
    union_area = torch.bitwise_or(pred, target).sum()
    inter_area = torch.bitwise_and(pred, target).sum()
    return inter_area / (union_area + 1e-6)

    
def calc_map(pred: Tensor, target: Tensor):
    target_bool = target > 0
    ious = []
    for t in torch.arange(0.5, 1, 0.05):
        pred_bool = pred >= t
        ious.append(calc_mask_iou(pred_bool, target_bool))
    return ious[0], sum(ious) / len(ious)


def calc_prf(pred: Tensor, target: Tensor, prob_range=None):
    # Compute precision, recall, f1_score
    if prob_range is None:
        prob_range = torch.linspace(0, 1, 1000)
    target_bool = target > 0
    sdim = (0, 2, 3)
    numl = target_bool.sum(dim=sdim)
    eps = 1e-6        
    ps, rs, fs = [], [], []
    for t in prob_range:
        pred_bool = pred >= t
        nump = pred_bool.sum(dim=sdim)
        tp = torch.bitwise_and(pred_bool, target_bool).sum(sdim)
        precision = tp / (nump + eps) if nump > 0 else torch.ones_like(nump, device=nump.device)
        recall = tp / (numl + eps)
        f1 = 2 * tp / (nump + numl + eps)        
        ps.append(precision)
        rs.append(recall)
        fs.append(f1)
    ps = torch.vstack(ps)
    rs = torch.vstack(rs)
    fs = torch.vstack(fs)
    return ps, rs, fs
        
        
def calc_metrics_item(pred: Tensor, target: Tensor, prob_range=None):
    if prob_range is None:
        prob_range = torch.linspace(0, 1, 1000)
    target_bool = target > 0
    sdim = (0, 2, 3)
    numl = target_bool.sum(dim=sdim)  
    numps, tps = [], []
    for t in prob_range:
        pred_bool = pred >= t
        nump = pred_bool.sum(dim=sdim)
        tp = torch.bitwise_and(pred_bool, target_bool).sum(sdim)
        numps.append(nump)
        tps.append(tp)
    return numl, numps, tps
        
        
        
    
    
    