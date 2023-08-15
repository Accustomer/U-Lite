import os
import yaml
import shutil
import torch
import random
import argparse
import warnings
from torch import nn
from tqdm import tqdm
from torch.backends import cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from utils import *


parser = argparse.ArgumentParser(description='ULite Sementic Segmentation Training')
parser.add_argument('--data', type=str, default='data', help='Data directory, train.txt and valid.txt are needed here.')
parser.add_argument('--weights', type=str, default="", help='weights path')
parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[height, width] image sizes')
parser.add_argument('--cfg', type=str, default='data/param.yaml', help='Model configuration')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=3, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', default=1, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--val-batch-size', default=1, type=int, metavar='V', help='validation batch size')
parser.add_argument('--adam', action='store_true', help='Apply Adam optimizer')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--seed', default=8888, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', type=str, default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
parser.add_argument('--save-dir', type=str, default='runs/exp', help='Directory for saving results')


def train_one_epoch(train_loader, model, criterions, optimizer, scaler, epoch, args, device):
    model.train()
    # Loss
    avg_cls_loss = torch.zeros(1, device=device)
    avg_dice_loss = torch.zeros(1, device=device)
    avg_total_loss = torch.zeros(1, device=device)
    # Performance 
    avg_map50 = torch.zeros(1, device=device)
    avg_map50_95 = torch.zeros(1, device=device)
    # Show results
    show_strs = ('Train', 'epoch', 'gpu_mem', 'cls_loss', 'dice_loss', 'total_loss', 'mAP@50', 'mAP@50-95')
    mlen = max([len(s) for s in show_strs]) + 1
    print((f"%{mlen}s" * len(show_strs)) % show_strs)
    # Iteration
    pbar = tqdm(enumerate(train_loader))
    for i, (image, masks) in pbar:
        image = image.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).float()
        # dists = dists.to(device, non_blocking=True).float()
        
        # Compute output and loss
        out = model(image)
        pred = out.sigmoid()
        cls_loss = criterions[0](pred, masks)
        dice_loss = criterions[1](pred, masks)
        # mse_loss = criterions[2](out, dists)
        total_loss = cls_loss + dice_loss   #  0.02 * mse_loss
        
        # Compute gradient and do optimizer step
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)  # optimizer.step
        scaler.update()
        
        # Compute measurement
        map50, map50_95 = calc_map(pred, masks)
        
        # Average
        avg_cls_loss = (avg_cls_loss * i + cls_loss) / (i + 1)
        avg_dice_loss = (avg_dice_loss * i + dice_loss) / (i + 1)
        avg_total_loss = (avg_total_loss * i + total_loss) / (i + 1)
        avg_map50 = (avg_map50 * i + map50) / (i + 1)
        avg_map50_95 = (avg_map50_95 * i + map50_95) / (i + 1)
        
        # Show results
        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
        s = (f'%{mlen}s' * 3 + f'%{mlen}.4g' * 5) % ('', epoch, mem, *avg_cls_loss, *avg_dice_loss, *avg_total_loss, *avg_map50, *avg_map50_95)
        pbar.set_description(s)
    pbar.close()    
    
    ret = (avg_cls_loss.detach().cpu().item(), 
           avg_dice_loss.detach().cpu().item(),
           avg_total_loss.detach().cpu().item(),
           avg_map50.detach().cpu().item(),
           avg_map50_95.detach().cpu().item()
           )
    return ret
        

def valid_one_epoch(valid_loader, model, criterions, epoch, device):
    with torch.no_grad():
        model.eval()
        # Loss
        avg_cls_loss = torch.zeros(1, device=device)
        avg_dice_loss = torch.zeros(1, device=device)
        avg_total_loss = torch.zeros(1, device=device)
        # Performance 
        avg_map50 = torch.zeros(1, device=device)
        avg_map50_95 = torch.zeros(1, device=device) 
        # Show results
        show_strs = ('Valid', 'epoch', 'gpu_mem', 'cls_loss', 'dice_loss', 'total_loss', 'mAP@50', 'mAP@50-95')
        mlen = max([len(s) for s in show_strs]) + 1
        print((f"%{mlen}s" * len(show_strs)) % show_strs)
        # Iteration
        pbar = tqdm(enumerate(valid_loader))
        for i, (image, masks) in pbar:
            image = image.to(device, non_blocking=True).float()
            masks = masks.to(device, non_blocking=True).float()
            # dists = dists.to(device, non_blocking=True).float()
            
            # Compute output and loss
            out = model(image)
            pred = out.sigmoid()
            cls_loss = criterions[0](pred, masks)
            dice_loss = criterions[1](pred, masks)
            # mse_loss = criterions[2](out, dists)
            # total_loss = 0.49 * cls_loss + 0.49 * dice_loss + 0.02 * mse_loss
            total_loss = cls_loss + dice_loss
                       
            # Compute measurement
            map50, map50_95 = calc_map(pred, masks)
            
            # Average
            avg_cls_loss = (avg_cls_loss * i + cls_loss) / (i + 1)
            avg_dice_loss = (avg_dice_loss * i + dice_loss) / (i + 1)
            avg_total_loss = (avg_total_loss * i + total_loss) / (i + 1)
            avg_map50 = (avg_map50 * i + map50) / (i + 1)
            avg_map50_95 = (avg_map50_95 * i + map50_95) / (i + 1)
            
            # Show results
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = (f'%{mlen}s' * 3 + f'%{mlen}.4g' * 5) % ('', epoch, mem, *avg_cls_loss, *avg_dice_loss, *avg_total_loss, *avg_map50, *avg_map50_95)
            pbar.set_description(s)
        pbar.close()    
        
        ret = (avg_cls_loss.detach().cpu().item(), 
               avg_dice_loss.detach().cpu().item(),
               avg_total_loss.detach().cpu().item(),
               avg_map50.detach().cpu().item(),
               avg_map50_95.detach().cpu().item()
               )
        return ret
    
    
def evaluate(test_loader, model, criterions, device, classes, save_dir, n=201):
    with torch.no_grad():
        model.eval()
        # Loss
        avg_cls_loss = torch.zeros(1, device=device)
        avg_dice_loss = torch.zeros(1, device=device)
        avg_total_loss = torch.zeros(1, device=device)
        # Performance 
        avg_map50 = torch.zeros(1, device=device)
        avg_map50_95 = torch.zeros(1, device=device) 

        prob_arange = torch.linspace(0, 1, n)        
        total_numl, total_numps, total_tps = 0, [], []

        # Show results
        show_strs = ('Test', 'gpu_mem', 'cls_loss', 'dice_loss', 'total_loss', 'mAP@50', 'mAP@50-95')
        mlen = max([len(s) for s in show_strs]) + 1
        print((f"%{mlen}s" * len(show_strs)) % show_strs)
        # Iteration
        pbar = tqdm(enumerate(test_loader))      
        for i, (image, masks) in pbar:
            image = image.to(device, non_blocking=True).float()
            masks = masks.to(device, non_blocking=True).float()
            # dists = dists.to(device, non_blocking=True).float()
            
            # Compute output and loss
            out = model(image)
            pred = out.sigmoid()
            cls_loss = criterions[0](pred, masks)
            dice_loss = criterions[1](pred, masks)
            # mse_loss = criterions[2](out, dists)
            # total_loss = 0.49 * cls_loss + 0.49 * dice_loss + 0.02 * mse_loss
            total_loss = cls_loss + dice_loss
                       
            # Compute measurement
            map50, map50_95 = calc_map(pred, masks)            
            # pres, recs, f1s = calc_prf(pred, masks, prob_arange)
            numl, numps, tps = calc_metrics_item(pred, masks, prob_arange)
            
            # Average
            avg_cls_loss = (avg_cls_loss * i + cls_loss) / (i + 1)
            avg_dice_loss = (avg_dice_loss * i + dice_loss) / (i + 1)
            avg_total_loss = (avg_total_loss * i + total_loss) / (i + 1)
            avg_map50 = (avg_map50 * i + map50) / (i + 1)
            avg_map50_95 = (avg_map50_95 * i + map50_95) / (i + 1)
            # avg_precision = (avg_precision * i + pres) / (i + 1)
            # avg_recall = (avg_recall * i + recs) / (i + 1)
            # avg_f1 = (avg_f1 * i + f1s) / (i + 1)
            if i == 0:
                total_numl, total_numps, total_tps = numl, numps, tps
            else:
                total_numl += numl
                total_numps = [total_numps[j] + numps[j] for j in range(n)]
                total_tps = [total_tps[j] + tps[j] for j in range(n)]                
            
            # Show results
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = (f'%{mlen}s' * 2 + f'%{mlen}.4g' * 5) % ('', mem, *avg_cls_loss, *avg_dice_loss, *avg_total_loss, *avg_map50, *avg_map50_95)
            pbar.set_description(s)
        pbar.close()    
        
        eps = 1e-6        
        recalls = torch.vstack([total_tps[i] / (total_numl + eps) for i in range(n)])
        f1s = torch.vstack([total_tps[i] * 2 / (total_numl + total_numps[i] + eps) for i in range(n)])
        precisions = torch.vstack([(total_tps[i] / (total_numps[i] + eps) if total_numps[i] > 0 else torch.ones_like(numl, device=device)) for i in range(n)])
        recalls = recalls.cpu().numpy()
        f1s = f1s.cpu().numpy()
        precisions = precisions.cpu().numpy()
        prob_arange = prob_arange.cpu().numpy()

        plot_pr_curve(rec=recalls, pre=precisions, save_path='/'.join((save_dir, "PR_curve.png")), names=classes)
        plot_mc_curve(px=prob_arange, py=precisions, save_path='/'.join((save_dir, "P_curve.png")), names=classes, ylabel="Precision")
        plot_mc_curve(px=prob_arange, py=recalls, save_path='/'.join((save_dir, "R_curve.png")), names=classes, ylabel="Recall")
        plot_mc_curve(px=prob_arange, py=f1s, save_path='/'.join((save_dir, "F1_curve.png")), names=classes, ylabel="F1 Score")
    
    
def main_worker():
    args = parser.parse_args()
    print(args)
    
    # Save directory
    if not check_savedir(args.save_dir):
        print("Please change a new directory to save results.")
        return
    
    # Save arguments
    args_file = '/'.join((args.save_dir, 'args.yaml'))
    with open(args_file, 'w') as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)
    
    # Save configure
    cfg_file = '/'.join((args.save_dir, 'cfg.yaml'))
    shutil.copy(args.cfg, cfg_file)
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Check traning and validation settings file
    train_path = '/'.join((args.data, 'train.txt'))
    valid_path = '/'.join((args.data, 'valid.txt'))
    class_path = '/'.join((args.data, 'classes.txt'))
    if not (os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(class_path)):
        print('train.txt, valid.txt and classes.txt are needed!')
        return
    with open(class_path, 'r') as f:
        classes = f.read().strip().splitlines()
    if len(classes) != cfg["num_classes"]:
        warnings.warn("The number of classes in classes.txt and configuration are not equal!")
        if cfg["num_classes"] == 1:
            classes = ("Total", )
            warnings.warn("Use single class mode!")
        else:
            return

    # Random seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
    # Device
    if args.gpu is not None and args.gpu.lower() != 'cpu':
        device = torch.device('cuda')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    else:
        device = torch.device('cpu')
        warnings.warn('Using cpu for training.')
        
    # Load model
    model = load_model(args.weights, device, cfg)
    model.to(device)
    
    # Parallel
    ngpu = len(args.gpu.split(','))
    if ngpu > 1:
        print(f"{ngpu} GPUs, use data parallel")
        model = nn.DataParallel(model)        
        
    # Loss
    criterions = (
        # nn.CrossEntropyLoss() if cfg['num_classes'] > 2 else nn.BCEWithLogitsLoss(),
        nn.BCELoss(), 
        DiceLoss(cfg['num_classes'], False), 
        # nn.MSELoss()
    )
    
    # Load data
    _, test_loader = get_yoloval_loader(cfg, args, "test.txt", False)

    # Validation ?
    if args.evaluate:
        evaluate(test_loader, model, criterions, device, classes, args.save_dir)
        return
    
    train_data, train_loader = get_yolotrain_loader(cfg, args)
    _, valid_loader = get_yoloval_loader(cfg, args, "valid.txt", True)
    ns = len(train_data)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr0'], betas=(cfg['momentum'], 0.999), weight_decay=cfg['weight_decay']) if args.adam else \
                torch.optim.SGD(model.parameters(), lr=cfg['lr0'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs * ns // args.batch_size // ngpu)
    # lr_scheduler = ReduceLROnPlateau(optimizer, 'min')
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    # Training 
    best_loss = 1e6
    log_file = '/'.join((args.save_dir, 'result.csv'))
    metrics = ("train/cls_loss", "train/dice_loss", "train/total_loss", "train/mAP@50", "train/mAP@50-95", 
               "valid/cls_loss", "valid/dice_loss", "valid/total_loss", "valid/mAP@50", "valid/mAP@50-95")
    with open(log_file, 'w') as f:
        f.write(",".join(metrics) + "\n")
    last_file = '/'.join((args.save_dir, 'last.pth.tar'))
    best_file = '/'.join((args.save_dir, 'best.pth.tar'))
    for epoch in range(args.epochs):
        train_result = train_one_epoch(train_loader, model, criterions, optimizer, scaler, epoch, args, device)        
        valid_result = valid_one_epoch(valid_loader, model, criterions, epoch, device)     
        total_loss = train_result[2] * 0.1 + valid_result[2] * 0.9
        lr_scheduler.step() # total_loss
        is_best = total_loss < best_loss
        best_loss = total_loss if is_best else best_loss
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict() if type(model) is nn.DataParallel else model.state_dict(),  
            'train_result': train_result,
            'valid_result': valid_result,
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
        }, is_best, filename=last_file, best_filename=best_file)
        with open(log_file, 'a') as f:
            print(','.join([str(v) for v in train_result + valid_result]), file=f)
    
    # Draw results
    show_file = '/'.join((args.save_dir, 'result.png'))
    plot_results(log_file, show_file)
    
    # Test best model
    model = load_model(best_file, device, cfg)
    model.to(device)
    evaluate(test_loader, model, criterions, device, classes, args.save_dir)

    # Release CUDA memory
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main_worker()

    
    