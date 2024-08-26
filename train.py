import argparse
import os 
import yaml
import torch.distributed as dist
import torch
from modules.model import AttHRnetPose
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from common.dataset import LoadImagesAndLabels
from tqdm import tqdm
from torch.cuda import amp
from common.loss import ComputeLoss
import logging
import warnings
from test import test
from torch.utils.data import random_split
import torch.optim.lr_scheduler as lr_scheduler
from common.general import one_cycle
from shutil import copyfile
 
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SEED = 13


#nohup python train.py >/dev/null 2>&1 &

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=18, help='total batch size for all GPUs')    #set
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='dataloader worker')
    parser.add_argument('--hyp', type=str, default='cfg/hyp.yaml', help='hyperparameters path')

    parser.add_argument('--st_epochs', type=int, default=11)                                                         #set
    parser.add_argument('--epochs', type=int, default=30)                                                          #set
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')                      #set
    parser.add_argument('--cfg', type=str, default='cfg/model_small3.yaml', help='model.yaml path')                  #set
    parser.add_argument('--data', type=str, default='cfg/data_06.yaml', help='data.yaml path')                     #set
    parser.add_argument('--pretrained', type=str, default='trains/train0623/last10.pt', help='pretrained model')     #set
    parser.add_argument('--name', type=str, default='trains/train0623/', help='save pt and txt path')              #set
    opt = parser.parse_args()

    if not os.path.exists(opt.name):
        os.makedirs(opt.name)
    copyfile(opt.cfg, opt.name+opt.cfg.split('/')[-1])
    copyfile(opt.data, opt.name+opt.data.split('/')[-1])
    
    with open(opt.cfg) as f:
        opt.cfg = yaml.safe_load(f)
    with open(opt.hyp) as f:
        opt.hyp = yaml.safe_load(f)
    

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
  

    if opt.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    else:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {opt.device} requested'  # check availability
    
    
    #model
    if opt.global_rank in [0,-1]: #not DDP  cpu or one GPU 
        device = torch.device('cpu') if opt.device == 'cpu' else torch.device('cuda')
        model = AttHRnetPose(opt.cfg['MODEL']).to(device)
        if os.path.isfile(opt.pretrained):
            model=torch.load(opt.pretrained)
            print('load pretrained model')
    else:
        opt.local_rank = int(os.environ['LOCAL_RANK'])
        os.environ[ 'MASTER_PORT'] = '12355'
        os.environ['MASTER_ADDR'] = 'localhost'
        torch.cuda.set_device(opt.local_rank)
        device = torch.device("cuda", opt.local_rank)
        dist.init_process_group(backend='nccl')
        model = AttHRnetPose(opt.cfg['MODEL']).to(device)
        if os.path.isfile(opt.pretrained):
            model=torch.load(opt.pretrained)
            print('load pretrained model')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) 
        model = DDP(model,device_ids=[opt.local_rank],output_device=[opt.local_rank], find_unused_parameters=True)
    
    cuda = device.type != 'cpu'
    scaler = amp.GradScaler(enabled=cuda)


    #data
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)  # data dict
    data_path = data_dict['path']
    cache_path = data_dict['cache_path']
    

    dataset = LoadImagesAndLabels(data_path, cache_path, img_size=640, hyp=opt.hyp)

    dataL = int(len(dataset))
    generator = torch.Generator().manual_seed(SEED)
    trainset, testset = random_split(
        dataset=dataset,
        lengths=[dataL - 10000, 10000],
        generator=generator
    )
    
    sampler = torch.utils.data.distributed.DistributedSampler(trainset) if opt.local_rank != -1 else None
    dataloader = torch.utils.data.DataLoader(trainset,
                                             batch_size=opt.batch_size,
                                             num_workers=opt.workers,
                                             sampler=sampler,
                                             pin_memory=True,
                                             collate_fn=LoadImagesAndLabels.collate_fn,
                                             shuffle = True if sampler is None else False)
    
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=1,
                                             num_workers=opt.workers,
                                             sampler=None,
                                             pin_memory=True,
                                             collate_fn=LoadImagesAndLabels.collate_fn,
                                             shuffle = False)

    #optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.hyp['lr0'], betas=(opt.hyp['momentum'], 0.999))
    lf = one_cycle(1, opt.hyp['lrf'], opt.st_epochs+opt.epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = opt.st_epochs - 1

    #loss
    compute_loss = ComputeLoss(opt,model)
    with open(opt.name+"results.txt",'a') as f:
        f.write(('%10s' * 8 +'\n') % ('Epoch', 'gpu_mem', 'box', 'dfl', 'kpt', 'kptv' ,'box conf','total'))
    for epoch in range(opt.st_epochs,opt.st_epochs+opt.epochs):
        model.train()
        mloss = torch.zeros(6, device=device)
        optimizer.zero_grad()
        pbar = enumerate(dataloader)
        if opt.local_rank in [-1, 0]:
            pbar = tqdm(pbar, total=len(dataloader))  # progress bar
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'dfl', 'kpt', 'kptv' ,'box conf','total'))  # box, dfl, kpt, kptv, confidence
        
        for i, (imgs, targets, paths, _) in pbar:
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            #with amp.autocast(enabled=cuda):
            pred = model(imgs)  # forward
            loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
            if opt.local_rank != -1:
                loss *= opt.world_size  # gradient averaged between devices in DDP mode

            scaler.scale(loss).backward()
            scaler.step(optimizer)  # optimizer.step
            scaler.update()
            optimizer.zero_grad()

            if opt.local_rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, opt.st_epochs+opt.epochs), mem, *mloss)
                pbar.set_description(s)
        with open(opt.name+"results.txt",'a') as f:
            f.write(s+'\n')
        #end batch
        scheduler.step()
        #test
        if epoch%5 ==0 and opt.global_rank in [-1, 0]:
            test(model, testloader, compute_loss=False, half_precision=False, plot = False, txtname = opt.name, epoch=epoch, doeval=True)


        #save
        if epoch%5 ==0:
            if opt.global_rank in [-1, 0]:
                torch.save(model, opt.name+"last{}.pt".format(epoch))
            elif dist.get_rank() == 0:
                torch.save(model.module,opt.name+"last{}.pt".format(epoch))
        else:
            if opt.global_rank in [-1, 0]:
                torch.save(model, opt.name+"last.pt")
            elif dist.get_rank() == 0:
                torch.save(model.module,opt.name+"last.pt")
    #end epoch
    test(model, testloader, compute_loss=False, half_precision=False, plot = False, txtname = opt.name, epoch=opt.st_epochs+opt.epochs, doeval=True)