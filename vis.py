from torchviz import make_dot
import torch
import argparse
import yaml
from modules.model import AttHRnetPose
import os
import onnx

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=12, help='total batch size for all GPUs')
parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
parser.add_argument('--workers', type=int, default=4, help='dataloader worker')
parser.add_argument('--hyp', type=str, default='cfg/hyp.yaml', help='hyperparameters path')

parser.add_argument('--st_epochs', type=int, default=34)                                                         #set
parser.add_argument('--epochs', type=int, default=150)                                                          #set
parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')                      #set
parser.add_argument('--cfg', type=str, default='cfg/model_small1.yaml', help='model.yaml path')                  #set
parser.add_argument('--data', type=str, default='cfg/data_0523.yaml', help='data.yaml path')                     #set
parser.add_argument('--pretrained', type=str, default='/yq/RE/CAttHRnetPose/trains/train0523_0/last.pt', help='pretrained model')                              #set
parser.add_argument('--name', type=str, default='trains/train0523_0/', help='save pt and txt path')              #set
opt = parser.parse_args()


with open(opt.cfg) as f:
    opt.cfg = yaml.safe_load(f)
with open(opt.hyp) as f:
    opt.hyp = yaml.safe_load(f)


if opt.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
else:  # non-cpu device requested
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device  # set environment variable
    assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {opt.device} requested'  # check availability

device = torch.device('cpu') if opt.device == 'cpu' else torch.device('cuda')
model = AttHRnetPose(opt.cfg['MODEL']).to(device)

'''if os.path.isfile(opt.pretrained):
    model.load_state_dict(torch.load(opt.pretrained))
    print('load pretrained model')'''

model.eval()

 
input_names = ['input']
output_names = ['output']
 
x = torch.randn(1,4,640,640,requires_grad=True).to(device)
 
torch.onnx.export(model, x, 'last.onnx', input_names=input_names, output_names=output_names, verbose='True')