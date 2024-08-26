from modules.layers import *
import yaml


class ImplicitM(nn.Module):
    def __init__(self, channel):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=1., std=.02)        
        
    def forward(self, x):
        return self.implicit * x 

class ImplicitA(nn.Module):
    def __init__(self, channel):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, std=.02)

    def forward(self, x):
        return self.implicit + x
    


class BoxKeyDetect(nn.Module):
    def __init__(self, cfg):
        super(BoxKeyDetect,self).__init__()
        self.imgsz = cfg['IMG_SZ']
        self.nc = cfg['NC'] #number of classes
        self.nkpt = cfg['NUM_KEYPOINTS'] #number of keypoints   
        self.reg_max = cfg['REG'] * 4
        self.no_det=(1 + self.reg_max)  # number of outputs per anchor for box and class, (nc+5): nc confidence , class, x, y ,w ,h
        self.no_kpt = 3*self.nkpt ## number of outputs per anchor for keypoints
        self.no = self.no_det + self.no_kpt
        self.start = cfg['START_LAYER']
        self.end = cfg['END_LAYER']
        self.nl = self.end - self.start
        self.na = cfg['ANCHORS_NUM']
        self.training = True
        '''output layers'''
        self.ch = list(cfg['STAGE'].values())[-1]['NUM_CHANNELS']
        self.m_kpt = nn.ModuleList(
                            nn.Sequential(DWConv(x, x, k=3), Conv(x,x),
                                          DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), Conv(x,x),
                                          DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), Conv(x, x),
                                          DWConv(x, x, k=3), nn.Conv2d(x, self.no_kpt * self.na, 1)) for x in self.ch[self.start:self.end])
        c2 = 16
        c3 = 16
        self.m_box = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, self.reg_max, 1)) for x in self.ch[self.start:self.end])
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, 1, 1)) for x in self.ch[self.start:self.end])
        self.ia2 = nn.ModuleList(ImplicitA(x) for x in self.ch[self.start:self.end])
        self.im2 = nn.ModuleList(ImplicitM(self.reg_max) for _ in self.ch[self.start:self.end])
        self.im3 = nn.ModuleList(ImplicitM(1) for _ in self.ch[self.start:self.end])
        self.ia3 = nn.ModuleList(ImplicitA(x) for x in self.ch[self.start:self.end])



    def forward(self,x):
        shape = x[0].shape
        for i in range(self.nl):
            box = self.im2[i](self.m_box[i](self.ia2[i](x[i])))
            conf = self.im3[i](self.cv3[i](self.ia3[i](x[i])))
            kpt = self.m_kpt[i](x[i])
            x[i] = torch.cat((box,conf,kpt), axis=1) #x[i].shape  torch.Size([2, 117, 160, 160])
        box, confidence, kpt = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max, 1,self.no_kpt), 1) #box xyxy:16,confidence:1,nc,kpt
        if self.training:
            return x, box, confidence, kpt
        else:
            predkpt = self.process_kpt(x,shape)
            return x, box, confidence,  kpt, predkpt


    def process_kpt(self, x, shape,offset = 0.5):
        def make_grid(nx=20, ny=20):
            yv, xv = torch.meshgrid([torch.arange(ny)+0.5, torch.arange(nx)+0.5])
            return torch.stack((xv, yv), 2).view((1,ny*nx, 2)).float()
        all_kpt = []
        for xi in x:
            _, _, pkpt = xi.view(shape[0], self.no, -1).split((self.reg_max, 1, self.no_kpt), 1) #box xyxy:16,confidence:1,nc,kpt
            pkpt = pkpt.permute(0,2,1)  # bs, w*h, number of output
            bs, _, ny, nx = xi.shape
            stride = self.imgsz/ny
            grid = make_grid(nx, ny).to(xi.device)
            kpt_grid_x = grid[..., 0:1]
            kpt_grid_y = grid[..., 1:2]
            
            pkpt[..., 0::3] = (pkpt[..., ::3] * 2. - 0.5 + kpt_grid_x.repeat(1,1,1,1,17)) * stride 
            pkpt[..., 1::3] = (pkpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(1,1,1,1,17)) * stride 
            pkpt[..., 2::3] = pkpt[..., 2::3].sigmoid()  
            all_kpt.append(pkpt)
        return torch.cat(all_kpt, dim = 1)   #640 scale

   

class AttHRnetPose(nn.Module):
    def __init__(self,cfg):
        super(AttHRnetPose,self).__init__()
        self.backbone = AttHRNet(cfg)
        self.detect = BoxKeyDetect(cfg)
    
    def forward(self,x):
        y_list = self.backbone(x)
        y_list = self.detect(y_list)
        return y_list
    def set_inference(self):
        self.detect.training = False
    def set_train(self):
        self.detect.training = True

if __name__ == '__main__':
    with open('C:/Users/PC/Desktop/radarmodel/CAttHRnetPose/cfg/model.yaml') as f:
        cfg = yaml.safe_load(f)
    model = AttHRnetPose(cfg['MODEL'])

    input=torch.randn(2,4,640,480)

    out = model(input) #[2, 57, 320, 240]  [2, 57, 160, 120]
    print('')