import torch
import torch.nn as nn
import numpy as np 


class RefineLoss(nn.Module):
    def __init__(self,alpha=1.5,alpha1=0.5,reduction="mean"):
        super(RefineLoss,self).__init__()
        self.alpha = alpha
        self.alpha1 = alpha1
        self.reduction = reduction
        self.fx = nn.Conv2d(1,1,3,padding=1,bias=False)
        self.fy = nn.Conv2d(1,1,3,padding=1,bias=False)

        ngx = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]],dtype=np.float32)
        ngy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=np.float32)

        self.fx.weight.data.copy_(torch.from_numpy(ngx))
        self.fy.weight.data.copy_(torch.from_numpy(ngy))

        for param in self.fx.parameters():
            param.requires_grad = False
        for param in self.fy.parameters():
            param.requires_grad = False
        
    def forward(self,grayimg,pred,mask):
        '''
        grayimg: gray scale input image
        pred: predicted mask
        mask: boundary mask. can be generate from ground truth foreground mask by  morphological transformation  
        '''
        gx = self.fx(grayimg)
        gy = self.fy(grayimg)

        px = self.fx(pred)
        py = self.fy(pred)
        
        gm = torch.sqrt(gx*gx + gy*gy+1e-6)
        pm = torch.sqrt(px*px + py*py+1e-6)
        
        gv = (gx/gm, gy/gm)
        pv = (px/pm, py/pm)


        Lcos = (1 - torch.abs(gv[0]*pv[0] + gv[1]*pv[1]))*pm 
        Lmag = torch.clamp_min(self.alpha*gm-pm, 0)

        Lrefine = (self.alpha1*Lcos + (1-self.alpha1)*Lmag) * mask

        if self.reduction == "mean":
            Lrefine = Lrefine.mean()
        elif self.reduction == "sum":
            Lrefine = Lrefine.sum()
        
        return Lrefine

bce_loss = nn.BCELoss(reduction=reduction_type) # size_average
refine_loss = RefineLoss(reduction=reduction_type)

def ba_loss(pred,target,ba,mask,grayimg):
    '''
    grayimg: gray scale input image
    pred: predicted mask
    mask: boundary mask. can be generate from ground truth foreground mask by  morphological transformation  
    ba: predicted boundary attention
    '''
    alpha,beta,gamma = 0.6, 0.3, 0.1
    Lbound = bce_loss(ba, mask)
    Lseg = bce_loss(pred, target)
    Lrefine = refine_loss(grayimg, pred, trimap)
    return alpha*Lseg + beta*Lbound + gamma*Lrefine
