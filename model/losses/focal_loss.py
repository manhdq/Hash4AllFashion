import torch
import torch.nn.functional as F
from torch.autograd import Variable


##TODO: Make this dynamic, optional for chosen and hyperparameters
# self.focal_loss = FocalLoss(gamma=2, alpha=0.25, size_average=True)
def focal_loss(input, target, gamma=3.5, alpha=0.3, size_average=True):
    if isinstance(alpha,(float,int)): alpha = torch.Tensor([alpha,1-alpha])
    if isinstance(alpha,list): alpha = torch.Tensor(alpha)

    if input.dim()>2:
        input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
    target = target.view(-1,1)

    logpt = F.log_softmax(input)
    logpt = logpt.gather(1,target)
    logpt = logpt.view(-1)
    pt = Variable(logpt.data.exp())

    if alpha is not None:
        if alpha.type()!=input.data.type():
            alpha = alpha.type_as(input.data)
        at = alpha.gather(0,target.data.view(-1))
        logpt = logpt * Variable(at)

    loss = -1 * (1-pt)**gamma * logpt
    if size_average: return loss.mean()
    else: return loss.sum()