import torch


def contrastive_loss(margin, im, s):
    size, dim = im.shape
    scores = im.matmul(s.t()) / dim
    diag = scores.diag()
    zeros = torch.zeros_like(scores)
    # shape #item x #item
    # sum along the row to get the VSE loss from each image
    cost_im = torch.max(zeros, margin - diag.view(-1, 1) + scores)
    # sum along the column to get the VSE loss from each sentence
    cost_s = torch.max(zeros, margin - diag.view(1, -1) + scores)
    # to fit parallel, only compute the average for each item
    vse_loss = cost_im.sum(dim=1) + cost_s.sum(dim=0) - 2 * margin
    # for data parallel, reshape to (size, 1)
    return vse_loss / (size - 1)