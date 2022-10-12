import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class CrossEntropy2d(nn.Module):

    def __init__(self, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, reduction='mean')
        return loss


class CtsEntropy2d(nn.Module):

    def __init__(self, ignore_label=255):
        super(CtsEntropy2d, self).__init__()
        self.ignore_label = ignore_label


    def rce(self, pred, labels):
        softmax_pred = F.softmax(pred, dim=1)
        softmax_pred = torch.clamp(softmax_pred, min=1e-7, max=1.0)
        mask = (labels != self.ignore_label).float()
        class_numbers=softmax_pred.size(1)
        labels[labels==self.ignore_label] = class_numbers
        label_one_hot = torch.nn.functional.one_hot(labels, class_numbers + 1).float().cuda()
        label_one_hot = torch.clamp(label_one_hot.permute(0,3,1,2)[:,:-1,:,:], min=1e-4, max=1.0)
        rce = -(torch.sum(softmax_pred * torch.log(label_one_hot), dim=1) * mask).sum() / (mask.sum() + 1e-6)
        return rce

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3

        loss1 = F.cross_entropy(predict, target, weight=weight,size_average=True, ignore_index=self.ignore_label, reduction='mean')
        #loss2 = self.rce(predict, target.clone())

        #loss = 0.1 * loss1 + loss2
        return loss1


class CrossEntropyLoss2dPixelWiseWeighted(nn.Module):
    def __init__(self, weight=None, ignore_index=250, reduction='none'):
        super(CrossEntropyLoss2dPixelWiseWeighted, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target, pixelWiseWeight):
        loss = self.CE(output, target)
        loss = torch.mean(loss * pixelWiseWeight)
        return loss

class ContrLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=250, reduction='none'):
        super(ContrLoss, self).__init__()

    def forward(self,memory_bank, pred_rep, labels, mask, which_memory, num_classes, temp=0.5):
        b, num_feat, w, j = pred_rep.shape


        # 到这一步就有对应的labels和rep，每个像素都对应起来
        pred_rep = pred_rep.permute(0, 2, 3, 1)
        loss = torch.nn.DataParallel(torch.tensor(0.0),device_ids=[0,1]).cuda()
        class_count = 0

        size = int(memory_bank.size(1) / 2)

        eps = 1e-12

        for i in range(num_classes):
            # print(pred_rep.shape)

            per_count_size = 0
            with torch.no_grad():
                negative_feat = memory_bank[[j for j in range(num_classes) if j != i]].view(-1, num_feat)

            loss_pixel = torch.nn.DataParallel(torch.tensor(0.0),device_ids=[0,1]).cuda()

            which_choice = (mask & (labels == i))
            if which_choice.sum() == 0:
                continue
            class_count += 1

            for which_net in range(2):

                pred_rep_i = pred_rep[which_choice & (which_memory == (1 - which_net))]

                if len(pred_rep_i) == 0:
                    continue
                per_count = len(pred_rep_i)
                per_count_size += per_count

                with torch.no_grad():
                    if which_net == 0:
                        positive_feat = memory_bank[i, :size, :]
                    else:
                        positive_feat = memory_bank[i, size:, :]
                    all_feats = torch.cat([positive_feat, negative_feat])

                logits = ((pred_rep_i @ all_feats.T) / (
                            torch.norm(pred_rep_i, dim=1).unsqueeze(0).T @ torch.norm(all_feats, dim=1).unsqueeze(0)))
                logits = logits / temp
                logits = torch.exp(logits)

                logits_down = torch.sum(logits[:, size:], dim=1).unsqueeze(1)
                logits_posi = logits[:, :size]
                loss_pixel += (-torch.log((logits_posi / (logits_posi + logits_down + eps)) + eps)).sum()

            loss += (loss_pixel / (size * per_count_size))

        del mask, labels, which_memory

        return loss / class_count

class MSELoss2d(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean', ignore_index=255):
        super(MSELoss2d, self).__init__()
        self.MSE = nn.MSELoss(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, output, target):
        loss = self.MSE(torch.softmax(output, dim=1), target)
        return loss

def customsoftmax(inp, multihotmask):
    """
    Custom Softmax
    """
    soft = torch.softmax(inp,dim=1)
    # This takes the mask * softmax ( sums it up hence summing up the classes in border
    # then takes of summed up version vs no summed version
    return torch.log(
        torch.max(soft, (multihotmask * (soft * multihotmask).sum(1, keepdim=True)))
    )

class ImgWtLossSoftNLL(nn.Module):
    """
    Relax Loss
    """

    def __init__(self, classes, ignore_index=255, weights=None, upper_bound=1.0,
                 norm=False):
        super(ImgWtLossSoftNLL, self).__init__()
        self.weights = weights
        self.num_classes = classes
        self.ignore_index = ignore_index
        self.upper_bound = upper_bound
        self.norm = norm
        self.batch_weights = False
        self.fp16 = False


    def calculate_weights(self, target):
        """
        Calculate weights of the classes based on training crop
        """
        if len(target.shape) == 3:
            hist = np.sum(target, axis=(1, 2)) * 1.0 / target.sum()
        else:
            hist = np.sum(target, axis=(0, 2, 3)) * 1.0 / target.sum()
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist[:]

    def custom_nll(self, inputs, target, class_weights, border_weights, mask):
        """
        NLL Relaxed Loss Implementation
        """
        #if (cfg.REDUCE_BORDER_EPOCH != -1 and cfg.EPOCH > cfg.REDUCE_BORDER_EPOCH):
        #    border_weights = 1 / border_weights
        #    target[target > 1] = 1
        if self.fp16:
            loss_matrix = (-1 / border_weights *
                           (target[:, :, :, :].half() *
                            class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) *
                            customsoftmax(inputs, target[:, :, :, :].half())).sum(1)) * \
                          (1. - mask.half())
        else:
            loss_matrix = (-1 / border_weights *
                           (target[:, :, :, :].float() *
                            class_weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) *
                            customsoftmax(inputs, target[:, :, :, :].float())).sum(1)) * \
                          (1. - mask.float())

            # loss_matrix[border_weights > 1] = 0
        loss = loss_matrix.sum()

        # +1 to prevent division by 0
        loss = loss / (target.shape[0] * target.shape[2] * target.shape[3] - mask.sum().item() + 1)
        return loss

    def forward(self, inputs, target):
        if self.fp16:
            weights = target[:, :, :, :].sum(1).half()
        else:
            weights = target[:, :, :, :].sum(1).float()
        ignore_mask = (weights == 0)
        weights[ignore_mask] = 1

        loss = 0
        target_cpu = target.data.cpu().numpy()

        if self.batch_weights:
            class_weights = self.calculate_weights(target_cpu)

        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                class_weights = self.calculate_weights(target_cpu[i])

            class_weights = torch.ones((class_weights.shape))
            loss = loss + self.custom_nll(inputs[i].unsqueeze(0),
                                          target[i].unsqueeze(0),
                                          class_weights=torch.Tensor(class_weights).cuda(),
                                          border_weights=weights, mask=ignore_mask[i])

        return loss
