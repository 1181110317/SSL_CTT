import torch

def get_accuracy(sr,gt):
    corr=torch.sum(sr==gt)
    tensor_size=sr.size(0)*sr.size(1)
    acc=float(corr)/float(tensor_size)
    del corr
    return acc


def get_meaniou(segmentation_result, y, n_classes = 19):
    iou = []
    iou_sum = 0
    segmentation_result = segmentation_result.view(-1)
    y = y.view(-1)
    classes=torch.unique(y)
    #print(classes)

    for cls in range(1, n_classes):
        if cls not in classes:
            n_classes-=1
            continue
        result_inds = segmentation_result == cls
        y_inds = y == cls
        intersection = (result_inds[y_inds]).long().sum().data.cpu().item()
        union = result_inds.long().sum().data.cpu().item() + y_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            iou.append(float('nan'))
        else:
            iou.append(float(intersection) / float(max(union, 1)))
            iou_sum += float(intersection) / float(max(union, 1))
    #print(iou)
    del segmentation_result,y
    return iou_sum/n_classes