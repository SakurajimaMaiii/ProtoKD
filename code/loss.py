# loss function for training
import torch
from torch import nn, Tensor
import torch.nn.functional as F


def dice_loss(predict, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(predict * target)
    dice = (2 * intersect + smooth) / (
        torch.sum(target * target) + torch.sum(predict * predict) + smooth
    )
    loss = 1.0 - dice
    return loss


def dice_loss1(predict, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(predict * target)
    dice = (2 * intersect + smooth) / (torch.sum(target) + torch.sum(predict) + smooth)
    loss = 1.0 - dice
    return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def one_hot_encode(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            tmp = (input_tensor == i) * torch.ones_like(input_tensor)
            tensor_list.append(tmp)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, input, target, weight=None, softmax=True):
        if softmax:
            inputs = F.softmax(input, dim=1)
        target = self.one_hot_encode(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.shape == target.shape, "size must match"
        class_wise_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            diceloss = dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(diceloss)
            loss += diceloss * weight[i]
        loss = loss / self.n_classes
        return loss


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.eps = 1e-4
        self.num_classes = num_classes

    def forward(self, predict, target):
        weight = []
        for c in range(self.num_classes):
            weight_c = torch.sum(target == c).float()
            # print("weightc for c",c,weight_c)
            weight.append(weight_c)
        weight = torch.tensor(weight).to(target.device)
        weight = 1 - weight / (torch.sum(weight))
        if len(target.shape) == len(predict.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        wce_loss = F.cross_entropy(predict, target.long(), weight)
        return wce_loss


def softmax_dice_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


class DiceCeLoss(nn.Module):
    # predict : output of model (i.e. no softmax)[N,C,*]
    # target : gt of img [N,1,*]
    def __init__(self, num_classes, alpha=1.0, weighted=True):
        """
        calculate loss:
            celoss + alpha*celoss
            alpha : default is 1
        """
        super().__init__()
        self.alpha = alpha
        self.num_classes = num_classes
        self.diceloss = DiceLoss(self.num_classes)
        if weighted:
            self.celoss = WeightedCrossEntropyLoss(self.num_classes)
        else:
            self.celoss = RobustCrossEntropyLoss()

    def forward(self, predict, label):
        # predict is output of the model, i.e. without softmax [N,C,*]
        # label is not one hot encoding [N,1,*]

        diceloss = self.diceloss(predict, label)
        celoss = self.celoss(predict, label)
        loss = celoss + self.alpha * diceloss
        return diceloss, celoss, loss


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction="none")
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2) ** 2)


def prototype_loss(feature, feature_t, label, num_cls):
    """
    match prototype simarlity map for teacher and student model
    feature/feature_t [N,C,*]
    label [N,1,*]
    """
    eps = 1e-5
    N = len(feature.size()) - 2
    label = label[:, 0]  # label [N,*]
    s = []
    t = []

    for i in range(num_cls):
        mask = label == i
        if N == 3 and (torch.sum(mask, dim=(-3, -2, -1)) > 0).all():
            proto_s = torch.sum(feature * mask[:, None], dim=(-3, -2, -1)) / (
                torch.sum(mask[:, None], dim=(-3, -2, -1)) + eps
            )
            proto_t = torch.sum(feature_t * mask[:, None], dim=(-3, -2, -1)) / (
                torch.sum(mask[:, None], dim=(-3, -2, -1)) + eps
            )
            proto_map_s = F.cosine_similarity(
                feature, proto_s[:, :, None, None, None], dim=1, eps=eps
            )
            proto_map_t = F.cosine_similarity(
                feature_t, proto_t[:, :, None, None, None], dim=1, eps=eps
            )
            s.append(proto_map_s.unsqueeze(1))
            t.append(proto_map_t.unsqueeze(1))
    sim_map_s = torch.cat(s, dim=1)
    sim_map_t = torch.cat(t, dim=1)
    loss = torch.mean((sim_map_s - sim_map_t) ** 2)
    return sim_map_s, sim_map_t, loss
