import torch.nn.functional as F


def cross_entropy_loss(output, target):
    # logits를 추출
    logits = output.logits  # 또는 output[0]로 접근 가능
    return F.cross_entropy(logits, target)
