import torch
import numpy as np
import os
from transformers.models.electra.modeling_electra import ElectraForPreTrainingOutput

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def pearson_correlation(y_true, y_pred):
    # y_pred가 ElectraForPreTrainingOutput 객체인 경우 logits를 추출
    if isinstance(y_pred, ElectraForPreTrainingOutput):
        y_pred = y_pred.logits  # 또는 y_pred[0]로 접근 가능

    # y_true가 ElectraForPreTrainingOutput 객체인 경우, logits를 사용
    if isinstance(y_true, ElectraForPreTrainingOutput):
        y_true = y_true.logits  # 또는 y_true[0]로 접근

    # y_true와 y_pred가 텐서인지 확인
    if not isinstance(y_true, torch.Tensor):
        raise ValueError("y_true should be a tensor")
    if not isinstance(y_pred, torch.Tensor):
        raise ValueError("y_pred should be a tensor")

    # y_true와 y_pred가 같은 모양인지 확인하고, 아니라면 변환
    if y_true.shape != y_pred.shape:
        # y_true가 (batch_size, sequence_length)인 경우
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true = torch.argmax(y_true, dim=1)  # y_true에서 가장 높은 값을 가진 클래스 인덱스를 선택
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = torch.argmax(y_pred, dim=1)  # y_pred에서 가장 높은 값을 가진 클래스 인덱스를 선택

    # 텐서를 numpy 배열로 변환
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    return np.corrcoef(y_true, y_pred)[0, 1]


