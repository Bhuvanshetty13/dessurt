import numpy as np


def my_metric(y_input, y_target):
    assert len(y_input) == len(y_target)
    correct = 0
    for y0, y1 in zip(y_input, y_target):
        if np.array_equal(y0, y1):
            correct += 1
    return correct / len(y_input)


def meanIOU(y_output, y_target):
    assert len(y_output) == len(y_target)
    epsilon = 0.001
    iouSum = 0
    for out, targ in zip(y_output, y_target):
        binary = out>0 #torch.where(out>0,1,0)
        #binary = torch.round(y_input) #threshold at 0.5
        intersection = (binary * targ).sum()
        union = (binary + targ).sum() - intersection
        iouSum += float(intersection) / (union+epsilon)
    return iouSum / float(len(y_output))
