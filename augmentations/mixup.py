import numpy as np

def mixup(data, targets, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = np.random.permutation(len(data))
    return lam * data + (1 - lam) * data[index], lam * targets + (1 - lam) * targets[index]