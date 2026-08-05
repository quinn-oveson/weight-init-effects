import torch
from torchvision import datasets, transforms
import torch.nn.functional as F

_mnist_cache = {}

def load_mnist_full(root='./mnist_data'):
    if root not in _mnist_cache:
        transform = transforms.ToTensor()
        mnist_trainset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        mnist_testset = datasets.MNIST(root=root, train=False, download=True, transform=transform)

        X_train_full = (mnist_trainset.data.float() / 255.0).reshape(len(mnist_trainset), -1)
        y_train_full = mnist_trainset.targets
        X_test = (mnist_testset.data.float() / 255.0).reshape(len(mnist_testset), -1)
        y_test = mnist_testset.targets

        _mnist_cache[root] = (X_train_full, y_train_full, X_test, y_test)
    return _mnist_cache[root]

def load_mnist_subset(n, seed, root='./mnist_data'):
    X_train_full, y_train_full, X_test, y_test = load_mnist_full(root)

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X_train_full.shape[0], generator=g)[:n]

    X_train = X_train_full[idx]
    y_train = y_train_full[idx]

    return X_train, y_train, X_test, y_test

def onehot(y):
    return F.one_hot(y, num_classes=10).float()