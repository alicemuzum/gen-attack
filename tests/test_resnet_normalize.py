import torch

from gen_attack.models.resnet import IMAGENET_MEAN, IMAGENET_STD, Normalize


def test_normalize_matches_manual_computation():
    images = torch.rand(2, 3, 4, 4)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    assert torch.allclose(Normalize(IMAGENET_MEAN, IMAGENET_STD)(images), (images - mean) / std)


def test_normalize_is_differentiable():
    images = torch.rand(1, 3, 4, 4, requires_grad=True)
    Normalize(IMAGENET_MEAN, IMAGENET_STD)(images).sum().backward()
    assert images.grad is not None
