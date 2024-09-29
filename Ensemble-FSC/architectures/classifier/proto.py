"""
The metric-based protypical classifier (Nearest-Centroid Classifier) from ``Prototypical Networks for Few-shot Learning''.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .utils import compute_prototypes


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits

def jsproto( query_images: Tensor, support_images: Tensor, support_labels):
    """Take one task of few-shot support examples and query examples as input,
                output the logits of each query examples.

            Args:
                query_images: query examples. size: [num_query, c, h, w]
                support_images: support examples. size: [num_support, c, h, w]
                support_labels: labels of support examples. size: [num_support, way]
            Output:
                classification_scores: The calculated logits of query examples.
                                       size: [num_query, way]
            """
    if query_images.dim() == 4:
        support_images = F.adaptive_avg_pool2d(support_images, 1).squeeze_(-1).squeeze_(-1)
        query_images = F.adaptive_avg_pool2d(query_images, 1).squeeze_(-1).squeeze_(-1)

    assert support_images.dim() == query_images.dim() == 2

    support_images = F.normalize(support_images, p=2, dim=1, eps=1e-12)
    query_images = F.normalize(query_images, p=2, dim=1, eps=1e-12)

    one_hot_label = F.one_hot(support_labels, num_classes=torch.max(support_labels).item() + 1).float()

    # prototypes: [way, c]
    prototypes = compute_prototypes(support_images, one_hot_label)

    prototypes = F.normalize(prototypes, p=2, dim=1, eps=1e-12)
    return prototypes

