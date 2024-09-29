"""
module for gradient-based test-time methods, e.g., finetune, eTT, TSA, URL, Cosine Classifier
"""
from architectures import get_backbone, get_classifier
import torch.nn as nn
import  torch
import torch.nn.functional as F
from utils import accuracy
from utils import accuracy1
from architectures.classifier.proto import  jsproto
import numpy as np

def compute_distance(sample1,sample2):
    return torch.sqrt(torch.sum((sample1-sample2)**2))

class FinetuneModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.backbone = get_backbone(config.MODEL.BACKBONE, *config.MODEL.BACKBONE_HYPERPARAMETERS)

        # The last hyperparameter is the head mode
        self.mode = config.MODEL.CLASSIFIER_PARAMETERS[-1]

        
        if not self.mode == "NCC":
            classifier_hyperparameters = [self.backbone]+config.MODEL.CLASSIFIER_PARAMETERS
            self.classifier = get_classifier(config.MODEL.CLASSIFIER, *classifier_hyperparameters)
    
    def append_adapter(self):
        # append adapter to the backbone
        self.backbone = get_backbone("resnet_tsa",backbone=self.backbone)
        classifier_hyperparameters = [self.backbone]+self.config.MODEL.CLASSIFIER_PARAMETERS
        self.classifier = get_classifier(self.config.MODEL.CLASSIFIER, *classifier_hyperparameters)

    def test_forward(self, img_tasks,label_tasks, *args, **kwargs):
        batch_size = len(img_tasks)
        loss = 0.
        acc = []
        score=[]
        for i, img_task in enumerate(img_tasks):
            bscore = self.classifier(img_task["query"].squeeze_().cuda(), img_task["support"].squeeze_().cuda(),
                                     label_tasks[i]["support"].squeeze_().cuda())
            score.append(bscore)
        return score

def get_model(config):
    return FinetuneModule(config)