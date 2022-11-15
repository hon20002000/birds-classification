
from flask import Flask, render_template
from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, RandomResizedCrop, ToTensor
import json
from PIL import Image


def predict(img_path):
    feature_extractor = ConvNextFeatureExtractor.from_pretrained("facebook/convnext-xlarge-224-22k")
    path = '../model/tiny_model_221110_epoch_2.pt'
    with open('../model/id2label.json', 'r') as f:
        id2label = json.load(f)
    with open('../model/label2id.json', 'r') as f:
        label2id = json.load(f)
    model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224",
                                                            num_labels=len(id2label),
                                                            id2label=id2label,
                                                            label2id=label2id,
                                                            ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(path))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    transform = Compose([RandomResizedCrop(feature_extractor.size),
                         RandomHorizontalFlip(),
                         ToTensor(),
                         normalize])
    img = Image.open(img_path)
    img = transform(img.convert("RGB"))
    outputs = model(pixel_values=img.unsqueeze(0), labels=None)
    predicted = outputs.logits.argmax(-1)
    bird_name = id2label[str(predicted.item())]
    return bird_name
