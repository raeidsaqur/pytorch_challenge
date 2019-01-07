#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os, sys
import numpy as np
np.random.seed(42)

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
DATAROOT = "flower_data/"
CHECKPOINTS_ROOT_DIR = "checkpoints/"
BATCH_SIZE = 32
SHUFFLE = True
NUM_WORKERS = 4 * 1 if USE_CUDA else 2

isDebug = True
device = torch.device("cuda" if USE_CUDA else "cpu")

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
classes = list(set(cat_to_name.values()))  # 102 flower classes
name_to_cat = {v:k for k,v in cat_to_name.items()}


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    classes = model.classes
    probs = []
    return probs, classes

def train_model(opt):
    # LOAD DATA #
    data_dir = 'flower_data'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, ratio=(1, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    # Note: No scaling or rotation, only resize for validation set.
    val_data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = datasets.ImageFolder(train_dir, train_data_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, val_data_transforms)

    num_train = len(train_dataset)
    num_valid = len(valid_dataset)
    print(f"Number of training samples = {num_train}\nNumber of validation_samples = {num_valid}")
    # image_datasets = datasets.ImageFolder('flower_data', transform=data_transforms)
    image_datasets = [train_dataset, valid_dataset]

    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=SHUFFLE,
                              num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=SHUFFLE,
                              num_workers=NUM_WORKERS)

    dataloaders = [train_loader, valid_loader]

    # LOAD MODEL: VGG16 #
    print(f"Loading VGG16 for feature extraction")
    vgg16 = models.vgg16(pretrained=True)
    # Freeze training for all "features" layers
    for param in vgg16.features.parameters():
        param.requires_grad = False

    vgg16.to(device)
    print(f"vgg16 = {vgg16}")
    print(vgg16.classifier[6].in_features)
    print(vgg16.classifier[6].out_features)

    # Classifier
    n_inputs = vgg16.classifier[6].in_features
    # add last linear layer (n_inputs -> 5 flower classes)
    # new layers automatically have requires_grad = True
    last_layer = nn.Linear(n_inputs, len(classes))
    vgg16.classifier[6] = last_layer

    # check to see that your last layer produces the expected number of outputs
    print(vgg16.classifier[6].out_features)

    # Adding categories to index mapping to the model
    vgg16.class_to_idx = name_to_cat
    vgg16.idx_to_class = cat_to_name
    vgg16.classes = classes

    # Loss Function and Optimizer #
    criterion = nn.CrossEntropyLoss()  # (categorical cross-entropy)
    # SGD and learning rate = 0.001
    optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.001)

    ## TRAINING ##
    print(f"######### STARTING TRAINING #############")
    n_epochs = 4  # number of epochs to train the model
    train_on_gpu = torch.cuda.is_available()

    counter = 0
    print_every = 100
    clip = 5  # gradient clipping
    vgg16.train()

    for epoch in range(1, n_epochs + 1):

        for batch_i, (data, target) in enumerate(train_loader):
            print(f"|_Processing Batch {batch_i} ... ")
            counter += 1
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = vgg16(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Loss Stats, compare with validation set #
            if counter % print_every == 0:
                # Calculate validation loss after 100 batches
                print(f"\t|__Calculating Validation Loss__|")
                val_losses = []
                vgg16.eval()  # Set to eval mode
                for inputs, labels in valid_loader:
                    if (train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()
                    val_output = vgg16(inputs)
                    val_loss = criterion(val_output, labels)
                    val_losses.append(val_loss.item())

                mean_val_loss = np.mean(val_losses)
                vgg16.train()
                print("="*15)
                print("Epoch: {}/{}...".format(epoch, n_epochs + 1),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(mean_val_loss))
                print("=" * 15)
                # Save Checkpoint #
                print(f"\t####### SAVING CHECKING POINT ######## ")
                model_name = f"VGG16_epoch{epoch}_step{counter}"
                path = CHECKPOINTS_ROOT_DIR + model_name + ".tar"
                print(f"\t\tSaving to path: {path}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': vgg16.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'val_loss': mean_val_loss
                }, path)

                # Save Entire model #
                path = CHECKPOINTS_ROOT_DIR + model_name + ".pth"
                print(f"\t\tSaving entire model to path: {path}")
                torch.save(vgg16, path)

def main(opt):
    print("Image_Classifier_Project.main(opt)")
    train_model(opt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass configurations here")
    parser.add_argument('--dataroot', required=False, default=DATAROOT,  help='path to dataset')
    opt = parser.parse_args()

    main(opt)