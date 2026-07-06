# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import AvgMeter
from utils import get_lr
from torch.utils.data import DataLoader
import argparse

from get_HE_model import get_HE_encoder

from config import CFG

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout,
        eps=1e-06  #
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim, eps=eps)  #
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

#model256 = get_HE_encoder()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class HIPT_CLIP_Model(nn.Module):
    def __init__(self, temperature=1, image_embedding=CFG.image_embedding, spot_embedding=CFG.spot_embedding, projection_dim=384):
        super().__init__()
        # Use VisionTransformer as the new image_encoder
        self.image_encoder = get_HE_encoder()
        #self.image_projection = ProjectionHead(embedding_dim=image_embedding)  #
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding, projection_dim=projection_dim)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / temperature))

    def forward(self, batch):
        # Getting Image and spot Features
        #print(batch["image"].shape)
        image_features = self.image_encoder(batch["image"])
        #print(image_features.shape)
        spot_features = batch["reduced_expression"]

        # Getting Image and Spot Embeddings (with same dimension) 
        #image_embeddings = self.image_projection(image_features)
        image_embeddings = image_features
        spot_embeddings = self.spot_projection(spot_features)

        #L2 normalization
        norm_image = image_embeddings.norm(p=2, dim=1, keepdim=True) 
        image_embeddings = image_embeddings/norm_image

        norm_spot = spot_embeddings.norm(p=2, dim=1, keepdim=True) 

        spot_embeddings = spot_embeddings/norm_spot

        
        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature.exp()

        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T

        batch_size = image_embeddings.size(0)
        identity_mask = torch.eye(batch_size, device=image_embeddings.device)

        # Reduce similarity only for non-paired samples
        adjusted_images_similarity = images_similarity * identity_mask + images_similarity * (1 - identity_mask) / 1.1
        adjusted_spots_similarity = spots_similarity * identity_mask + spots_similarity * (1 - identity_mask) / 1.1

        # Combined similarity
        combined_similarity = (adjusted_images_similarity + adjusted_spots_similarity) / 2
        
        #combined_similarity = 1 * ((images_similarity + spots_similarity) / 2)

        targets = F.softmax(combined_similarity / self.temperature.exp(), dim=-1)
        #print(logits)
        #print(targets)
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

    def encode_image(self, image):
        return self.image_encoder(image)

    def encode_spot(self, spot):
        return self.spot_projection(spot)