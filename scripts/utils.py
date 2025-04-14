import numpy as np
import torch
import torch.nn.functional as F

def cosine_similarity(feat1, feat2):
    """
    Compute the cosine similarity between two feature vectors.
    """
    return F.cosine_similarity(torch.tensor(feat1), torch.tensor(feat2), dim=0).item()

def load_model(model_name='osnet_x1_0'):
    """
    Load a pre-trained model for feature extraction.
    """
    from torchreid import models
    model = models.build_model(
        name=model_name, 
        num_classes=1000, 
        loss='softmax', 
        pretrained=True
    )
    model.eval()
    model.cuda()  # Use GPU if available
    return model
