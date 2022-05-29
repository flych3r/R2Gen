import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModelForImageClassification


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats


class VitVisualExtractor(nn.Module):
    def __init__(self, args):
        super(VitVisualExtractor, self).__init__()
        model = AutoModelForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)

    def forward(self, images):
        patch_feats = self.model(images).last_hidden_state
        avg_feats = patch_feats.mean(dim=1)
        return patch_feats, avg_feats
