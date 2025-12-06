import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel


class MultimodalPneumoniaNet(nn.Module):
    def __init__(
        self,
        image_backbone_name: str = "densenet121",
        text_model_name: str = "bert-base-uncased",
        text_hidden_size: int = 768,
        fused_hidden_size: int = 512,
        num_classes: int = 2,
    ):
        super().__init__()

        # IMAGE BACKBONE
        if image_backbone_name == "densenet121":
            backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            in_features = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
            self.image_backbone = backbone
            image_feat_size = in_features
        else:
            raise ValueError("Unsupported image backbone")

        # TEXT BACKBONE
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.text_hidden_size = text_hidden_size

        # FUSION + CLASSIFIER
        self.fusion = nn.Sequential(
            nn.Linear(image_feat_size + text_hidden_size, fused_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fused_hidden_size, num_classes),
        )

    def forward(self, image, input_ids, attention_mask):
        # image: (B,1,H,W) -> replicate channel to 3 (for densenet)
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        img_feats = self.image_backbone(image)  # (B, img_feat)

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # use [CLS] token embedding
        cls_emb = text_outputs.last_hidden_state[:, 0, :]  # (B, text_hidden_size)

        fused = torch.cat([img_feats, cls_emb], dim=1)  # (B, img+txt)
        logits = self.fusion(fused)
        return logits
