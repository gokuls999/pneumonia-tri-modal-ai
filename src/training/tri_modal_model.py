# src/training/tri_modal_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel


class TriModalPneumoniaNet(nn.Module):
    """
    Tri-modal network:
      - X-ray image  → DenseNet121 backbone (expects 3-channel input)
      - CT image     → ResNet18 backbone modified to accept 1-channel
      - Text         → BERT encoder (AutoModel)

    Forward:
      logits = model(xray_img, ct_img, input_ids, attention_mask)
    """

    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        fused_hidden_size: int = 512,
    ):
        super().__init__()

        # -------- X-RAY BACKBONE (DenseNet121) --------
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.xray_features = densenet.features
        self.xray_num_ftrs = densenet.classifier.in_features  # typically 1024

        # keep a handle to the features module for Grad-CAM
        self.xray_backbone_for_cam = self.xray_features

        # -------- CT BACKBONE (ResNet18, 1-channel) --------
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Replace first conv to accept 1 channel but keep other conv1 properties
        orig_conv = resnet.conv1
        if orig_conv.in_channels != 1:
            resnet.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=orig_conv.out_channels,   # keep original out_channels (usually 64)
                kernel_size=orig_conv.kernel_size,     # keep original kernel (usually (7,7))
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False,
            )

        # Use the ResNet layers up to avgpool as ct backbone
        # `list(resnet.children())[:-1]` gives everything except final fc
        self.ct_backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> (B, 512, 1, 1)
        self.ct_num_ftrs = resnet.fc.in_features  # typically 512

        # For Grad-CAM we want the conv feature map BEFORE avgpool → layer4
        self.ct_backbone_for_cam = resnet.layer4

        # -------- TEXT BACKBONE (BERT) --------
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.text_hidden_size = self.text_model.config.hidden_size  # 768 for bert-base-uncased

        # -------- FUSION LAYERS --------
        fusion_input_dim = self.xray_num_ftrs + self.ct_num_ftrs + self.text_hidden_size
        self.fusion_fc1 = nn.Linear(fusion_input_dim, fused_hidden_size)
        self.fusion_dropout = nn.Dropout(p=0.3)
        self.fusion_fc_out = nn.Linear(fused_hidden_size, num_classes)

    def forward(self, xray_img, ct_img, input_ids, attention_mask):
        """
        xray_img:   (B, 1 or 3, H, W)  -> will convert to 3-channel if needed
        ct_img:     (B, 1, H, W)        -> 1-channel for modified ResNet
        input_ids:  (B, L)
        attention_mask: (B, L)
        """

        # ----- PREPROCESS CHANNELS -----
        # If xray is single-channel, replicate to 3 channels (DenseNet expects 3)
        if xray_img.dim() == 3:
            # (C,H,W) → (1,C,H,W) handled outside usually; still safe-check
            xray_img = xray_img.unsqueeze(0)
        if xray_img.shape[1] == 1:
            xray_img = xray_img.repeat(1, 3, 1, 1)

        # ct_img expected as (B,1,H,W) — keep as-is

        # ----- X-RAY BRANCH -----
        x = self.xray_features(xray_img)                 # (B, C, H, W)
        x = F.relu(x, inplace=False)
        x = F.adaptive_avg_pool2d(x, (1, 1))             # (B, C, 1, 1)
        xray_feat = torch.flatten(x, 1)                  # (B, 1024)

        # ----- CT BRANCH -----
        ct = self.ct_backbone(ct_img)                    # (B, 512, 1, 1)
        ct_feat = torch.flatten(ct, 1)                   # (B, 512)

        # ----- TEXT BRANCH -----
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # AutoModel returns either pooler_output or last_hidden_state
        if hasattr(text_outputs, "pooler_output") and text_outputs.pooler_output is not None:
            text_feat = text_outputs.pooler_output       # (B, hidden)
        else:
            text_feat = text_outputs.last_hidden_state[:, 0, :]  # CLS token

        # ----- FUSION -----
        fused = torch.cat([xray_feat, ct_feat, text_feat], dim=1)
        fused = self.fusion_dropout(F.relu(self.fusion_fc1(fused)))
        logits = self.fusion_fc_out(fused)

        return logits
