import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel


class TriModalPneumoniaNet(nn.Module):
    """
    True tri-modal network:
      - X-ray image  → DenseNet121 backbone (3-channel)
      - CT image     → ResNet18 backbone modified to 1-channel
      - Text         → BERT encoder

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
        self.xray_num_ftrs = densenet.classifier.in_features  # 1024

        # -------- CT BACKBONE (ResNet18, 1-channel) --------
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # change first conv to 1 channel (for grayscale CT)
        if resnet.conv1.in_channels != 1:
            resnet.conv1 = nn.Conv2d(
                1,
                resnet.conv1.out_channels,
                kernel_size=resnet.conv1.kernel_size,
                stride=resnet.conv1.stride,
                padding=resnet.conv1.padding,
                bias=False,
            )

        # keep everything except the final fc
        self.ct_backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> (B, 512, 1, 1)
        self.ct_num_ftrs = resnet.fc.in_features  # 512

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
        xray_img:   (B, 1, 224, 224)  -> we convert to 3-channel
        ct_img:     (B, 1, 224, 224)  -> stays 1-channel for modified ResNet
        input_ids:  (B, L)
        attention_mask: (B, L)
        """

        # ----- PREPROCESS CHANNELS -----

        # X-RAY: DenseNet expects 3 channels → repeat grayscale to (B, 3, H, W)
        if xray_img.shape[1] == 1:
            xray_img = xray_img.repeat(1, 3, 1, 1)

        # CT: ResNet conv1 was changed to accept 1 channel, so we DO NOT repeat
        # ct_img stays (B, 1, H, W)

        # ----- X-RAY BRANCH -----
        x = self.xray_features(xray_img)                 # (B, C, H, W)
        x = F.relu(x, inplace=True)
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

        if hasattr(text_outputs, "pooler_output") and text_outputs.pooler_output is not None:
            text_feat = text_outputs.pooler_output       # (B, 768)
        else:
            text_feat = text_outputs.last_hidden_state[:, 0, :]  # (B, 768)

        # ----- FUSION -----
        fused = torch.cat([xray_feat, ct_feat, text_feat], dim=1)
        fused = self.fusion_dropout(F.relu(self.fusion_fc1(fused)))
        logits = self.fusion_fc_out(fused)

        return logits
