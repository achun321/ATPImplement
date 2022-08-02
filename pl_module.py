import pytorch_lightning as pl
import torch
from torch import nn

from model import ATP


class ATPLightningModule(pl.LightningModule):
    def __init__(self, classes):
        super().__init__()
        self.atp = ATP()
        with torch.no_grad():
            self.class_tensors = self.atp.clip.get_text_features(classes)
            self.class_tensors.requires_grad = False
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.atp(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x, self.class_tensors)
        loss = self.loss(y_hat, y)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
