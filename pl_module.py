from dataclasses import asdict

import pytorch_lightning as pl
import torch
from torch import nn
from transformers import CLIPTokenizer

from config import ModelParams, OptimizerParams
from model import ATP


class ATPLightningModule(pl.LightningModule):
    def __init__(
        self, model_config: ModelParams, optim_config: OptimizerParams, classes
    ):
        super().__init__()
        self.atp = ATP(**asdict(model_config))
        self.optim_config = optim_config
        with torch.no_grad():
            tokenizer = CLIPTokenizer.from_pretrained(
                model_config.clip_checkpoint, use_fast=True
            )
            class_tensors = tokenizer(classes, padding=True, return_tensors="pt")
            self.atp.clip.eval()
            self.class_tensors = self.atp.clip.get_text_features(**class_tensors)
            self.atp.clip.train()
            self.class_tensors.requires_grad = False
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, class_tensors):
        return self.atp(x, class_tensors)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x, self.class_tensors)
        loss = self.loss(y_hat, y)
        return {"loss": loss}

    def configure_optimizers(self):
        optim_class = getattr(torch.optim, self.optim_config.optim)
        optimizer = optim_class(self.parameters(), **self.optim_config.optim_params)
        # TODO(Adam): Add scheduler
        return optimizer
