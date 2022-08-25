from ctypes import sizeof
from dataclasses import asdict

import transformers
import pytorch_lightning as pl
import torchmetrics
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
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.train_acc = torchmetrics.Accuracy(num_classes=101)
        self.valid_acc = torchmetrics.Accuracy(num_classes=101, ignore_index=-1)


    def forward(self, x, class_tensors):
        return self.atp(x, class_tensors)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x, self.class_tensors)
        loss = self.loss(y_hat, y)
        return {"loss": loss, 'preds': y_hat, 'target': y}

    def training_step_end(self, outputs):
        self.train_acc(torch.argmax(outputs['preds'], 1), outputs['target'])
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        self.log("train_loss", outputs['loss']) 
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x, self.class_tensors)
        loss = self.loss(y_hat, y)
        return {"loss": loss, 'preds': y_hat, 'target': y}

    def validation_step_end(self, outputs):
        self.valid_acc(torch.argmax(outputs['preds'], 1), outputs['target'])
        self.log('valid_acc', self.valid_acc, on_step=True, on_epoch=True)
        self.log("valid_loss", outputs['loss'])

    def configure_optimizers(self):
        optim_class = getattr(torch.optim, self.optim_config.optim)
        optimizer = optim_class(self.parameters(), **self.optim_config.optim_params)
        lr_class = getattr(transformers, self.optim_config.lr_scheduler)
        lr_scheduler = lr_class(optimizer, **self.optim_config.lr_scheduler_params)
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=500, num_training_steps=10000)
        return [optimizer], [lr_scheduler]
