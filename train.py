import pytorch_lightning as pl
import os

from config import get_config
from dataset import VideoDataModule
from pl_module import ATPLightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

if __name__ == "__main__":
    config = get_config()
    datamodule = VideoDataModule(config.data)
    datamodule.setup()
    datamodule.prepare_data()
    checkpoint_callback = ModelCheckpoint(dirpath="lightning_logs/save_top_10", save_top_k=10, monitor="val_loss")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    module = ATPLightningModule(config.model, config.optim, datamodule.classes)
    logger = TensorBoardLogger(save_dir="./tb_logs")
    
    trainer = pl.Trainer(callbacks=[checkpoint_callback, lr_monitor], logger=logger)
    trainer.fit(model=module, datamodule=datamodule)
