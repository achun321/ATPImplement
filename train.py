import pytorch_lightning as pl

from dataset import VideoDataModule
from pl_module import ATPLightningModule
from config import get_config

if __name__ == "__main__":
    config = get_config()
    module = ATPLightningModule(config.model)
    datamodule = VideoDataModule(
        train_data_path="data/train.txt", val_data_path="data/val.txt"
    )
    trainer = pl.Trainer()
    trainer.fit(model=module, datamodule=datamodule)
