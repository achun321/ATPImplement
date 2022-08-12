import pytorch_lightning as pl

from config import get_config
from dataset import VideoDataModule
from pl_module import ATPLightningModule

if __name__ == "__main__":
    config = get_config()
    datamodule = VideoDataModule(config.data)
    datamodule.setup()
    datamodule.prepare_data()
    module = ATPLightningModule(config.model, config.optim, datamodule.classes)
    trainer = pl.Trainer(accelerator="gpu", devices=4, num_nodes=2, strategy="ddp", default_root_dir="./checkpoints")
    trainer.fit(model=module, datamodule=datamodule) 
