import json
import logging
import os
import time
from dataclasses import InitVar, asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import git
from simple_parsing import ArgumentParser, Serializable
from simple_parsing.helpers import dict_field

logger = logging.getLogger(__name__)


@dataclass
class CfgFileConfig:
    """Config file args"""

    config: Optional[Path] = None  # path to config file
    save_config: bool = (
        True  # set to false if you don't want to save config automatically
    )


@dataclass
class ModelParams:
    clip_checkpoint: str = "openai/clip-vit-base-patch32"
    transformer_base_model: str = "bert-base-uncased"
    hidden_size: int = 256
    num_hidden_layers: int = 3
    num_attention_heads: int = 2
    freeze_vision_base: bool = True


@dataclass
class Hparams:
    """General Hyperparameters"""

    seed: int = 13
    resume_run: Optional[bool] = None
    do_validation: bool = True

    # --------------------
    # Training parameters
    # --------------------
    batch_size: int = 16
    grad_acc_size: int = 1  # gradient accumulation batch size
    grad_clip: float = 1.0
    max_num_opt_steps: Optional[
        int
    ] = 500_000  # int(max_num_tokens/ (batch_size * max_seq_len * grad_acc_size * num_processes))

    # --------------------
    # Logging parameters
    # --------------------
    train_logging_opt_steps: int = 50
    val_logging_opt_steps: int = train_logging_opt_steps * 5
    train_saving_opt_steps: int = train_logging_opt_steps * 5
    save_dir: Optional[Path] = "save"


@dataclass
class DatasetParams:
    """Dataset Parameters"""

    train_data_path: str = None
    val_data_path: Optional[str] = None
    video_path: Optional[str] = None
    num_partitions: int = 4
    num_candidates: int = 1
    keep_aspect_ratio: bool = False
    new_height: int = 224
    new_width: int = 224
    num_threads: int = 1
    num_workers: int = 0


@dataclass
class OptimizerParams:
    """Optimization parameters"""

    # --------------------
    # optim parameters
    # --------------------
    optim: str = "AdamW"  # Optimizer default: AdamW
    optim_params: Dict[str, Any] = dict_field(
        dict(
            lr=1e-4,  # learning rate, default = 1e-4
            betas=(0.9, 0.999),  # beta1 for adam. default = (0.9, 0.999)
            weight_decay=0.1,  # TODO: when we integrate the perceiver resampler, change this so the perceiver resampler params don't get decayed
        )
    )

    lr_scheduler: str = "get_constant_schedule_with_warmup"
    lr_scheduler_params: Dict[str, Any] = dict_field(
        dict(
            num_warmup_steps=5_000, last_epoch=-1
        )  # number of warmup steps for the learning rate
    )


@dataclass
class Parameters(Serializable):
    """base options."""

    hparams: Hparams = Hparams()
    optim: OptimizerParams = OptimizerParams()
    model: ModelParams = ModelParams()
    data: DatasetParams = DatasetParams()
    should_verify: InitVar[bool] = True

    def verify(self, should_verify: bool):
        if not should_verify:
            return

        dict_rep = vars(self)
        expected = vars(self.__class__(should_verify=False))
        for key, value in dict_rep.items():
            if isinstance(value, dict):
                diff = set(value.keys()) - set(asdict(expected[key]).keys())
                raise TypeError(
                    f"{key} in {self.__class__.__name__} has extra keys: {diff}. Please fix your config if you are"
                    " using one."
                )
            if key not in expected:
                raise ValueError(
                    f"{key} is not a valid parameter for {self.__class__.__name__}"
                )

    def __post_init__(self, should_verify: bool = True):
        """Post-initialization code"""
        self.verify(should_verify=should_verify)

        # Get commit id
        self.hparams.repo_commit_id = git.Repo(
            search_parent_directories=True
        ).head.object.hexsha

        # Assign batch size to data as well for dataloaders
        self.data.batch_size = self.hparams.batch_size

    @classmethod
    def parse(cls):
        cfgfile_parser = ArgumentParser(add_help=False)
        cfgfile_parser.add_arguments(CfgFileConfig, dest="cfgfile")
        cfgfile_args, rest = cfgfile_parser.parse_known_args()

        cfgfile: CfgFileConfig = cfgfile_args.cfgfile

        file_config: Optional[Parameters] = None
        if cfgfile.config is not None:
            file_config = Parameters.load(cfgfile.config)

        parser = ArgumentParser()

        # add cfgfile args so they appear in the help message
        parser.add_arguments(CfgFileConfig, dest="cfgfile")
        parser.add_arguments(Parameters, dest="parameters", default=file_config)

        args = parser.parse_args()

        parameters: Parameters = args.parameters

        if cfgfile.save_config is not None:
            parameters.hparams.save_dir.mkdir(parents=True, exist_ok=True)
            config_file_name = "config.yaml"
            parameters.save(parameters.hparams.save_dir / config_file_name, indent=4)

        return parameters


def get_config(print_config: bool = True):
    parameters: Parameters = Parameters.parse()
    if print_config:
        print(parameters)
    return parameters


if __name__ == "__main__":
    get_config()
