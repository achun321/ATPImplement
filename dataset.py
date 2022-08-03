import os
import random
import re
from math import ceil
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from config import DatasetParams


class VideoDataModule(pl.LightningDataModule):
    def __init__(self, config: DatasetParams):
        super().__init__()
        self.train_data_path = config.train_data_path
        self.val_data_path = (
            config.val_data_path
            if config.val_data_path is not None
            else config.train_data_path
        )
        self.video_path = config.video_path
        self.num_partitions = config.num_partitions
        self.num_candidates = config.num_candidates
        self.keep_aspect_ratio = config.keep_aspect_ratio
        self.new_height = config.new_height
        self.new_width = config.new_width
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = VideoDataset(
            data_path=self.train_data_path,
            video_path=self.video_path,
            split="train",
            num_partitions=self.num_partitions,
            num_candidates=self.num_candidates,
            keep_aspect_ratio=self.keep_aspect_ratio,
            new_height=self.new_height,
            new_width=self.new_width,
        )
        self.val_dataset = VideoDataset(
            data_path=self.val_data_path,
            video_path=self.video_path,
            split="test",
            num_partitions=self.num_partitions,
            num_candidates=self.num_candidates,
            keep_aspect_ratio=self.keep_aspect_ratio,
            new_height=self.new_height,
            new_width=self.new_width,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    @property
    def classes(self):
        return self.train_dataset.classes


def _ucf101_parser(path, video_path=None, split="train"):
    if video_path is None:
        video_path = os.path.join(path, "UCF-101")

    classes = []
    class_path = os.path.join(path, "classInd.txt")
    with open(class_path, "r") as f:
        for line in f:
            classname = line.split(" ")[1].strip()
            # Split upper case to words so that clip can process them
            classname = re.sub("([A-Z])", " \g<0>", classname).lower().strip()
            classes.append(classname)

    samples = []
    for item in os.listdir(path):
        if split in item:
            with open(os.path.join(path, item)) as f:
                for line in f:
                    line = line.strip().split(" ")
                    if len(line) == 2:
                        samples.append(
                            (os.path.join(video_path, line[0]), int(line[1]) - 1)
                        )
                    else:
                        samples.append((os.path.join(video_path, line[0]),))
    return samples, classes


class VideoDataset(Dataset):
    """
    Loads in video dataset as a csv of data paths to desired video dataset.
    """

    def __init__(
        self,
        data_path,
        video_path=None,
        split="train",
        num_partitions=4,
        num_candidates=1,
        keep_aspect_ratio=True,
        new_height=256,
        new_width=320,
        num_threads=1,
    ):
        self.data_path = data_path
        self.num_partitions = num_partitions
        self.num_candidates = num_candidates
        self.keep_aspect_ratio = keep_aspect_ratio
        self.new_height = new_height
        self.new_width = new_width
        self.num_threads = num_threads

        # Possibly can be extended to other parsers
        self.data_samples, self.classes = _ucf101_parser(
            self.data_path, video_path, split
        )
        self.data_samples = self._filter_existing(self.data_samples)

    def _filter_existing(self, data_samples):
        filtered = []
        for i in data_samples:
            if os.path.exists(i[0]):
                filtered.append(i)
        return filtered

    def __getitem__(self, index):
        """
        Given index of video from dataset, returns set of random shuffled
        frames from each partition of video.
        """
        sample = self.data_samples[index]
        if len(sample) == 2:
            video_path, label = sample
        else:
            video_path = sample[0]
            label = -1
        video_tensor = self.load_video(video_path)
        return video_tensor, torch.tensor(label)

    def __len__(self):
        return len(self.data_samples)

    def load_video(self, sample):
        """
        Takes video as input and reads with decord. Takes frames read from
        decord and divides them evenly into batches of frames which is then returned.
        """
        filename = sample

        if not (os.path.exists(filename)):
            print(f"Video from {filename} does not exist.")
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(filename, num_threads=self.num_threads, ctx=cpu(0))
            else:
                vr = VideoReader(
                    filename,
                    num_threads=self.num_threads,
                    ctx=cpu(0),
                    width=self.new_width,
                    height=self.new_height,
                )
        except:
            print(f"Video {filename} cannot be loaded by decord.")
            return []

        frames = []
        seg_len = ceil(len(vr) / self.num_partitions)
        curr_frame = 0

        while curr_frame < len(vr):
            # for each partition, create a batch of frames and append to set to be returned
            for _ in range(self.num_candidates):
                frame_idx = curr_frame + random.randint(
                    0, min(len(vr) - curr_frame, seg_len) - 1
                )
                frames.append(vr[frame_idx].asnumpy())
            curr_frame += seg_len

        assert len(frames) == self.num_partitions * self.num_candidates
        # Shuffle so that there is no order
        random.shuffle(frames)
        frames = np.array(frames)
        frames = torch.from_numpy(frames.transpose(0, 3, 1, 2).astype(np.float32))
        return frames
