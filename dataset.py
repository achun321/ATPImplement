import os
import random
import re
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset


class VideoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_path,
        val_data_path,
        batch_size=32,
        clip_len=32,
        partitions=4,
        candidates=4,
        keep_aspect_ratio=True,
        new_height=256,
        new_width=320,
    ):
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.clip_len = clip_len
        self.partitions = partitions
        self.candidates = candidates
        self.keep_aspect_ratio = keep_aspect_ratio
        self.new_height = new_height
        self.new_width = new_width
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = VideoDataset(
            self.train_data_path,
            self.partitions,
            self.candidates,
            self.keep_aspect_ratio,
            self.new_height,
            self.new_width,
        )
        self.val_dataset = VideoDataset(
            self.val_data_path,
            self.partitions,
            self.candidates,
            self.keep_aspect_ratio,
            self.new_height,
            self.new_width,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )

def _ucf101_parser(path, video_path=None, split="train"):
    if video_path is None:
        video_path = os.path.join(path, "UCF-101")

    classes = []
    class_path = os.path.join(path, "classInd.txt")
    with open(class_path, "r") as f:
        for line in f:
            classname = line.split(" ")[1].strip()
            # Split upper case to words so that clip can process them
            classname = re.sub("([A-Z])"," \g<0>", classname).lower().strip()
            classes.append(classname)
    
    samples = []
    for item in os.listdir(path):
        if split in item:
            with open(os.path.join(path, item)) as f:
                for line in f:
                    line = line.strip().split(" ")
                    if len(line) == 2:
                        samples.append((os.path.join(video_path, line[0]), int(line[1]) - 1))
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
        clip_len=32,
        partitions=4,
        candidates=4,
        keep_aspect_ratio=True,
        new_height=256,
        new_width=320,
        num_threads=1,
    ):
        self.data_path = data_path
        self.clip_len = clip_len
        self.partitions = partitions
        self.candidates = candidates
        self.keep_aspect_ratio = keep_aspect_ratio
        self.new_height = new_height
        self.new_width = new_width
        self.num_threads = num_threads

        # Possibly can be extended to other parsers
        self.data_samples, self.classes = _ucf101_parser(self.data_path, video_path, split)

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
        batches = self.load_video(video_path)
        return batches, label

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

        batches = []
        seg_len = len(vr) // self.partitions
        curr_frame = 0

        while curr_frame < len(vr):
            # for each partition, create a batch of frames and append to set to be returned
            frame_idx = curr_frame + random.randint(0, seg_len)
            batches.append(vr[frame_idx])
            curr_frame += seg_len

        assert len(batches) == self.partitions
        # Shuffle so that there is no order
        random.shuffle(batches)
        return batches