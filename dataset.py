import os
import numpy as np
import torch
import decord
from decord import VideoReader, cpu
from torch.utils.data import Dataset
import pandas as pd
import itertools
import random

class VideoDataset(Dataset):
    """
    Loads in video dataset as a csv of data paths to desired video dataset. 
    """
    def __init__(self, data_path, clip_len=32, partitions=4, candidates=4, keep_aspect_ratio=True,
                new_height=256, new_width=320):
        self.data_path = data_path
        self.clip_len = clip_len
        self.partitions = partitions
        self.candidates = candidates
        self.keep_aspect_ratio = keep_aspect_ratio
        self.new_height = new_height
        self.new_width = new_width

        # data would be csv with single col of file paths to videos
        cleaned_data = pd.read_csv(self.data_path, header=None, delimiter=' ')
        self.data_samples = list(cleaned_data.values[:, 0])

    def __getitem__(self, index):
        """
        Given index of video from dataset, returns set of random shuffled
        frames from each partition of video. 
        """
        sample = self.data_samples[index]
        batches = self.load_video(sample)
        # for every batch in batches, shuffle video frames and randomly choose self.candidates number of frames
        for batch in batches:
            batch = self.random_select_candidates(batch)
        return batches

    def load_video(self, sample):
        """
        Takes video as input and reads with decord. Takes frames read from
        decord and divides them evenly into batches of frames which is then returned.  
        """
        filename = sample

        if not(os.path.exists(filename)):
            print(f"Video from {filename} does not exist.")
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(filename, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(filename, num_threads=1, ctx=cpu(0), width=self.new_width, height=self.new_height)
        except:
            print(f"Video {filename} cannot be loaded by decord.")
            return []
        
        batches = []
        seg_len = len(vr) // self.partitions
        curr_frame = 0

        while curr_frame < len(vr):
            # for each partition, create a batch of frames and append to set to be returned
            batches.append(vr.get_batch(range(curr_frame, curr_frame + seg_len - 1)))
            curr_frame += seg_len
        
        return batches
        
    def random_select_candidates(self, batch: list):
        # takes in one batch of frames as input and randomly selects candidates / shuffles them
        while len(batch) > self.candidates:
            batch.pop(random.randrange(len(batch)))
        random.shuffle(batch)
        return batch



