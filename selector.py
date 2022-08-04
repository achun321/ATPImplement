"""
Takes in batches of frames for each partition from video --> encodes each image
using frozen ViT-B32 image encoder --> takes these encodings and inputs them into
frozen selector encoder (ATP).
"""
import clip
import torch

from atp import VideoTransformer
from dataset import VideoDataset

UCF_datapath = (
    "/Users/achun/Desktop/HFInternship/VideoMAE-UCF101-Strided/UCF101/ucfTrainTestlist"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ATP paper reports primary analysis uses ViT-B32 image encoder
model, preprocess = clip.load("ViT-B/32", device=device)

vr = VideoDataset(UCF_datapath, clip_len=32, partitions=4, candidates=4)

# freeze encoder to reinforce atemporality
with torch.no_grad():
    for batch in vr:
        for image in batch:
            model.encode_image(image)

# input these encoded images into a frozen selector encoder and use Gumbel-Softmax estimator on logits to select best frame
