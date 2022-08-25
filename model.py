import torch
from torch import nn
from transformers import BertConfig, CLIPModel
from transformers.models.bert.modeling_bert import BertEncoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ATP(nn.Module):
    def __init__(
        self,
        clip_checkpoint: str = "openai/clip-vit-base-patch32",
        transformer_base_model: str = "bert-base-uncased",
        hidden_size: int = 256,
        num_hidden_layers: int = 3,
        num_attention_heads: int = 2,
        freeze_vision_base: bool = True,
        **kwargs
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_checkpoint)
        bert_config = BertConfig.from_pretrained(
            transformer_base_model,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
        )
        self.initializer_range = bert_config.initializer_range
        self.projection = nn.Linear(self.clip.config.projection_dim, hidden_size)
        self.atp_selector = BertEncoder(bert_config)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.init_weights()

        if freeze_vision_base:
            for p in self.clip.parameters():
                p.requires_grad = False

    def init_weights(self):
        for n, p in self.named_parameters():
            if "clip" not in n:
                self._init_weights(p)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, class_tensors):
        with torch.no_grad():
            bs, num_partitions, c, h, w = x.size()
            x = x.view(bs * num_partitions, c, h, w)
            x = self.clip.get_image_features(x)
            x = x.view(bs, num_partitions, -1)
        projection = self.projection(x) 
        selection = self.atp_selector(projection).last_hidden_state
        logits = self.classifier(selection).squeeze()
        probs = nn.functional.softmax(logits, dim=-1)
        x = (probs.unsqueeze(-1) * x).sum(dim=1)
        x = x.to(device)
        class_tensors = class_tensors.to(device)
        class_logits = x @ class_tensors.t()
        return class_logits
