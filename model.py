import torch
from torch import nn
from transformers import BertConfig, CLIPModel
from transformers.models.bert import BertEncoder


class ATP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(config.clip_checkpoint)
        bert_config = BertConfig.from_pretrained(
            config.transformer_base_model,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
        )
        self.initializer_range = bert_config.initializer_range

        self.projection = nn.Linear(
            self.clip.config.hidden_size, bert_config.hidden_size
        )
        self.atp_selector = BertEncoder(bert_config)
        self.classifier = nn.Sequential(
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
            nn.ReLU(),
            nn.Linear(bert_config.hidden_size, config.num_partitions),
        )
        self.init_weights()

        if config.freeze_vision_base:
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
            x = self.clip.get_image_features(x)
        x = self.projection(x)
        x = self.atp_selector(x)
        logits = self.classifier(x)
        probs = nn.functional.softmax(logits, dim=-1)
        x = (probs * x).sum(dim=-1)
        class_logits = x @ class_tensors.t()
        return class_logits
