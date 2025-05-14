import torch
import torch.nn as nn
from transformers import BertModel

class BertTransformerFusion(nn.Module):
    def __init__(self, num_features, transformer_dim=64, fusion_dim=128):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.text_proj = nn.Linear(self.bert.config.hidden_size, fusion_dim)

        self.feature_proj = nn.Linear(num_features, transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim + transformer_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, numeric_features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = bert_output.last_hidden_state[:, 0, :]
        text_embed = self.text_proj(cls_embedding)

        x = self.feature_proj(numeric_features)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        num_embed = x.squeeze(1)

        combined = torch.cat([text_embed, num_embed], dim=1)
        output = self.fusion(combined)
        return output.squeeze(1)
