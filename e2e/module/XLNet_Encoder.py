from module.Encoder import Encoder
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class Xlnet_Encoder(nn.Module):
    def __init__(self, device, model_path, encoder_layers=12, drop_prob=0, class_num=2):
        super(Xlnet_Encoder, self).__init__()
        self.embed_dim = 768
        self.hidden_size = 768
        self.device = device
        self.xlnet = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.encode = Encoder(device=self.device, n_layers=encoder_layers, d__model=self.embed_dim, d_k=self.embed_dim,
                              d_v=self.embed_dim, h=12)
        self.dropout = nn.Dropout(p=drop_prob)

        self.classifier = nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.embed_dim, class_num)
        )

    def forward(self, sentence, sep_index):
        embedding = self.xlnet(sentence)[0]
        sen_vec_list = []
        for index in sep_index:
            sen_vec_list.append(embedding[:, index, :])
        sen_vec = torch.stack(sen_vec_list).squeeze(0)  # [batch,sen_num,embed_dim]
        encoder_input = self.dropout(sen_vec)
        out = self.encode(encoder_input)
        class_out = self.classifier(out).squeeze(0)
        return class_out
