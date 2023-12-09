import numpy as np
import torch
from torch import nn
from torch.nn import init
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attentions
    '''

    def __init__(self, device,d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.device=device
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attentions values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attentions values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        # return out
        return nn.LayerNorm(self.d_model).to(self.device)(out + queries)


# d_ff=1024 #中间变量hidden
# d_model=768 #词嵌入维度
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,device,d__model,d_ff=1024):
        super(PoswiseFeedForwardNet, self).__init__()
        self.device=device
        self.d__model=d__model
        self.d_ff=d_ff
        self.fc = nn.Sequential(
            nn.Linear(self.d__model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d__model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d__model).to(self.device)(output + residual)  # [batch_size, seq_len, d_model]

class EncoderLayer(nn.Module):
    def __init__(self,device,d__model, d_k, d_v, h,dropout=0.1):
        super(EncoderLayer, self,).__init__()
        self.enc_self_attn = ScaledDotProductAttention(device,d__model, d_k, d_v, h,dropout=0.1)
        self.pos_ffn = PoswiseFeedForwardNet(device,d__model)

    def forward(self,input):
        enc_outputs=self.enc_self_attn(input,input,input)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs

class Encoder(nn.Module):
    def __init__(self,device,n_layers,d__model, d_k, d_v, h,dropout=0.1):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEncoding(d__model)  # Transformer中位置编码时固定的，不需要学习
        self.layers = nn.ModuleList([EncoderLayer(device,d__model, d_k, d_v, h,dropout=0.1) for _ in range(n_layers)])

    def forward(self,inputs):
        enc_outputs = self.pos_emb(inputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        for layer in self.layers:
            enc_outputs=layer(enc_outputs)
        return enc_outputs

# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     input=torch.randn(30,100,768).to(device)
#     # sa = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=8)
#     encoder=Encoder(device=device,n_layers=6,d__model=768, d_k=768, d_v=768, h=8).to(device)
#     output=encoder(input)
#     print(output.shape)
