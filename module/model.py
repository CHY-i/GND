import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel
from utils import scaled_dot_product_attention


class SimpleMADE(BaseModel):
    def __init__(self, n, device="cpu", dtype="float", z2=False, *args, **kwargs):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(n, n) / math.sqrt(n))
        self.bias = nn.Parameter(torch.zeros(n))
        mask = torch.tril(torch.ones(n, n), diagonal=-1)
        self.register_buffer("mask", mask)

        self.n = n
        self.device = device
        self.dtype = torch.float64 if dtype == "double" else torch.float32
        self.z2 = z2

        nn.init.xavier_normal_(self.weight)

    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n})"

    def _forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)

    def forward(self, x):
        logits = self._forward(x)
        log_prob = -F.binary_cross_entropy_with_logits(logits, x, reduction="none")

        if self.z2:
            x_inv = 1 - x
            logits_inv = self._forward(x_inv)
            log_prob_inv = -F.binary_cross_entropy_with_logits(logits_inv, x_inv, reduction="none")
            log_prob = torch.logsumexp(torch.stack([log_prob, log_prob_inv], dim=0), dim=0)
            log_prob = log_prob - math.log(2)

        return log_prob.sum(-1)

    @torch.no_grad()
    def sample(self, batch_size):
        x = torch.zeros(batch_size, self.n, dtype=self.dtype, device=self.device)
        for i in range(self.n):
            logits = self._forward(x)[:, i]
            x[:, i] = torch.bernoulli(torch.sigmoid(logits))

        if self.z2:
            mask = torch.rand(batch_size) < 0.5
            x[mask] = 1 - x[mask]

        return x


class GRU(BaseModel):
    def __init__(self, n, hidden_dim, device="cpu", dtype="float", *args, **kwargs):
        super().__init__()

        self.gru_cell = nn.GRUCell(2, hidden_dim)
        self.fc_layer = nn.Linear(hidden_dim, 1)

        self.n = n
        self.hidden_dim = hidden_dim
        self.device = device
        self.dtype = torch.float64 if dtype == "double" else torch.float32

        # TODO: initialize parameters

    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n}, hidden_dim={self.hidden_dim})"

    def _forward(self, x, h=None):
        x = torch.stack([x, 1 - x], dim=1)  # 1 -> (1, 0), 0 -> (0, 1), (batch_size, 2)
        h_next = self.gru_cell(x, h)  # h_{i+1}
        logits = self.fc_layer(h_next).squeeze(1)
        return h_next, logits

    def forward(self, x):
        log_prob_list = []
        x = torch.cat([torch.zeros(x.shape[0], 1, dtype=self.dtype, device=self.device), x], dim=1)  # cat x_0
        h = torch.zeros(x.shape[0], self.hidden_dim, dtype=self.dtype, device=self.device)  # h_0
        for i in range(self.n):
            h, logits = self._forward(x[:, i], h)
            log_prob = -F.binary_cross_entropy_with_logits(logits, x[:, i + 1], reduction="none")
            log_prob_list.append(log_prob)
        return torch.stack(log_prob_list, dim=1).sum(dim=1)

    @torch.no_grad()
    def sample(self, batch_size):
        x = torch.zeros(batch_size, self.n + 1, dtype=self.dtype, device=self.device)
        for i in range(self.n):
            h, logits = self._forward(x[:, i], h=None if i == 0 else h)
            x[:, i + 1] = torch.bernoulli(torch.sigmoid(logits))
        return x[:, 1:]


class NADE(BaseModel):
    def __init__(self, n, hidden_dim, device="cpu", dtype="float", z2=False, *args, **kwargs):
        super().__init__()

        self.register_parameter("W", nn.Parameter(torch.randn(hidden_dim, n)))
        self.register_parameter("c", nn.Parameter(torch.zeros(hidden_dim)))
        self.register_parameter("V", nn.Parameter(torch.randn(n, hidden_dim)))
        self.register_parameter("b", nn.Parameter(torch.zeros(n)))

        self.n = n
        self.hidden_dim = hidden_dim
        self.device = device
        self.dtype = torch.float64 if dtype == "double" else torch.float32
        self.z2 = z2

        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.V)

    def __repr__(self):
        return f"{self.__class__.__name__}(n={self.n}, hidden_dim={self.hidden_dim})"

    def _forward(self, x):
        logits_list = []
        for i in range(self.n):
            h_i = torch.sigmoid(self.c + torch.einsum("hi,bi->bh", self.W[:, :i], x[:, :i]))
            logits = self.b[i] + torch.einsum("h,bh->b", self.V[i, :], h_i)
            logits_list.append(logits)

        return torch.stack(logits_list, dim=1)

    def forward(self, x):
        logits = self._forward(x)
        log_prob = -F.binary_cross_entropy_with_logits(logits, x, reduction="none")

        if self.z2:
            x_inv = 1 - x
            logits_inv = self._forward(x_inv)
            log_prob_inv = -F.binary_cross_entropy_with_logits(logits_inv, x_inv, reduction="none")
            log_prob = torch.logsumexp(torch.stack([log_prob, log_prob_inv], dim=0), dim=0)
            log_prob = log_prob - math.log(2)

        return log_prob.sum(-1)

    @torch.no_grad()
    def sample(self, batch_size):
        x = torch.zeros(batch_size, self.n, dtype=self.dtype, device=self.device)
        for i in range(self.n):
            h_i = torch.sigmoid(self.c + torch.einsum("hi,bi->bh", self.W[:, :i], x[:, :i]))
            logits = self.b[i] + torch.einsum("h,bh->b", self.V[i, :], h_i)
            x[:, i] = torch.bernoulli(torch.sigmoid(logits))

        if self.z2:
            mask = torch.rand(batch_size) < 0.5
            x[mask] = 1 - x[mask]

        return x


@dataclass
class TransformerConfig:
    phy_dim: int = 2
    max_len: int = 20
    emb_dim: int = 32
    mlp_dim: int = 128
    num_heads: int = 2
    num_layers: int = 1
    use_bias: bool = False
    device: str = "cpu"


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.emb_dim % config.num_heads == 0
        self.W_in = nn.Linear(config.emb_dim, 3 * config.emb_dim, bias=config.use_bias)
        self.W_out = nn.Linear(config.emb_dim, config.emb_dim, bias=config.use_bias)
        self.emb_dim = config.emb_dim
        self.num_heads = config.num_heads
        self.d_k = config.emb_dim // config.num_heads

    def forward(self, x):
        B, T, C = x.size()  # batch size, length, embedding dimension
        assert C == self.emb_dim

        Q, K, V = self.W_in(x).split(self.emb_dim, dim=2)  # (B, T, C)
        Q = Q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, T, d_k)
        K = K.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, T, d_k)
        V = V.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, T, d_k)
        y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)  # (B, num_heads, T, d_k)
        # y = scaled_dot_product_attention(Q, K, V, is_causal=True)  # (B, num_heads, T, d_k)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.W_out(y)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Linear(config.emb_dim, config.mlp_dim, bias=config.use_bias)
        self.l2 = nn.Linear(config.mlp_dim, config.emb_dim, bias=config.use_bias)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)  # gelu()
        x = self.l2(x)

        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.emb_dim, bias=config.use_bias)
        self.mha = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.emb_dim, bias=config.use_bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


# class Block(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.mha = MultiHeadAttention(config)
#         self.mlp = MLP(config)

#     def forward(self, x):
#         x = x + self.mha(x)
#         x = x + self.mlp(x)

#         return x


class TransformerARModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.phy_dim, config.emb_dim),
                wpe=nn.Embedding(config.max_len, config.emb_dim),
                blocks=nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
                layer_norm=nn.LayerNorm(config.emb_dim, bias=config.use_bias),
                linear=nn.Linear(config.emb_dim, config.phy_dim, bias=False),
            )
        )
        # TODO: investigate
        self.transformer.wte.weight = self.transformer.linear.weight  # weight-typing

        # initialize parameters
        self.apply(self._init_weights)

    # TODO: investigate
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _forward(self, x, target=None):
        # x: (x_0, x_1, ..., x_{T-1}), shape (B, T)
        # target: (x_1, x_2, ..., x_T), shape (B, T)
        # device = x.device
        B, T = x.size()  # batch size, length
        assert T <= self.config.max_len
        pos = torch.arange(0, T, dtype=torch.long, device=self.config.device)

        # forward pass
        tok_emb = self.transformer.wte(x)  # (B, T, emb_dim)
        pos_emb = self.transformer.wpe(pos)  # (T, emb_dim)
        x = tok_emb + pos_emb
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.layer_norm(x)

        if target is not None:
            logits = self.transformer.linear(x)  # (B, T, phy_dim)
            loss = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")  # (B, T)
        else:
            logits = self.transformer.linear(x[:, [-1], :])  # (B, 1, phy_dim)
            loss = None

        return logits, loss

    @torch.no_grad()
    def sample(self, batch_size):
        # use auxiliary variable x_0=0 to sample
        x = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=self.config.device)
        for _ in range(self.config.max_len):
            logits, _ = self._forward(x)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, x_next), dim=1)

        return x[:, 1:]

    @staticmethod
    def shift_inputs(x):
        # add an auxiliary variable x_0=0 and return [x_0, x_1, ..., x_{N-1}]
        device = x.device
        B, T = x.size()
        aux_x = torch.zeros(size=(B, 1), dtype=torch.long, device=device)

        return torch.cat((aux_x, x[:, :-1]), dim=1)

    def forward(self, x):
        B, T = x.size()
        assert T == self.config.max_len
        _, loss = self._forward(self.shift_inputs(x), x)  # (B, T)

        return -loss.sum(-1)
