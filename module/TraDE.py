import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor



torch.set_default_dtype(torch.float64)


class PositionalEncoding(nn.Module):
    def __init__(self, n, d_model):
        super().__init__()
        den = torch.exp(- torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        pos = torch.arange(0, n).reshape(n, 1)
        pos_embedding = torch.zeros((n, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x):
        return x + self.pos_embedding



class LearnablePositionalEncoding(nn.Module):
    def __init__(self, n, d_model):
        super().__init__()
        self.positional_embedding = nn.Embedding(n, d_model)
        positions = torch.arange(n)
        self.register_buffer('positions', positions)

    def forward(self, x):
        return x + self.positional_embedding(self.positions)
    

class myTransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False, norm=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(myTransformerEncoderLayer, self).__init__()
        self.norm = norm
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        # self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        
        self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(myTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, is_causal: Optional[bool] = None,) -> Tensor:

        x = src
        if self.norm == True:
            if self.norm_first:
                x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
                x = x + self._ff_block(self.norm2(x))
            else:
                x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
                x = self.norm2(x + self._ff_block(x))
        else:
            x = x + self._sa_block(x, src_mask, src_key_padding_mask)
            x = x + self._ff_block(x)

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return x #self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.activation(self.linear1(x))) #self.linear2(self.dropout(self.activation(self.linear1(x))))
        return x #self.dropout2(x)


class TraDE_binary(nn.Module):
    """
    Transformers for density estimation or stat-mech problems
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.n = kwargs['n']
        self.d_model = kwargs['d_model']
        self.d_ff = kwargs['d_ff']
        self.n_layers = kwargs['n_layers']
        self.n_heads = kwargs['n_heads']
        self.device = kwargs['device']
        self.dropout = kwargs['dropout']

        self.fc_in = nn.Embedding(2, self.d_model)
        # self.positional_encoding = PositionalEncoding(self.n, self.d_model)
        self.positional_encoding = LearnablePositionalEncoding(self.n, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                   nhead=self.n_heads,
                                                   dim_feedforward=self.d_ff,
                                                   dropout=self.dropout,
                                                   batch_first=True,
                                                   norm_first=False,
                                                   #norm=True,
                                                   )
        self.encoder = nn.TransformerEncoder(encoder_layer, self.n_layers)
        # print(self.encoder)
        self.fc_out = nn.Linear(self.d_model, 1)

        self.register_buffer('mask', torch.ones(self.n, self.n))
        self.mask = torch.tril(self.mask, diagonal=0)
        self.mask = self.mask.masked_fill(self.mask == 0, float('-inf'))#.masked_fill(self.mask == 1, float(0.0))
        #self.mask[0, 0] = 0
    def forward(self, x):
        x = torch.cat((torch.ones(x.size(0), 1, device=self.device), x[:, :-1]), dim=1)
        x = F.relu(self.fc_in(x.to(int)))  # (batch_size, n, d_model)
        x = self.positional_encoding(x)
        x = self.encoder(x, mask=self.mask)
        return torch.sigmoid(self.fc_out(x)).squeeze(2)

    def log_prob(self, x):
        x_hat = self.forward(x)
        log_prob = torch.log(x_hat+1e-30) * x + torch.log(1 - x_hat+1e-30) * (1 - x)
        return log_prob.sum(dim=1)
    
    def samples(self, batch_size):
        samples = torch.zeros(batch_size, self.n, device=self.device, dtype=torch.float64)#torch.randint(0, 2, size=(batch_size, self.n), dtype=torch.float64, device=self.device)
        for i in range(self.n):
            x_hat = self.forward(samples)
            samples[:, i] = torch.bernoulli(x_hat[:, i])
        return samples

    def partial_samples(self, n_s, condition, device, dtype):
        with torch.no_grad():
            m = condition.size(0)
            x = torch.zeros(n_s, self.n, device=device, dtype=dtype)
            x[:, :m] = torch.vstack([condition]*n_s)
            for i in range(self.n-m):
                s_hat = self.forward(x)
                x[:, m+i] = torch.bernoulli(s_hat[:, m+i])
        return x
    
    def partial_forward(self, n_s, condition, device, dtype, k=1):
        with torch.no_grad():
            if n_s >1 :
                m = condition.size(1)
            else:
                m = condition.size(0)
            x = torch.zeros(n_s, self.n, device=device, dtype=dtype)
            x[:, :m] = condition
            for i in range(2*k):
                s_hat = self.forward(x)
                x[:, m+i] = torch.floor(2*s_hat[:, m+i])
        return x


class TraDE(nn.Module):
    """
    Transformers for density estimation or stat-mech problems
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.n = kwargs['n']
        self.d_model = kwargs['d_model']
        self.d_ff = kwargs['d_ff']
        self.n_layers = kwargs['n_layers']
        self.n_heads = kwargs['n_heads']
        self.device = kwargs['device']
        self.dropout = kwargs['dropout']
        self.nb = kwargs['nb'] # number of block spin

        self.fc_in = nn.Embedding(2**self.nb, self.d_model)
        # self.positional_encoding = PositionalEncoding(self.n, self.d_model)
        self.positional_encoding = LearnablePositionalEncoding(self.n, self.d_model)
        encoder_layer = myTransformerEncoderLayer(d_model=self.d_model,
                                                   nhead=self.n_heads,
                                                   dim_feedforward=self.d_ff,
                                                   dropout=self.dropout,
                                                   batch_first=True,
                                                   norm_first=False,
                                                   norm=True,
                                                   )
        self.encoder = nn.TransformerEncoder(encoder_layer, self.n_layers)
        self.fc_out = nn.Linear(self.d_model, 2**self.nb)

        self.register_buffer('mask', torch.ones(self.n, self.n))
        self.mask = torch.tril(self.mask, diagonal=0)
        self.mask = self.mask.masked_fill(self.mask == 0, float('-inf'))#.masked_fill(self.mask == 1, float(0.0))
        #self.mask[0, 0] = 0
    def forward(self, x):
        x = torch.cat((torch.ones(x.size(0), 1, device=self.device), x[:, :-1]), dim=1)
        x = F.relu(self.fc_in(x.to(int)))  # (batch_size, n, d_model)
        x = self.positional_encoding(x)
        x = self.encoder(x, mask=self.mask)#, mask=self.mask
        # x = self.fc_out(x)
        # print(x.size())
        return torch.softmax(self.fc_out(x), dim=-1)

    def log_prob(self, x):
        index = x.unsqueeze(-1).long()
        x_hat = self.forward(x)
        x_hat = x_hat.gather(2, index).squeeze()
        log_prob = torch.log(x_hat+1e-30).sum(1)
        return log_prob
    

    
    def partial_forward(self, batch, nbl, nrel, condition, device, dtype):
        with torch.no_grad():
            if batch >1 :
                m = condition.size(1)
            else:
                m = condition.size(0)
            x = torch.zeros(batch, self.n, device=device, dtype=dtype)
            x[:, :m] = condition
            for i in range(nbl):
                s_hat = self.forward(x)
                if nrel !=0 and i == nbl-1:
                    x[:, m+i] = torch.argmax(s_hat[:, m+i, :2**nrel], dim=-1)
                else:
                    x[:, m+i] = torch.argmax(s_hat[:, m+i, :], dim=-1)
            x = x[:, m:]
        return x


if __name__ == '__main__':
    import sys
    from os.path import abspath, dirname
    sys.path.append(abspath(dirname(__file__)).strip('module'))
    print(abspath(dirname(__file__)).strip('module'))
    L, dim= 2, 2
    beta = 0.8
    device='cpu'
    dtype=torch.float64
    n = L if dim==1 else L**2

    batch, epoch, lr = 1000, 5000, 0.001

    kwargs_dict = {
        'n': L if dim==1 else L**2,
        'd_model': 10,
        'd_ff': 10,
        'n_layers': 2,
        'n_heads': 2,
        'device': device,
        'dropout':0,#'cuda:5'
    }


    def test(trade):
        res = []
        s0 = torch.ones(1, kwargs_dict['n'], requires_grad=True).to(kwargs_dict['device']).int()
        s = F.relu(trade.fc_in(s0))
        s = trade.positional_encoding(s)
        s.retain_grad()
        for k in range(trade.n):
            x = trade.encoder(s, mask=trade.mask)
            x = torch.sigmoid(trade.fc_out(x)).squeeze(2)
            loss = x[0, k]
            loss.backward(retain_graph=True)
            grad = s.grad.sum(2)
            depends = (grad[0].numpy() != 0).astype(np.uint8)
            depends_ix = list(np.where(depends)[0])
            isok = k % trade.n not in depends_ix
            
            res.append((len(depends_ix), k, depends_ix, isok))
        
            # pretty print the dependencies
            res.sort()
        for nl, k, ix, isok in res:
            print("output %2d depends on inputs: %70s : %s" % (k, ix, "OK" if isok else "NOTOK"))



    T = TraDE_binary(**kwargs_dict).to(kwargs_dict['device'])
    test(T)
    print(T)
    
