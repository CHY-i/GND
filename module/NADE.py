import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

class NADE(nn.Module):
    def __init__(self, n, hidden_dim, device="cpu", dtype="float", z2=False, *args, **kwargs):
        super().__init__()

        self.register_parameter("W", nn.Parameter(torch.randn(hidden_dim, n, device=device, dtype=dtype)))
        self.register_parameter("c", nn.Parameter(torch.zeros(hidden_dim, device=device, dtype=dtype)))
        self.register_parameter("V", nn.Parameter(torch.randn(n, hidden_dim, device=device, dtype=dtype)))
        self.register_parameter("b", nn.Parameter(torch.zeros(n, device=device, dtype=dtype)))

        self.n = n
        self.hidden_dim = hidden_dim
        self.device = device
        self.dtype = dtype
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
    
    def partial_forward(self, n_s, condition, device, dtype, k=1):
        with torch.no_grad():
            if n_s >1 :
                m = condition.size(1)
            else:
                m = condition.size(0)
            x = torch.zeros(n_s, self.n, device=device, dtype=dtype)
            x[:, :m] = condition
            for i in range(int(2*k)):
                h_i = torch.sigmoid(self.c + torch.einsum("hi,bi->bh", self.W[:, :m+i], x[:, :m+i]))
                prob = torch.sigmoid(self.b[i] + torch.einsum("h,bh->b", self.V[m+i, :], h_i))
                # print(prob.size())
                x[:, m+i] = torch.floor(2*prob)
        return x
    


    

# N  = NADE(10,  256)
# print(N.W.size())
# print(N.c.size())
# print(N.V.size())
# print(N.b.size())