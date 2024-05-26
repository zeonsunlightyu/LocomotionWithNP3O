import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(2 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ActionCausalTransformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        # action embedding
        self.action_embedding = nn.Sequential(
            nn.Linear(config.n_action, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd)
        )
        # obs embedding
        self.obs_embedding = nn.Sequential(
            nn.Linear(config.n_obs, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd)
        )
        # transformer 
        self.transformer = nn.ModuleDict(dict(
            #wpe = nn.Embedding(config.block_size, config.n_embd),
            wpe = PositionalEncoding(config.n_embd,dropout=0.0),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # mlp
        self.mlp_head = nn.Sequential(
            # nn.Linear(config.n_embd, config.n_embd),
            # nn.GELU(),
            nn.Linear(config.n_embd, config.n_action)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def convert_action_obs_sequence(action_history,obs_history):
        # permute to steps, num_envs, size
        num_envs,steps,size = action_history.size()
        # action t-4, action t-3, action t-2, action t-1
        action_history = action_history.permute(1,0,2)
        # obs t-3, obs t-2, obs t-1, obs t    
        obs_history = obs_history.permute(1,0,2)
        sequence = torch.stack((action_history,obs_history), dim=1).view(steps*2,num_envs,size)
        # action t-4, obs_t-3, action t-3, obs t-2, action t-2, obs t-1, action t-1, obs t 
        sequence = sequence.permute(1,0,2)
        # first step of sequence should be zero,should remove it,obs_t-3, action t-3, obs t-2, action t-2, obs t-1, action t-1, obs t 
        return sequence[:,1:,:]
    
    def forward(self, obs_history, action_history):

        # get embedding 
        obs_history_emb = self.obs_embedding(obs_history)
        action_history_emb = self.action_embedding(action_history)
        # transfrom embedding 
        tok_emb = self.convert_action_obs_sequence(action_history_emb,obs_history_emb)
   
        # device = tok_emb.device
        # n_envs,t,_ = tok_emb.size() 
        # assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        # # add positional embedding
        # pos_emb = self.transformer.wpe(pos).unsqueeze(0)
        # pos_emb = pos_emb.repeat(n_envs,1,1)
        # # pass through transformer
        # x = self.transformer.drop(tok_emb + pos_emb)
        x = self.transformer.wpe(tok_emb.permute(1,0,2))
        x = x.permute(1,0,2)
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x = self.mlp_head(x[:,-1,:])

        return x

class StateCausalTransformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        # obs embedding
        self.embedding = nn.Sequential(
            nn.Linear(config.n_obs, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd)
        )
        # transformer 
        self.transformer = nn.ModuleDict(dict(
            wpe = PositionalEncoding(config.n_embd,dropout=0.0),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # mlp
        self.mlp_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_action)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, obs_history):

        # get embedding 
        tok_emb = self.embedding(obs_history)

        x = self.transformer.wpe(tok_emb.permute(1,0,2))
        x = x.permute(1,0,2)
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x = self.mlp_head(x[:,-1,:])

        return x

class StateCausalHeadlessTransformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        # obs embedding
        self.embedding = nn.Sequential(
            nn.Linear(config.n_obs, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd)
        )
        # transformer 
        self.transformer = nn.ModuleDict(dict(
            wpe = PositionalEncoding(config.n_embd,dropout=0.0),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, obs_history):

        # get embedding 
        tok_emb = self.embedding(obs_history)

        x = self.transformer.wpe(tok_emb.permute(1,0,2))
        x = x.permute(1,0,2)
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        return x[:,-1,:]
    
class StateCausalClsTransformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        # obs embedding
        self.embedding = nn.Sequential(
            nn.Linear(config.n_obs, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_embd)
        )
        # transformer 
        self.transformer = nn.ModuleDict(dict(
            wpe = PositionalEncoding(config.n_embd,dropout=0.0),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # mlp
        self.mlp_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.Linear(config.n_embd, config.n_action)
        )

        # cls token
        self.cls_embedding = nn.Linear(3,config.n_embd)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, obs_history):

        # get embedding 
        tok_emb = self.embedding(obs_history)
        # get cls token
        cls_token = self.cls_embedding(obs_history[:,-1,6:9])
        # combine token emb 
        tok_emb = torch.cat([
                tok_emb,
                cls_token.unsqueeze(1)
            ], dim=1)

        x = self.transformer.wpe(tok_emb.permute(1,0,2))
        x = x.permute(1,0,2)
        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x = self.mlp_head(x[:,-1,:])

        return x