import torch.nn as nn
import math
import torch
from copy import deepcopy
from collections import defaultdict
from torch.nn import functional as F

from modules.transformer_modules import StateCausalTransformer

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
    
def mlp_factory(activation, input_dims, out_dims, hidden_dims,last_act=False):
    layers = []
    layers.append(nn.Linear(input_dims, hidden_dims[0]))
    layers.append(activation)
    for l in range(len(hidden_dims)-1):
        layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
        layers.append(activation)

    if out_dims:
        layers.append(nn.Linear(hidden_dims[-1], out_dims))
    if last_act:
        layers.append(activation)

    return layers

def mlp_layernorm_factory(activation, input_dims, out_dims, hidden_dims,last_act=False):
    layers = []
    layers.append(nn.Linear(input_dims, hidden_dims[0]))
    layers.append(nn.LayerNorm(hidden_dims[0]))
    layers.append(activation)

    for l in range(len(hidden_dims)-1):
        layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
        layers.append(nn.LayerNorm(hidden_dims[l + 1]))
        layers.append(activation)

    if out_dims:
        layers.append(nn.Linear(hidden_dims[-1], out_dims))
    if last_act:
        layers.append(activation)

    return layers

def mlp_batchnorm_factory(activation, input_dims, out_dims, hidden_dims,last_act=False,bias=True):
    layers = []
    layers.append(nn.Linear(input_dims, hidden_dims[0],bias=bias))
    layers.append(nn.BatchNorm1d(hidden_dims[0]))
    layers.append(activation)
    for l in range(len(hidden_dims)-1):
        layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1],bias=bias))
        layers.append(nn.BatchNorm1d(hidden_dims[l + 1]))
        layers.append(activation)

    if out_dims:
        layers.append(nn.Linear(hidden_dims[-1], out_dims,bias=bias))
    if last_act:
        layers.append(activation)

    return layers

class RnnStateHistoryEncoder(nn.Module):
    def __init__(self,activation_fn, input_size, encoder_dims,hidden_size,output_size):
        super(RnnStateHistoryEncoder,self).__init__()
        self.activation_fn = activation_fn
        self.encoder_dims = encoder_dims
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(*mlp_factory(activation=activation_fn,
                                   input_dims=input_size,
                                   hidden_dims=encoder_dims,
                                   out_dims=output_size))
        
        self.rnn = nn.GRU(input_size=output_size,
                           hidden_size=hidden_size,
                           batch_first=True)
        
    def forward(self,obs):
        obs = self.encoder(obs)
        out, h_n = self.rnn(obs)
        return out
    
class RnnBarlowTwinsStateHistoryEncoder(nn.Module):
    def __init__(self,activation_fn, input_size, encoder_dims,hidden_size,output_size):
        super(RnnBarlowTwinsStateHistoryEncoder,self).__init__()
        self.activation_fn = activation_fn
        self.encoder_dims = encoder_dims
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(*mlp_factory(activation=activation_fn,
                                   input_dims=input_size,
                                   hidden_dims=encoder_dims,
                                   out_dims=int(hidden_size/2)))
        
        self.rnn = nn.GRU(input_size=int(hidden_size/2),
                           hidden_size=hidden_size,
                           batch_first=True,
                           num_layers = 2)
        
        self.final_layer = nn.Linear(hidden_size,output_size)
        
    def forward(self,obs):
        h_0 = torch.zeros(2,obs.size(0),self.hidden_size,device=obs.device).requires_grad_().half()
        obs = self.encoder(obs)
        out, h_n = self.rnn(obs,h_0)
        latent = self.final_layer(out[:,-1,:])
        return latent
    
class RnnEncoder(nn.Module):
    def __init__(self,input_size, hidden_size,output_size):
        super(RnnEncoder,self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.rnn = nn.GRU(input_size=input_size,
                           hidden_size=hidden_size,
                           batch_first=True,
                           num_layers = 2)
        
        self.final_layers = nn.Sequential(nn.Linear(hidden_size, int(hidden_size/2), bias=False),
                            nn.BatchNorm1d(int(hidden_size/2)),
                            nn.ReLU(inplace=True),
                            nn.Linear(int(hidden_size/2),output_size, bias=False),
                            nn.BatchNorm1d(output_size,affine=False))
        
    def forward(self,obs):
        h_0 = torch.zeros(2,obs.size(0),self.hidden_size,device=obs.device).requires_grad_()
        out, h_n = self.rnn(obs,h_0)
        latent = self.final_layers(out[:,-1,:])
        return latent
    
# class RnnDoubleHeadEncoder(nn.Module):
#     def __init__(self,input_size, hidden_size,output_size):
#         super(RnnDoubleHeadEncoder,self).__init__()

#         self.output_size = output_size
#         self.hidden_size = hidden_size
        
#         self.rnn = nn.GRU(input_size=input_size,
#                            hidden_size=hidden_size,
#                            batch_first=True,
#                            num_layers = 2)
        
#         self.final_layers = nn.Sequential(nn.Linear(hidden_size, int(hidden_size/2), bias=False),
#                             nn.BatchNorm1d(int(hidden_size/2)),
#                             nn.ReLU(inplace=True),
#                             nn.Linear(int(hidden_size/2),output_size, bias=False),
#                             nn.BatchNorm1d(output_size,affine=False))
#         self.vel_est = nn.Sequential(nn.Linear(hidden_size, int(hidden_size/2)),
#                             nn.ReLU(inplace=True),
#                             nn.Linear(int(hidden_size/2),3))
        
#     def forward(self,obs):
#         h_0 = torch.zeros(2,obs.size(0),self.hidden_size,device=obs.device).requires_grad_()
#         out, h_n = self.rnn(obs,h_0)
#         latent = self.final_layers(out[:,-1,:])
#         vel = self.vel_est(out[:,-1,:])
#         return latent,vel
    
class RnnDoubleHeadEncoder(nn.Module):
    def __init__(self,input_size, hidden_size,output_size):
        super(RnnDoubleHeadEncoder,self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.rnn = nn.GRU(input_size=input_size,
                           hidden_size=hidden_size,
                           batch_first=True,
                           num_layers = 2)
        
        self.final_layers = nn.Sequential(nn.Linear(hidden_size, int(hidden_size/2)),
                            nn.LayerNorm(int(hidden_size/2)),
                            nn.ELU(inplace=True),
                            nn.Linear(int(hidden_size/2),output_size))
        
        self.vel_est = nn.Sequential(nn.Linear(hidden_size, int(hidden_size/2)),
                            nn.LayerNorm(int(hidden_size/2)),
                            nn.ELU(inplace=True),
                            nn.Linear(int(hidden_size/2),3))
        
    def forward(self,obs):
        h_0 = torch.zeros(2,obs.size(0),self.hidden_size,device=obs.device).requires_grad_()
        out, h_n = self.rnn(obs,h_0)
        latent = self.final_layers(out[:,-1,:])
        vel = self.vel_est(out[:,-1,:])
        return latent,vel
    
    
class AutoEncoder(nn.Module):
    def __init__(self,activation_fn, input_size, encoder_dims,latent_dim,decoder_dims,output_size):
        super(AutoEncoder,self).__init__()
        self.activation_fn = activation_fn
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.input_size = input_size
        self.output_size = output_size
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(*mlp_factory(activation=activation_fn,
                                   input_dims=input_size,
                                   hidden_dims=encoder_dims,
                                   out_dims=latent_dim))
        
        self.decoder = nn.Sequential(*mlp_factory(activation=activation_fn,
                                   input_dims=latent_dim,
                                   hidden_dims=decoder_dims,
                                   out_dims=output_size))
        
    def forward(self,obs):
        return self.encode(obs)

    def encode(self,obs):
        latent = self.encoder(obs)
        return latent

    def decode(self,latent):
        out = self.decoder(latent)
        return out

class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False,final_act=True):
        # self.device = device
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        if final_act:
            self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )
        else:
            self.linear_output = nn.Sequential(nn.Linear(channel_size * 3, output_size))

    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output
    
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

# class BetaVAE(nn.Module):

#     def __init__(self,
#                  in_dim= 45*5,
#                  latent_dim = 19,
#                  encoder_hidden_dims = [128,64],
#                  decoder_hidden_dims = [64,128],
#                  output_dim = 45,
#                  beta: int = 0.1) -> None:
        
#         super(BetaVAE, self).__init__()

#         self.latent_dim = latent_dim
#         self.beta = beta
        
#         encoder_layers = []
#         encoder_layers.append(nn.Sequential(nn.Linear(in_dim, encoder_hidden_dims[0]),
#                                             nn.LayerNorm(encoder_hidden_dims[0]),
#                                             nn.ELU()))
#         for l in range(len(encoder_hidden_dims)-1):
#             encoder_layers.append(nn.Sequential(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l+1]),
#                                         nn.LayerNorm(encoder_hidden_dims[l+1]),
#                                         nn.ELU()))
#         self.encoder = nn.Sequential(*encoder_layers)

#         self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim-3)
#         self.fc_var = nn.Linear(encoder_hidden_dims[-1], latent_dim-3)
#         self.fc_vel = nn.Linear(encoder_hidden_dims[-1], 3)


#         # Build Decoder
#         decoder_layers = []
#         decoder_layers.append(nn.Sequential(nn.Linear(latent_dim-3, decoder_hidden_dims[0]),
#                                             nn.LayerNorm(decoder_hidden_dims[0]),
#                                             nn.ELU()))
#         for l in range(len(decoder_hidden_dims)):
#             if l == len(decoder_hidden_dims) - 1:
#                 decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
#             else:
#                 decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
#                                         nn.LayerNorm(decoder_hidden_dims[l+1]),
#                                         nn.ELU()))

#         self.decoder = nn.Sequential(*decoder_layers)

#         self.kl_weight = beta

#     def encode(self, input):
       
#         result = self.encoder(input)
#         mu = self.fc_mu(result)
#         log_var = self.fc_var(result)
#         vel = self.fc_vel(result)

#         return [mu,log_var,vel]
    
#     def get_latent(self,input):
#         mu,log_var,vel = self.encode(input)
#         z = self.reparameterize(mu, log_var)
#         return z,vel

#     def decode(self,z):
#         result = self.decoder(z)
#         return result

#     def reparameterize(self, mu, logvar):
       
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return eps * std + mu

#     def forward(self, input):
#         mu,log_var,vel = self.encode(input)
#         z = self.reparameterize(mu, log_var)
#         return  [self.decode(z),z, mu, log_var, vel]
    
#     def loss_fn(self,y, y_hat, mean, logvar):
#         recons_loss = F.mse_loss(y_hat, y)
#         kl_loss = torch.mean(
#             -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar), 1), 0)

#         return recons_loss,kl_loss*self.kl_weight

class BetaVAE(nn.Module):

    def __init__(self,
                 in_dim= 45*5,
                 latent_dim = 16,
                 encoder_hidden_dims = [128,64],
                 decoder_hidden_dims = [64,128],
                 output_dim = 45,
                 beta: int = 0.1) -> None:
        
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        
        encoder_layers = []
        encoder_layers.append(nn.Sequential(nn.Linear(in_dim, encoder_hidden_dims[0]),
                                            nn.BatchNorm1d(encoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(encoder_hidden_dims)-1):
            encoder_layers.append(nn.Sequential(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l+1]),
                                        nn.BatchNorm1d(encoder_hidden_dims[l+1]),
                                        nn.ELU()))
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.fc_vel = nn.Linear(encoder_hidden_dims[-1],3)

        # Build Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            nn.BatchNorm1d(decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
                                        nn.BatchNorm1d(decoder_hidden_dims[l+1]),
                                        nn.ELU()))

        self.decoder = nn.Sequential(*decoder_layers)

        self.kl_weight = beta

    def encode(self, input):
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        vel = self.fc_vel(result)

        return [mu,log_var,vel]
    
    def get_latent(self,input):
        mu,log_var,vel = self.encode(input)
        
        #z = self.reparameterize(mu, log_var)
        return mu,vel

    def decode(self,z):
        result = self.decoder(z)
        return result

    def reparameterize(self, mu, logvar):
       
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        
        mu,log_var,vel = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z),z, mu, log_var, vel]
    
    def loss_fn(self,y, y_hat, mean, logvar):
     
        # recons_loss = 0.5*F.mse_loss(y_hat,y,reduction="none").sum(dim=-1)
        # kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), -1)
        # loss = (recons_loss + self.beta * kl_loss).mean(dim=0)

        recons_loss = F.mse_loss(y_hat,y)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), -1))
        loss = recons_loss + self.beta * kl_loss

        return loss
    
class MAE(nn.Module):

    def __init__(self,
                 in_dim= 45*5,
                 latent_dim = 32,
                 encoder_hidden_dims = [128,64],
                 decoder_hidden_dims = [64,128],
                 output_dim = 45+3) -> None:
        
        super(MAE, self).__init__()

        self.latent_dim = latent_dim
        
        encoder_layers = []
        encoder_layers.append(nn.Sequential(nn.Linear(in_dim, encoder_hidden_dims[0]),
                                            nn.LayerNorm(encoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(encoder_hidden_dims)-1):
            encoder_layers.append(nn.Sequential(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l+1]),
                                        nn.LayerNorm(encoder_hidden_dims[l+1]),
                                        nn.ELU()))
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)

        # Build Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            nn.LayerNorm(decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l],in_dim))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
                                        nn.LayerNorm(decoder_hidden_dims[l+1]),
                                        nn.ELU()))

        self.decoder = nn.Sequential(*decoder_layers)

        # build est
        est_decoder_layers = []
        est_decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            nn.LayerNorm(decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                est_decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
            else:
                est_decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
                                        nn.LayerNorm(decoder_hidden_dims[l+1]),
                                        nn.ELU()))

        self.est_decoder = nn.Sequential(*est_decoder_layers)

        self.random_mask = nn.Dropout1d(p=0.25)
    
    def get_latent(self,input):
        z = self.encode(input)
        return z

    def encode(self,input):
        latent = self.encoder(input)
        latent = self.fc_mu(latent)
        return latent
    
    def decode(self,latent):
        input_hat = self.decoder(latent)
        est_hat = self.est_decoder(latent)
        return input_hat,est_hat
    
    def forward(self, input):
        input_masked = self.random_mask(input)
        z = self.encode(input_masked)
        input_hat,est_hat = self.decode(z)

        return  input_hat,est_hat
    
    def loss_fn(self,y, y_hat, est, est_hat):
        recons_loss = F.mse_loss(y_hat, y)
        recons_est_loss = F.mse_loss(est_hat,est)
        return recons_loss + recons_est_loss
    
class Quantizer(nn.Module):
    def __init__(self,embedding_dim,num_embeddings):
        nn.Module.__init__(self)

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

        self.embeddings.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )
        # self.linear_proj = nn.Linear(embedding_dim,int(embedding_dim/2))

    def forward(self, z: torch.Tensor):
        # z_norm = F.normalize(self.linear_proj(z))
        #emb_norm = F.normalize(self.linear_proj(self.embeddings.weight))

        # z_norm = F.normalize(z)
        # emb_norm = F.normalize(self.embeddings.weight)

        distances = (
            (z ** 2).sum(dim=-1, keepdim=True)
            + (self.embeddings.weight**2).sum(dim=-1)
            - 2 * z @ self.embeddings.weight.T
        )

        closest = distances.argmin(-1).unsqueeze(-1)

        one_hot_encoding = (
            F.one_hot(closest, num_classes=self.num_embeddings).type_as(z)
            .squeeze(1)
        )

        # quantization
        quantized = one_hot_encoding @ self.embeddings.weight

        return quantized
    
class QuantizerNorm(nn.Module):
    def __init__(self,embedding_dim,num_embeddings):
        nn.Module.__init__(self)

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

        # self.embeddings.weight.data.uniform_(
        #     -1 / self.num_embeddings, 1 / self.num_embeddings
        # )
        self.linear_proj = nn.Linear(embedding_dim,int(embedding_dim/2))

    def forward(self, z: torch.Tensor):
        z_ = self.linear_proj(z)
        emb_ = self.linear_proj(self.embeddings.weight)

        z_norm = F.normalize(z_, dim=-1) 
        emb_norm = F.normalize(emb_,dim=-1)

        distances = (
            (z_norm ** 2).sum(dim=-1, keepdim=True)
            + (emb_norm**2).sum(dim=-1)
            - 2 * z_norm @ emb_norm.T
        )

        closest = distances.argmin(-1).unsqueeze(-1)

        one_hot_encoding = (
            F.one_hot(closest, num_classes=self.num_embeddings)
            .type(torch.float)
            .squeeze(1)
        )

        # quantization
        quantized = one_hot_encoding @ self.embeddings.weight
        quantized = F.normalize(quantized,dim=-1)
        return quantized
    
class QuantizerEMA(nn.Module):
    def __init__(self,embedding_dim,num_embeddings):
        nn.Module.__init__(self)

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.decay = 0.99

        embeddings = torch.empty(self.num_embeddings, self.embedding_dim)
        embeddings.data.normal_()

        self.register_buffer("cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer(
            "ema_embed", torch.zeros(self.num_embeddings, self.embedding_dim)
        )

        self.register_buffer("embeddings", embeddings)

        self.linear_proj = nn.Linear(embedding_dim,int(embedding_dim/2))

    def update_codebook(self,z,one_hot_encoding):
        n_i = torch.sum(one_hot_encoding, dim=0)

        self.cluster_size = self.cluster_size * self.decay + n_i * (1 - self.decay)

        dw = one_hot_encoding.T @ z.reshape(-1, self.embedding_dim)

        ema_embed = self.ema_embed * self.decay + dw * (1 - self.decay)

        n = torch.sum(self.cluster_size)

        self.cluster_size = (
            (self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
        )

        self.embeddings.data.copy_(ema_embed / self.cluster_size.unsqueeze(-1))
        self.ema_embed.data.copy_(ema_embed)

    def forward(self, z: torch.Tensor):

        z_ = self.linear_proj(z)
        emb_ = self.linear_proj(self.embeddings)

        # z_norm = F.normalize(z_, dim=-1) 
        # emb_norm = F.normalize(emb_,dim=-1)

        distances = (
            (z_ ** 2).sum(dim=-1, keepdim=True)
            + (emb_**2).sum(dim=-1)
            - 2 * z_ @ emb_.T
        )

        closest = distances.argmin(-1).unsqueeze(-1)

        one_hot_encoding = (
            F.one_hot(closest, num_classes=self.num_embeddings)
            .type(torch.float)
            .squeeze(1)
        )

        # quantization
        quantized = one_hot_encoding @ self.embeddings


        return quantized,one_hot_encoding
    
    
# class VQVAE(nn.Module):

#     def __init__(self,
#                  in_dim= 45*5,
#                  latent_dim = 16,
#                  encoder_hidden_dims = [128,64],
#                  decoder_hidden_dims = [64,128],
#                  output_dim = 45) -> None:
        
#         super(VQVAE, self).__init__()

#         self.latent_dim = latent_dim
        
#         encoder_layers = []
#         encoder_layers.append(nn.Sequential(nn.Linear(in_dim, encoder_hidden_dims[0]),
#                                             nn.ELU()))
#         for l in range(len(encoder_hidden_dims)-1):
#             encoder_layers.append(nn.Sequential(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l+1]),
#                                         nn.ELU()))
#         self.encoder = nn.Sequential(*encoder_layers)

#         self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
#         self.normalize = nn.BatchNorm1d(latent_dim,affine=False)
#         self.fc_vel = nn.Linear(encoder_hidden_dims[-1], 3)

#         # Build Decoder
#         decoder_layers = []
#         decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
#                                             nn.ELU()))
#         for l in range(len(decoder_hidden_dims)):
#             if l == len(decoder_hidden_dims) - 1:
#                 decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
#             else:
#                 decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
#                                         nn.ELU()))

#         self.decoder = nn.Sequential(*decoder_layers)
#         self.embedding_dim = latent_dim
#         #self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=64)
#         #self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=256)
#         self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=512)
    
#     def get_latent(self,input):
#         z,vel = self.encode(input)
#         #z = F.normalize(z,dim=-1,p=2)
#         z = self.normalize(z)
#         # z = z.reshape(z.shape[0], 1, 1, -1)
#         # z = z.permute(0, 2, 3, 1)
#         # quantize = self.quantizer(z)
#         # quantize = quantize.reshape(z.shape[0], -1)
#         return z,vel

#     def encode(self,input):
#         latent = self.encoder(input)
#         z = self.fc_mu(latent)
#         #z = F.normalize(z,dim=-1,p=2)
#         z = self.normalize(z)
#         vel = self.fc_vel(latent)
#         return z,vel
    
#     def decode(self,quantized,z):
#         quantized = z + (quantized - z).detach()
#         input_hat = self.decoder(quantized)
#         return input_hat
    
#     def forward(self, input):
#         z,vel = self.encode(input)
#         z = z.reshape(z.shape[0], 1, 1, -1)
#         z = z.permute(0, 2, 3, 1)
#         quantize = self.quantizer(z)
#         quantize = quantize.reshape(z.shape[0], -1)
#         z = z.reshape(z.shape[0],-1)
#         input_hat = self.decode(quantize,z)
#         return input_hat,quantize,z,vel
    
#     def loss_fn(self,y, y_hat,quantized,z):
#         recon_loss = F.mse_loss(y_hat, y, reduction="none").sum(dim=-1)
        
#         commitment_loss = F.mse_loss(
#             quantized.detach(),
#             z,
#             reduction="sum",
#         )

#         embedding_loss = F.mse_loss(
#             quantized,
#             z.detach(),
#             reduction="sum",
#         )

#         vq_loss = 0.25*commitment_loss + embedding_loss

#         return (recon_loss + vq_loss).mean(dim=0)
    
# class VQVAE_EMA(nn.Module):

#     def __init__(self,
#                  in_dim= 45*5,
#                  latent_dim = 16,
#                  encoder_hidden_dims = [128,64],
#                  decoder_hidden_dims = [64,128],
#                  output_dim = 45) -> None:
        
#         super(VQVAE_EMA, self).__init__()

#         self.latent_dim = latent_dim
        
#         encoder_layers = []
#         encoder_layers.append(nn.Sequential(nn.Linear(in_dim, encoder_hidden_dims[0]),
#                                             nn.ELU()))
#         for l in range(len(encoder_hidden_dims)-1):
#             encoder_layers.append(nn.Sequential(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l+1]),
#                                         nn.ELU()))
#         self.encoder = nn.Sequential(*encoder_layers)

#         self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
#         # self.normalize = nn.BatchNorm1d(latent_dim,affine=False)
#         self.fc_vel = nn.Linear(encoder_hidden_dims[-1], 3)

#         # Build Decoder
#         decoder_layers = []
#         decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
#                                             nn.ELU()))
#         for l in range(len(decoder_hidden_dims)):
#             if l == len(decoder_hidden_dims) - 1:
#                 decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
#             else:
#                 decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
#                                         nn.ELU()))

#         self.decoder = nn.Sequential(*decoder_layers)
#         self.embedding_dim = latent_dim
#         #self.quantizer = QuantizerEMA(embedding_dim=self.embedding_dim,num_embeddings=16)
#         #self.quantizer = QuantizerEMA(embedding_dim=self.embedding_dim,num_embeddings=512)
#         self.quantizer = QuantizerEMA(embedding_dim=self.embedding_dim,num_embeddings=64)
    
#     def get_latent(self,input):
#         z,vel = self.encode(input)
#         z = F.normalize(z,dim=-1,p=2)
#         #z = self.normalize(z)
#         # z = z.reshape(z.shape[0], 1, 1, -1)
#         # z = z.permute(0, 2, 3, 1)
#         # quantize,_ = self.quantizer(z)
#         # quantize = quantize.reshape(z.shape[0], -1)
#         return z,vel

#     def encode(self,input):
#         latent = self.encoder(input)
#         z = self.fc_mu(latent)
#         z = F.normalize(z,dim=-1,p=2)
#         #z = self.normalize(z)
#         vel = self.fc_vel(latent)
#         return z,vel
    
#     def decode(self,quantized,z):
#         quantized = z + (quantized - z).detach()
#         input_hat = self.decoder(quantized)
#         return input_hat
    
#     def forward(self, input):
#         z,vel = self.encode(input)
#         z = z.reshape(z.shape[0], 1, 1, -1)
#         z = z.permute(0, 2, 3, 1)
#         quantize,onehot_encode = self.quantizer(z)
#         quantize = quantize.reshape(z.shape[0], -1)
#         z = z.reshape(z.shape[0],-1)
#         input_hat = self.decode(quantize,z)
#         return input_hat,quantize,z,vel,onehot_encode
    
#     def loss_fn(self,y, y_hat,quantized,z,onehot_encode):
#         recon_loss = F.mse_loss(y_hat, y, reduction="none").sum(dim=-1)
        
#         commitment_loss = F.mse_loss(
#             quantized.detach(),
#             z,
#             reduction="sum",
#         )

#         # embedding_loss = F.mse_loss(
#         #     quantized,
#         #     z.detach(),
#         #     reduction="sum",
#         # )
#         self.quantizer.update_codebook(z,onehot_encode)

#         vq_loss = 0.25*commitment_loss #+ embedding_loss

#         return (recon_loss + vq_loss).mean(dim=0)

# class VQVAE(nn.Module):

#     def __init__(self,
#                  in_dim= 45*5,
#                  latent_dim = 16,
#                  encoder_hidden_dims = [128,64],
#                  decoder_hidden_dims = [64,128],
#                  output_dim = 45) -> None:
        
#         super(VQVAE, self).__init__()

#         self.latent_dim = latent_dim
        
#         encoder_layers = []
#         encoder_layers.append(nn.Sequential(nn.Linear(in_dim, encoder_hidden_dims[0]),
#                                             nn.ELU()))
#         for l in range(len(encoder_hidden_dims)-1):
#             encoder_layers.append(nn.Sequential(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l+1]),
#                                         nn.ELU()))
#         self.encoder = nn.Sequential(*encoder_layers)

#         self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
#         self.fc_vel = nn.Linear(encoder_hidden_dims[-1], 3)
#         # self.normalize = nn.BatchNorm1d(latent_dim,affine=False)

#         # Build Decoder
#         decoder_layers = []
#         decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
#                                             nn.ELU()))
#         for l in range(len(decoder_hidden_dims)):
#             if l == len(decoder_hidden_dims) - 1:
#                 decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
#             else:
#                 decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
#                                         nn.ELU()))

#         self.decoder = nn.Sequential(*decoder_layers)
#         self.embedding_dim = latent_dim
#         self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=64)
#         #self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=256)
#         # self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=512)
    
#     def get_latent(self,input):
#         z,vel = self.encode(input)
#         z = F.normalize(z,dim=-1,p=2)
#         #z = self.normalize(z)
#         # z = z.reshape(z.shape[0], 1, 1, -1)
#         # z = z.permute(0, 2, 3, 1)
#         # quantize = self.quantizer(z)
#         # quantize = quantize.reshape(z.shape[0], -1)
#         return z,vel

#     def encode(self,input):
#         latent = self.encoder(input)
#         z = self.fc_mu(latent)
#         z = F.normalize(z,dim=-1,p=2)
#         #z = self.normalize(z)
#         vel = self.fc_vel(latent)
#         return z ,vel
    
#     def decode(self,quantized,z):
#         quantized = z + (quantized - z).detach()
#         input_hat = self.decoder(quantized)
#         return input_hat
    
#     def forward(self, input):
#         z,vel = self.encode(input)
#         z = z.reshape(z.shape[0], 1, 1, -1)
#         z = z.permute(0, 2, 3, 1)
#         quantize = self.quantizer(z)
#         quantize = quantize.reshape(z.shape[0], -1)
#         z = z.reshape(z.shape[0],-1)
#         input_hat = self.decode(quantize,z)
#         return input_hat,quantize,z,vel
    
#     def loss_fn(self,y, y_hat,quantized,z):
#         recon_loss = F.mse_loss(y_hat, y, reduction="none").sum(dim=-1)
        
#         commitment_loss = F.mse_loss(
#             quantized.detach(),
#             z,
#             reduction="sum",
#         )

#         embedding_loss = F.mse_loss(
#             quantized,
#             z.detach(),
#             reduction="sum",
#         )

#         vq_loss = 0.25*commitment_loss + embedding_loss

#         return (recon_loss + vq_loss).mean(dim=0)

# class VQVAE(nn.Module):

#     def __init__(self,
#                  in_dim= 45*5,
#                  latent_dim = 16,
#                  encoder_hidden_dims = [128,64],
#                  decoder_hidden_dims = [64,128],
#                  output_dim = 45) -> None:
        
#         super(VQVAE, self).__init__()

#         self.latent_dim = latent_dim
        
#         encoder_layers = []
#         encoder_layers.append(nn.Sequential(nn.Linear(in_dim, encoder_hidden_dims[0]),
#                                             nn.ELU()))
#         for l in range(len(encoder_hidden_dims)-1):
#             encoder_layers.append(nn.Sequential(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l+1]),
#                                         nn.ELU()))
#         self.encoder = nn.Sequential(*encoder_layers)

#         self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
#         self.fc_vel = nn.Linear(encoder_hidden_dims[-1], 3)
#         # self.normalize = nn.BatchNorm1d(latent_dim,affine=False)

#         # Build Decoder
#         decoder_layers = []
#         decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
#                                             nn.ELU()))
#         for l in range(len(decoder_hidden_dims)):
#             if l == len(decoder_hidden_dims) - 1:
#                 decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
#             else:
#                 decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
#                                         nn.ELU()))

#         self.decoder = nn.Sequential(*decoder_layers)
#         self.embedding_dim = latent_dim
#         self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=64)
#         #self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=256)
#         # self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=512)
    
#     def get_latent(self,input):
#         z,vel = self.encode(input)
#         z = F.normalize(z,dim=-1,p=2)
#         #z = self.normalize(z)
#         # z = z.reshape(z.shape[0], 1, 1, -1)
#         # z = z.permute(0, 2, 3, 1)
#         # quantize = self.quantizer(z)
#         # quantize = quantize.reshape(z.shape[0], -1)
#         return z,vel

#     def encode(self,input):
#         latent = self.encoder(input)
#         z = self.fc_mu(latent)
#         z = F.normalize(z,dim=-1,p=2)
#         #z = self.normalize(z)
#         vel = self.fc_vel(latent)
#         return z ,vel
    
#     def decode(self,quantized,z):
#         quantized = z + (quantized - z).detach()
#         input_hat = self.decoder(quantized)
#         return input_hat
    
#     def forward(self, input):
#         z,vel = self.encode(input)
#         z = z.reshape(z.shape[0], 1, 1, -1)
#         z = z.permute(0, 2, 3, 1)
#         quantize = self.quantizer(z)
#         quantize = quantize.reshape(z.shape[0], -1)
#         z = z.reshape(z.shape[0],-1)
#         input_hat = self.decode(quantize,z)
#         return input_hat,quantize,z,vel
    
#     def loss_fn(self,y, y_hat,quantized,z):
#         recon_loss = F.mse_loss(y_hat, y, reduction="none").sum(dim=-1)
        
#         commitment_loss = F.mse_loss(
#             quantized.detach(),
#             z,
#             reduction="sum",
#         )

#         embedding_loss = F.mse_loss(
#             quantized,
#             z.detach(),
#             reduction="sum",
#         )

#         vq_loss = 0.25*commitment_loss + embedding_loss

#         return (recon_loss + vq_loss).mean(dim=0)

# class VQVAE(nn.Module):

#     def __init__(self,
#                  in_dim= 45*5,
#                  latent_dim = 16,
#                  encoder_hidden_dims = [128,64],
#                  decoder_hidden_dims = [64,128],
#                  output_dim = 45) -> None:
        
#         super(VQVAE, self).__init__()

#         self.latent_dim = latent_dim
        
#         encoder_layers = []
#         encoder_layers.append(nn.Sequential(nn.Linear(in_dim, encoder_hidden_dims[0]),
#                                             nn.ELU()))
#         for l in range(len(encoder_hidden_dims)-1):
#             encoder_layers.append(nn.Sequential(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l+1]),
#                                         nn.ELU()))
#         self.encoder = nn.Sequential(*encoder_layers)

#         self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
#         # self.fc_vel = nn.Linear(encoder_hidden_dims[-1], 3)
#         # self.normalize = nn.BatchNorm1d(latent_dim,affine=False)

#         # Build Decoder
#         decoder_layers = []
#         decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
#                                             nn.ELU()))
#         for l in range(len(decoder_hidden_dims)):
#             if l == len(decoder_hidden_dims) - 1:
#                 decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
#             else:
#                 decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
#                                         nn.ELU()))

#         self.decoder = nn.Sequential(*decoder_layers)
#         self.embedding_dim = latent_dim
#         self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=64)
#         #self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=256)
#         # self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=512)
    
#     def get_latent(self,input):
#         z = self.encode(input)
#         z = F.normalize(z,dim=-1,p=2)
#         #z = self.normalize(z)
#         # z = z.reshape(z.shape[0], 1, 1, -1)
#         # z = z.permute(0, 2, 3, 1)
#         # quantize = self.quantizer(z)
#         # quantize = quantize.reshape(z.shape[0], -1)
#         return z

#     def encode(self,input):
#         latent = self.encoder(input)
#         z = self.fc_mu(latent)
#         z = F.normalize(z,dim=-1,p=2)
#         #z = self.normalize(z)
#         # vel = self.fc_vel(latent)
#         return z 
    
#     def decode(self,quantized,z):
#         quantized = z + (quantized - z).detach()
#         input_hat = self.decoder(quantized)
#         return input_hat
    
#     def forward(self, input):
#         z = self.encode(input)
#         z = z.reshape(z.shape[0], 1, 1, -1)
#         z = z.permute(0, 2, 3, 1)
#         quantize = self.quantizer(z)
#         quantize = quantize.reshape(z.shape[0], -1)
#         z = z.reshape(z.shape[0],-1)
#         input_hat = self.decode(quantize,z)
#         return input_hat,quantize,z
    
#     def loss_fn(self,y, y_hat,quantized,z):
#         recon_loss = F.mse_loss(y_hat, y, reduction="none").sum(dim=-1)
        
#         commitment_loss = F.mse_loss(
#             quantized.detach(),
#             z,
#             reduction="sum",
#         )

#         embedding_loss = F.mse_loss(
#             quantized,
#             z.detach(),
#             reduction="sum",
#         )

#         vq_loss = 0.25*commitment_loss + embedding_loss

#         return (recon_loss + vq_loss).mean(dim=0)
    
class VQVAE(nn.Module):

    def __init__(self,
                 in_dim= 45*5,
                 latent_dim = 16,
                 encoder_hidden_dims = [128,64],
                 decoder_hidden_dims = [64,128],
                 output_dim = 45) -> None:
        
        super(VQVAE, self).__init__()

        self.latent_dim = latent_dim
        
        encoder_layers = []
        encoder_layers.append(nn.Sequential(nn.Linear(in_dim, encoder_hidden_dims[0]),
                                            # nn.LayerNorm(encoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(encoder_hidden_dims)-1):
            encoder_layers.append(nn.Sequential(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l+1]),
                                        # nn.LayerNorm(encoder_hidden_dims[l+1]),   
                                        nn.ELU()))
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        # self.fc_vel = nn.Linear(encoder_hidden_dims[-1], 3)
        # self.normalize = nn.LayerNorm(latent_dim,elementwise_affine=False)

        # Build Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            # nn.LayerNorm(decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
                                    #   nn.LayerNorm(decoder_hidden_dims[l+1]),
                                        nn.ELU()))

        self.decoder = nn.Sequential(*decoder_layers)
        self.embedding_dim = latent_dim
        self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=64)
        # self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=256)
        # self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=512)
    
    def get_latent(self,input):
        z = self.encode(input)
        z = F.normalize(z,dim=-1,p=2)
        # z = self.normalize(z)
        # z = z.reshape(z.shape[0], 1, 1, -1)
        # z = z.permute(0, 2, 3, 1)
        # quantize = self.quantizer(z)
        # quantize = quantize.reshape(z.shape[0], -1)
        return z

    def encode(self,input):
        latent = self.encoder(input)
        z = self.fc_mu(latent)
        #z = F.normalize(z,dim=-1,p=2)
        # z = self.normalize(z)
        # vel = self.fc_vel(latent)
        return z 
    
    def decode(self,quantized,z):
        quantized = z + (quantized - z).detach()
        input_hat = self.decoder(quantized)
        return input_hat
    
    def forward(self, input):
        z = self.encode(input)
        quantize = self.quantizer(z)
        input_hat = self.decode(quantize,z)
        return input_hat,quantize,z
    
    def loss_fn(self,y, y_hat,quantized,z):
        recon_loss = F.mse_loss(y_hat, y)
        
        commitment_loss = F.mse_loss(
            quantized.detach(),
            z
        )

        embedding_loss = F.mse_loss(
            quantized,
            z.detach()
        )

        vq_loss = 0.25*commitment_loss + embedding_loss

        return recon_loss + vq_loss
    
class VQVAE_vel(nn.Module):

    def __init__(self,
                 in_dim= 45*5,
                 latent_dim = 16,
                 encoder_hidden_dims = [128,64],
                 decoder_hidden_dims = [64,128],
                 output_dim = 45) -> None:
        
        super(VQVAE_vel, self).__init__()

        self.latent_dim = latent_dim
        
        encoder_layers = []
        encoder_layers.append(nn.Sequential(nn.Linear(in_dim, encoder_hidden_dims[0]),
                                            nn.BatchNorm1d(encoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(encoder_hidden_dims)-1):
            encoder_layers.append(nn.Sequential(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l+1]),
                                        nn.BatchNorm1d(encoder_hidden_dims[l+1]),   
                                        nn.ELU()))
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.fc_vel = nn.Sequential(nn.Linear(encoder_hidden_dims[-1], 3))
        # self.normalize = nn.LayerNorm(latent_dim,elementwise_affine=False)

        # Build Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            nn.BatchNorm1d(decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
                                      nn.BatchNorm1d(decoder_hidden_dims[l+1]),
                                      nn.ELU()))

        self.decoder = nn.Sequential(*decoder_layers)
        self.embedding_dim = latent_dim
        self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=512)
        # self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=256)
        # self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=512)
    
    def get_latent(self,input):
        z,vel = self.encode(input)
        #z = F.normalize(z,dim=-1,p=2)
        # z = self.normalize(z)
        # z = z.reshape(z.shape[0], 1, 1, -1)
        # z = z.permute(0, 2, 3, 1)
        z = self.quantizer(z)
        #z = F.normalize(z,dim=-1,p=2)
        # quantize = quantize.reshape(z.shape[0], -1)
        return z,vel

    def encode(self,input):
        
        latent = self.encoder(input)
        z = self.fc_mu(latent)
        #z = F.normalize(z,dim=-1,p=2)
        # z = self.normalize(z)
        vel = self.fc_vel(latent)
        return z,vel 
    
    def decode(self,quantized,z):
        quantized = z + (quantized - z).detach()
        input_hat = self.decoder(quantized)
        return input_hat
    
    def forward(self, input):
        z,vel = self.encode(input)
        quantize = self.quantizer(z)
        #quantize = F.normalize(quantize,dim=-1,p=2)
        input_hat = self.decode(quantize,z)
        return input_hat,quantize,z,vel
    
    def loss_fn(self,y, y_hat,quantized,z):
        recon_loss = F.mse_loss(y_hat, y)
        
        commitment_loss = F.mse_loss(
            quantized.detach(),
            z
        )

        embedding_loss = F.mse_loss(
            quantized,
            z.detach()
        )

        vq_loss = 0.25*commitment_loss + embedding_loss

        return recon_loss + vq_loss
    
class VQVAE_vel_conv(nn.Module):

    def __init__(self,
                 in_dim= 45*5,
                 latent_dim = 16,
                 encoder_hidden_dims = [128,64],
                 decoder_hidden_dims = [64,128],
                 output_dim = 45) -> None:
        
        super(VQVAE_vel_conv, self).__init__()

        self.latent_dim = latent_dim
        
        self.merge_conv = nn.Sequential(nn.Conv1d(in_channels=45,out_channels=45,kernel_size=2,stride=2),
                                        nn.ELU())

        encoder_layers = []
        encoder_layers.append(nn.Sequential(nn.Linear(in_dim, encoder_hidden_dims[0]),
                                            nn.BatchNorm1d(encoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(encoder_hidden_dims)-1):
            encoder_layers.append(nn.Sequential(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l+1]),
                                        nn.BatchNorm1d(encoder_hidden_dims[l+1]),   
                                        nn.ELU()))
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        self.fc_vel = nn.Sequential(nn.Linear(encoder_hidden_dims[-1], 32),
                                    #nn.BatchNorm1d(32),   
                                    nn.ELU(),
                                    nn.Linear(32,3))
        # self.normalize = nn.LayerNorm(latent_dim,elementwise_affine=False)

        # Build Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            nn.BatchNorm1d(decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
                                      nn.BatchNorm1d(decoder_hidden_dims[l+1]),
                                      nn.ELU()))

        self.decoder = nn.Sequential(*decoder_layers)
        self.embedding_dim = latent_dim
        self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=512)
        # self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=256)
        # self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=512)
    
    def get_latent(self,input):
        z,vel = self.encode(input)
        #z = F.normalize(z,dim=-1,p=2)
        # z = self.normalize(z)
        # z = z.reshape(z.shape[0], 1, 1, -1)
        # z = z.permute(0, 2, 3, 1)
        z = self.quantizer(z)
        z = F.normalize(z,dim=-1,p=2)
        # quantize = quantize.reshape(z.shape[0], -1)
        return z,vel

    def encode(self,input):
        b = input.size()[0]
        input = self.merge_conv(input.permute((0,2,1)))
        input = input.view(b,-1)
        
        latent = self.encoder(input)
        z = self.fc_mu(latent)
        z = F.normalize(z,dim=-1,p=2)
        # z = self.normalize(z)
        vel = self.fc_vel(latent)
        return z,vel 
    
    def decode(self,quantized,z):
        quantized = z + (quantized - z).detach()
        input_hat = self.decoder(quantized)
        return input_hat
    
    def forward(self, input):
        z,vel = self.encode(input)
        quantize = self.quantizer(z)
        quantize = F.normalize(quantize,dim=-1,p=2)
        input_hat = self.decode(quantize,z)
        return input_hat,quantize,z,vel
    
    def loss_fn(self,y, y_hat,quantized,z):
        recon_loss = F.mse_loss(y_hat, y)
        
        commitment_loss = F.mse_loss(
            quantized.detach(),
            z
        )

        embedding_loss = F.mse_loss(
            quantized,
            z.detach()
        )

        vq_loss = 0.25*commitment_loss + embedding_loss

        return recon_loss + vq_loss
    
class VQVAE_RNN(nn.Module):

    def __init__(self,
                 in_dim= 45*5,
                 latent_dim = 16,
                 encoder_hidden_dims = [128,64],
                 decoder_hidden_dims = [64,128],
                 output_dim = 45) -> None:
        
        super(VQVAE_RNN, self).__init__()

        self.latent_dim = latent_dim
        
        self.encoder = RnnDoubleHeadEncoder(input_size=45,output_size=latent_dim,hidden_size=64)

        #self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        # self.fc_vel = nn.Linear(encoder_hidden_dims[-1], 3)
        # self.normalize = nn.LayerNorm(latent_dim,elementwise_affine=False)

        # Build Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            nn.LayerNorm(decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
                                      nn.LayerNorm(decoder_hidden_dims[l+1]),
                                        nn.ELU()))

        self.decoder = nn.Sequential(*decoder_layers)
        self.embedding_dim = latent_dim
        self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=64)
        # self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=256)
        # self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=512)
    
    def get_latent(self,input):
        z,vel= self.encode(input)
        #z = F.normalize(z,dim=-1,p=2)
        # z = self.normalize(z)
        # z = z.reshape(z.shape[0], 1, 1, -1)
        # z = z.permute(0, 2, 3, 1)
        # quantize = self.quantizer(z)
        # quantize = quantize.reshape(z.shape[0], -1)
        return z,vel

    def encode(self,input):
        z,vel = self.encoder(input)
        #z = F.normalize(z,dim=-1,p=2)
        # z = self.normalize(z)
        # vel = self.fc_vel(latent)
        return z,vel
    
    def decode(self,quantized,z):
        quantized = z + (quantized - z).detach()
        input_hat = self.decoder(quantized)
        return input_hat
    
    def forward(self, input):
        z,vel = self.encode(input)
        z = z.reshape(z.shape[0], 1, 1, -1)
        z = z.permute(0, 2, 3, 1)
        quantize = self.quantizer(z)
        quantize = quantize.reshape(z.shape[0], -1)
        z = z.reshape(z.shape[0],-1)
        input_hat = self.decode(quantize,z)
        return input_hat,quantize,z,vel
    
    def loss_fn(self,y, y_hat,quantized,z):
        recon_loss = F.mse_loss(y_hat, y, reduction="none").sum(dim=-1)
        
        commitment_loss = F.mse_loss(
            quantized.detach(),
            z,
            reduction="sum",
        )

        embedding_loss = F.mse_loss(
            quantized,
            z.detach(),
            reduction="sum",
        )

        vq_loss = 0.25*commitment_loss + embedding_loss

        return (recon_loss + vq_loss).mean(dim=0)
    

class CnnHistoryEncoder(nn.Module):
    def __init__(self, input_size, tsteps, output_size):
        # self.device = device
        super(CnnHistoryEncoder, self).__init__()
        self.tsteps = tsteps

        channel_size = 16
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), 
                nn.BatchNorm1d(3*channel_size),
                nn.ELU(),
                )

        self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2),
                nn.BatchNorm1d(2*channel_size),
                nn.ELU(),
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1),
                nn.BatchNorm1d(channel_size),
                nn.ELU(),
                nn.Flatten())

        self.linear_output = nn.Linear(channel_size * 3, output_size)
        self.vel_output = nn.Linear(channel_size * 3, 3)

    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        latent = self.linear_output(output)
        vel = self.vel_output(output.detach())
        return latent,vel
    
class VQVAE_CNN(nn.Module):

    def __init__(self,
                 in_dim= 45*5,
                 latent_dim = 16,
                 encoder_hidden_dims = [128,64],
                 decoder_hidden_dims = [64,128],
                 output_dim = 45) -> None:
        
        super(VQVAE_CNN, self).__init__()

        self.latent_dim = latent_dim
        
        self.encoder = CnnHistoryEncoder(45,10,latent_dim)

        # Build Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            nn.BatchNorm1d(decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
                                        nn.BatchNorm1d(decoder_hidden_dims[l+1]),
                                        nn.ELU()))

        self.decoder = nn.Sequential(*decoder_layers)
        self.embedding_dim = latent_dim
        #self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=64)
        # self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=256)
        self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=512)
    
    def get_latent(self,input):
        z,vel = self.encode(input)
        #z = F.normalize(z,dim=-1,p=2)
        # z = self.normalize(z)
        # z = z.reshape(z.shape[0], 1, 1, -1)
        # z = z.permute(0, 2, 3, 1)
        #z = self.quantizer(z)
        #z = F.normalize(z,dim=-1,p=2)
        # quantize = quantize.reshape(z.shape[0], -1)
        return z,vel

    def encode(self,input):
        
        z,vel = self.encoder(input)
        z = F.normalize(z,dim=-1,p=2)
        
        return z,vel 
    
    def decode(self,quantized,z):
        quantized = z + (quantized - z).detach()
        input_hat = self.decoder(quantized)
        return input_hat
    
    def forward(self, input):
        z,vel = self.encode(input)
        quantize = self.quantizer(z)
        quantize = F.normalize(quantize,dim=-1,p=2)
        input_hat = self.decode(quantize,z)
        return input_hat,quantize,z,vel
    
    def loss_fn(self,y, y_hat,quantized,z):
        recon_loss = F.mse_loss(y_hat, y)
        
        commitment_loss = F.mse_loss(
            quantized.detach(),
            z
        )

        embedding_loss = F.mse_loss(
            quantized,
            z.detach()
        )

        vq_loss = 0.25*commitment_loss + embedding_loss

        return recon_loss + vq_loss
    
    
class Config:
    def __init__(self):
        self.n_obs = 45
        self.block_size = 9
        self.n_action = 45+3
        self.n_layer: int = 4
        self.n_head: int = 4
        self.n_embd: int = 32
        self.dropout: float = 0.0
        self.bias: bool = True

class VQVAE_Trans(nn.Module):

    def __init__(self,
                 in_dim= 45*5,
                 latent_dim = 16,
                 encoder_hidden_dims = [128,64],
                 decoder_hidden_dims = [64,128],
                 output_dim = 45) -> None:
        
        super(VQVAE_Trans, self).__init__()

        self.latent_dim = latent_dim
        

        self.transformer_config = Config()
        self.transformer_config.n_layer = 2
        self.transformer_config.n_action = latent_dim
        self.transformer_config.n_obs = 45
        self.encoder = StateCausalTransformer(config=self.transformer_config)

        # self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)
        # self.fc_vel = nn.Linear(encoder_hidden_dims[-1], 3)
        # self.normalize = nn.LayerNorm(latent_dim,elementwise_affine=False)

        # Build Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            nn.LayerNorm(decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
                                      nn.LayerNorm(decoder_hidden_dims[l+1]),
                                        nn.ELU()))

        self.decoder = nn.Sequential(*decoder_layers)
        self.embedding_dim = latent_dim
        #self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=64)
        # self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=256)
        self.quantizer = Quantizer(embedding_dim=self.embedding_dim,num_embeddings=512)
    
    def get_latent(self,input):
        z = self.encode(input)
        #z = F.normalize(z,dim=-1,p=2)
        # z = self.normalize(z)
        # z = z.reshape(z.shape[0], 1, 1, -1)
        # z = z.permute(0, 2, 3, 1)
        # quantize = self.quantizer(z)
        # quantize = quantize.reshape(z.shape[0], -1)
        return z

    def encode(self,input):
        z = self.encoder(input)
        # z = self.fc_mu(latent)
        #z = F.normalize(z,dim=-1,p=2)
        # z = self.normalize(z)
        # vel = self.fc_vel(latent)
        return z 
    
    def decode(self,quantized,z):
        quantized = z + (quantized - z).detach()
        input_hat = self.decoder(quantized)
        return input_hat
    
    def forward(self, input):
        z = self.encode(input)
        z = z.reshape(z.shape[0], 1, 1, -1)
        z = z.permute(0, 2, 3, 1)
        quantize = self.quantizer(z)
        quantize = quantize.reshape(z.shape[0], -1)
        z = z.reshape(z.shape[0],-1)
        input_hat = self.decode(quantize,z)
        return input_hat,quantize,z
    
    def loss_fn(self,y, y_hat,quantized,z):
        recon_loss = F.mse_loss(y_hat, y, reduction="none").sum(dim=-1)
        
        commitment_loss = F.mse_loss(
            quantized.detach(),
            z,
            reduction="sum",
        )

        embedding_loss = F.mse_loss(
            quantized,
            z.detach(),
            reduction="sum",
        )

        vq_loss = 0.25*commitment_loss + embedding_loss

        return (recon_loss + vq_loss).mean(dim=0)
    

class VQVAE_EMA(nn.Module):

    def __init__(self,
                 in_dim= 45*5,
                 latent_dim = 16,
                 encoder_hidden_dims = [128,64],
                 decoder_hidden_dims = [64,128],
                 output_dim = 45) -> None:
        
        super(VQVAE_EMA, self).__init__()

        self.latent_dim = latent_dim
        
        encoder_layers = []
        encoder_layers.append(nn.Sequential(nn.Linear(in_dim, encoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(encoder_hidden_dims)-1):
            encoder_layers.append(nn.Sequential(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l+1]),  
                                        nn.ELU()))
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim)

        # Build Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            nn.ELU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
                                        nn.ELU()))

        self.decoder = nn.Sequential(*decoder_layers)
        self.embedding_dim = latent_dim
        #self.quantizer = QuantizerEMA(embedding_dim=self.embedding_dim,num_embeddings=32)
        self.quantizer = QuantizerEMA(embedding_dim=self.embedding_dim,num_embeddings=512)
        # self.quantizer = QuantizerEMA(embedding_dim=self.embedding_dim,num_embeddings=512)
    
    def get_latent(self,input):
        z = self.encode(input)
        z = F.normalize(z,dim=-1,p=2)
        #z = self.normalize(z)
        # z = z_.reshape(z_.shape[0], 1, 1, -1)
        # z = z.permute(0, 2, 3, 1)
        # quantize,_ = self.quantizer(z)
        # quantize = quantize.reshape(z.shape[0], -1)
        return z

    def encode(self,input):
        latent = self.encoder(input)
        z = self.fc_mu(latent)
        #z = F.normalize(z,dim=-1,p=2)
        #z = self.normalize(z)

        return z
    
    def decode(self,quantized,z):
        quantized = z + (quantized - z).detach()
        input_hat = self.decoder(quantized)
        return input_hat
    
    def forward(self, input):
        z = self.encode(input)
        quantize,onehot_encode = self.quantizer(z)
        input_hat = self.decode(quantize,z)
        return input_hat,quantize,z,onehot_encode
    
    def loss_fn(self,y, y_hat,quantized,z,onehot_encode):
        recon_loss = F.mse_loss(y_hat, y)
        
        commitment_loss = F.mse_loss(
            quantized.detach(),
            z
        )

        # embedding_loss = F.mse_loss(
        #     quantized,
        #     z.detach(),
        #     reduction="sum",
        # )
        self.quantizer.update_codebook(z,onehot_encode)

        vq_loss = 0.25*commitment_loss #+ embedding_loss

        return recon_loss + vq_loss
    
class MixedMlp(nn.Module):
    def __init__(
        self,
        input_size,
        latent_size,
        hidden_size,
        num_actions,
        num_experts,
    ):
        super().__init__()

        input_size = latent_size + input_size
        inter_size = hidden_size + latent_size
        output_size = num_actions

        self.mlp_layers = [
            (
                nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
                nn.Parameter(torch.empty(num_experts, output_size)),
                None,
            ),
        ]

        for index, (weight, bias, _) in enumerate(self.mlp_layers):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)

        # Gating network
        gate_hsize = 128
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, num_experts),
        )

        self.c_norm = nn.LayerNorm(input_size - latent_size)

    def forward(self, z, c):
        c = self.c_norm(c)

        coefficients = F.softmax(self.gate(torch.cat((z, c), dim=1)), dim=1)
        layer_out = c
        for (weight, bias, activation) in self.mlp_layers:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            )

            input = torch.cat((z, layer_out), dim=1).unsqueeze(1)
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
            out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
            layer_out = activation(out) if activation is not None else out

        return layer_out
    
class LipschitzLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), requires_grad=True))
        self.bias = torch.nn.Parameter(torch.empty((out_features), requires_grad=True))
        self.c = torch.nn.Parameter(torch.empty((1), requires_grad=True))
        self.softplus = torch.nn.Softplus()
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

        # compute lipschitz constant of initial weight to initialize self.c
        W = self.weight.data
        W_abs_row_sum = torch.abs(W).sum(1)
        self.c.data = W_abs_row_sum.max() # just a rough initialization

    def get_lipschitz_constant(self):
        return self.softplus(self.c)

    def forward(self, input):
        lipc = self.softplus(self.c)
        scale = lipc / torch.abs(self.weight).sum(1)
        scale = torch.clamp(scale, max=1.0)
        return torch.nn.functional.linear(input, self.weight * scale.unsqueeze(1), self.bias)

class lipmlp(torch.nn.Module):
    def __init__(self, dims):
        """
        dim[0]: input dim
        dim[1:-1]: hidden dims
        dim[-1]: out dim

        assume len(dims) >= 3
        """
        super().__init__()

        self.layers = torch.nn.ModuleList()
        for ii in range(len(dims)-2):
            self.layers.append(LipschitzLinear(dims[ii], dims[ii+1]))

        self.layer_output = LipschitzLinear(dims[-2], dims[-1])
        # self.relu = torch.nn.ReLU()
        self.relu = torch.nn.ELU()

    def get_lipschitz_loss(self):
        loss_lipc = 1.0
        for ii in range(len(self.layers)):
            loss_lipc = loss_lipc * self.layers[ii].get_lipschitz_constant()
        loss_lipc = loss_lipc *  self.layer_output.get_lipschitz_constant()
        return loss_lipc

    def forward(self, x):
        for ii in range(len(self.layers)):
            x = self.layers[ii](x)
            x = self.relu(x)
        return self.layer_output(x)
    
class MixedLipMlp(nn.Module):
    def __init__(
        self,
        input_size,
        latent_size,
        hidden_size,
        num_actions,
        num_experts,
    ):
        super().__init__()

        input_size = latent_size + input_size
        inter_size = hidden_size + latent_size
        output_size = num_actions

        self.mlp_layers = [
            (
                nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
                nn.Parameter(torch.empty(num_experts, output_size)),
                None,
            ),
        ]

        for index, (weight, bias, _) in enumerate(self.mlp_layers):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)

        # Gating network
        gate_hsize = 128
        self.gate = lipmlp([input_size,gate_hsize,gate_hsize,num_experts])

    def get_gate_lip_loss(self):
        return 5e-5*self.gate.get_lipschitz_loss()

    def forward(self, z, c):
        coefficients = F.softmax(self.gate(torch.cat((z, c), dim=1)), dim=1)
        layer_out = c
        for (weight, bias, activation) in self.mlp_layers:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            )

            input = torch.cat((z, layer_out), dim=1).unsqueeze(1)
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
            out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
            layer_out = activation(out) if activation is not None else out

        return layer_out
    
# class MixedLipschitzLinear(torch.nn.Module):
#     def __init__(self, in_features, out_features,num_experts):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = torch.nn.Parameter(torch.empty((num_experts,out_features, in_features), requires_grad=True))
#         self.bias = torch.nn.Parameter(torch.empty((num_experts,out_features), requires_grad=True))
#         self.c = torch.nn.Parameter(torch.empty((1), requires_grad=True))
#         self.softplus = torch.nn.Softplus()
#         self.initialize_parameters()

#     def initialize_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         self.bias.data.uniform_(-stdv, stdv)

#         # compute lipschitz constant of initial weight to initialize self.c
#         W = self.weight.data
#         W_abs_row_sum = torch.abs(W).sum(1)
#         self.c.data = W_abs_row_sum.max() # just a rough initialization

#     def get_lipschitz_constant(self):
#         return self.softplus(self.c)

#     def forward(self, input,coefficients):

#         flat_weight = self.weight.flatten(start_dim=1, end_dim=2)
#         mixed_weight = torch.matmul(coefficients, flat_weight).view(
#                 coefficients.shape[0], *self.weight.shape[1:3]
#             )
#         mixed_bias = torch.matmul(coefficients, self.bias).unsqueeze(1)
        
#         lipc = self.softplus(self.c)
#         scale = lipc / torch.abs(mixed_weight).sum(1)
#         scale = torch.clamp(scale, max=1.0)

#         return torch.nn.functional.linear(input, mixed_weight * scale.unsqueeze(1), mixed_bias)
    
# class lipMixedmlp(torch.nn.Module):
#     def __init__(self, dims):
#         """
#         dim[0]: input dim
#         dim[1:-1]: hidden dims
#         dim[-1]: out dim

#         assume len(dims) >= 3
#         """
#         super().__init__()

#         self.layers = torch.nn.ModuleList()
#         for ii in range(len(dims)-2):
#             self.layers.append(MixedLipschitzLinear(dims[ii], dims[ii+1]))

#         self.layer_output = LipschitzLinear(dims[-2], dims[-1])
#         self.elu = torch.nn.ELU()

#     def get_lipschitz_loss(self):
#         loss_lipc = 1.0
#         for ii in range(len(self.layers)):
#             loss_lipc = loss_lipc * self.layers[ii].get_lipschitz_constant()
#         loss_lipc = loss_lipc *  self.layer_output.get_lipschitz_constant()
#         return loss_lipc

#     def forward(self, x, coefficients):
#         for ii in range(len(self.layers)):
#             x = self.layers[ii](x,coefficients)
#             x = self.elu(x)
#         return self.layer_output(x)
    
# class MixedLipMlp(nn.Module):
#     def __init__(
#         self,
#         input_size,
#         latent_size,
#         hidden_size,
#         num_actions,
#         num_experts,
#     ):
#         super().__init__()

#         input_size = latent_size + input_size
#         inter_size = hidden_size + latent_size
#         output_size = num_actions

#         self.mlp_layers = torch.ModuleList()
#         self.mlp_layers = [
#             (
#                 nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
#                 F.elu,
#             ),
#             (
#                 nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
#                 F.elu,
#             ),
#             (
#                 nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
#                 None,
#             ),
#         ]

#         # Gating network
#         gate_hsize = 128
#         self.gate = lipmlp([input_size,gate_hsize,gate_hsize,num_experts])

#     def get_mlp_lipschitz_loss(self):
#         loss_lipc = 1.0
#         for ii in range(len(self.layers)):
#             loss_lipc = loss_lipc * self.layers[ii].get_lipschitz_constant()
#         loss_lipc = loss_lipc *  self.layer_output.get_lipschitz_constant()
#         return loss_lipc
    
#     def get_gate_lipschitz_loss(self):
#         return 

#     def forward(self, z, c):
#         coefficients = F.softmax(self.gate(torch.cat((z, c), dim=1)), dim=1)
#         layer_out = c
#         for (weight, bias, activation) in self.mlp_layers:
#             flat_weight = weight.flatten(start_dim=1, end_dim=2)
#             mixed_weight = torch.matmul(coefficients, flat_weight).view(
#                 coefficients.shape[0], *weight.shape[1:3]
#             )

#             input = torch.cat((z, layer_out), dim=1).unsqueeze(1)
#             mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
#             out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
#             layer_out = activation(out) if activation is not None else out

#         return layer_out

class MixedLayerNormMlp(nn.Module):
    def __init__(
        self,
        input_size,
        latent_size,
        hidden_size,
        num_actions,
        num_experts,
    ):
        super().__init__()

        input_size = latent_size + input_size
        inter_size = hidden_size + latent_size
        output_size = num_actions

        self.mlp_layers = [
            (
                nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
                nn.Parameter(torch.empty(num_experts, output_size)),
                None,
            ),
        ]

        for index, (weight, bias, _) in enumerate(self.mlp_layers):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)

        # Gating network
        gate_hsize = 128
        self.gate = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, gate_hsize),
            nn.LayerNorm(gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, gate_hsize),
            nn.LayerNorm(gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, num_experts),
        )

    def forward(self, z, c):
        coefficients = F.softmax(self.gate(torch.cat((z, c), dim=1)), dim=1)
        layer_out = c
        for (weight, bias, activation) in self.mlp_layers:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            )

            input = torch.cat((z, layer_out), dim=1).unsqueeze(1)
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
            out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
            layer_out = activation(out) if activation is not None else out

        return layer_out
    