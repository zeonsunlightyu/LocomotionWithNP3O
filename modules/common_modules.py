import torch.nn as nn
import torch
from copy import deepcopy
from collections import defaultdict
from torch.nn import functional as F

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
    layers.append(activation)
    for l in range(len(hidden_dims)-1):
        layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
        layers.append(activation)
        layers.append(nn.LayerNorm(hidden_dims[l + 1]))

    if out_dims:
        layers.append(nn.Linear(hidden_dims[-1], out_dims))
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
    def __init__(self,activation_fn, input_size, encoder_dims,hidden_size,output_size,final_output_size):
        super(RnnBarlowTwinsStateHistoryEncoder,self).__init__()
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
        
        self.layer_norm = nn.LayerNorm(output_size)
        
        self.final_layer = nn.Linear(hidden_size,final_output_size)
        
    def forward(self,obs):
        obs = self.encoder(obs)
        out, h_n = self.rnn(obs)
        latent = self.final_layer(out[:,-1,:])
        return latent
    
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
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
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

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )

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

class BetaVAE(nn.Module):

    def __init__(self,
                 in_dim,
                 latent_dim = 19,
                 encoder_hidden_dims = [128,64],
                 decoder_hidden_dims = [64,128],
                 output_dim = 48,
                 beta: int = 0.1) -> None:
        
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        
        encoder_layers = []
        encoder_layers.append(nn.Sequential(nn.Linear(in_dim, encoder_hidden_dims[0]),
                                            nn.LayerNorm(encoder_hidden_dims[0]),
                                            nn.LeakyReLU()))
        for l in range(len(encoder_hidden_dims)-1):
            encoder_layers.append(nn.Sequential(nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l+1]),
                                        nn.LayerNorm(encoder_hidden_dims[l+1]),
                                        nn.LeakyReLU()))
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(encoder_hidden_dims[-1], latent_dim-3)
        self.fc_var = nn.Linear(encoder_hidden_dims[-1], latent_dim-3)
        self.fc_vel = nn.Linear(encoder_hidden_dims[-1], 3)


        # Build Decoder
        decoder_layers = []
        decoder_layers.append(nn.Sequential(nn.Linear(latent_dim, decoder_hidden_dims[0]),
                                            nn.LayerNorm(decoder_hidden_dims[0]),
                                            nn.LeakyReLU()))
        for l in range(len(decoder_hidden_dims)):
            if l == len(decoder_hidden_dims) - 1:
                decoder_layers.append(nn.Linear(decoder_hidden_dims[l],output_dim))
            else:
                decoder_layers.append(nn.Sequential(nn.Linear(decoder_hidden_dims[l], decoder_hidden_dims[l+1]),
                                        nn.LayerNorm(decoder_hidden_dims[l+1]),
                                        nn.LeakyReLU()))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, input):
       
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        vel = self.fc_vel(result)

        return [mu,log_var,vel]

    def decode(self, z,vel):
        result = torch.cat([z,vel],dim=-1)
        result = self.decoder(result)
        return result

    def reparameterize(self, mu, logvar):
       
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu,log_var,vel = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z,vel),z, mu, log_var, vel]
    
