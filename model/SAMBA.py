import torch
from torch import nn
import math
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from layers.mixer_seq_simple import MixerModel as Mamba
from layers.mixer_seq_simple import create_block
def positional_encoding(pe, learn_pe, q_len, d_model):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model)) # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d': W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d': W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d': W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos': W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else: raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual =False
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs x feature x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x feature x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x

class Model(nn.Module):
    """
    """

    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.patch_num = int((configs.seq_len - self.patch_len)/self.stride + 1)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.value_embedding = nn.Linear(configs.patch_len, configs.d_model, bias=False)
        self.e_layers=configs.e_layers
        self.d_layers=configs.d_layers
        self.d_model=configs.d_model
        # self.revin_layer = RevIN(configs.enc_in, affine=True, subtract_last=False)
        self.W_P = nn.Linear(self.patch_len, self.d_model)
        self.W_P_v = nn.Linear(self.patch_len, self.d_model)
        self.W_pos = positional_encoding('zero', True, self.patch_num, self.d_model)
        print(configs.enc_in)
        self.encoder_time  = nn.ModuleList(
            [
                create_block(
                    self.d_model,
                    d_state=configs.d_state1,
                    use_act=configs.use_act,
                    d_intermediate=configs.d_ff,
                    fused_add_norm=False,
                    layer_idx=i,
                )
                for i in range(configs.e_layers)
            ]
        )
        self.encoder_var= nn.ModuleList(
            [
                create_block(
                    self.d_model,
                    d_state=configs.d_state2,
                    use_act=configs.use_act,
                    d_intermediate=configs.d_ff,
                    fused_add_norm=False,
                    layer_idx=i,
                )
                for i in range(configs.d_layers)
            ]
        )
        # Prediction Head
        self.d_model=configs.d_model
        self.n_vars=configs.enc_in
        self.head_nf = configs.d_model * self.patch_num
        self.head= FlattenHead( self.n_vars, self.head_nf, configs.pred_len)
        self.ff = nn.Sequential(nn.Linear(configs.d_model*2, configs.d_model),
                                nn.GELU(),
                                nn.Dropout(configs.dropout),
                                nn.Linear(configs.d_model, configs.d_model))
        self.norm=nn.LayerNorm(configs.d_model)

        self.norm_v=nn.LayerNorm(configs.d_model)                        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # print(x_enc.shape)
        b, _, n_vars = x_enc.shape

        # Normalization from RevIN
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        # do patching and embedding
        x = x_enc.permute(0, 2, 1)
        x=self.padding_patch_layer(x)


        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        enc_in=self.W_P(x)

        enc_in_time=rearrange(enc_in,'b c l d -> (b c ) l d')#
        enc_in_time=enc_in_time+self.W_pos
        enc_in_var=rearrange(enc_in,'b c l d -> (b l) c d')#
        
        if self.e_layers!=0:
            hidden_states_time=enc_in_time
            residual_time=None
            for idx,layer in enumerate(self.encoder_time):
                hidden_states_time,residual_time=layer(hidden_states_time,residual_time)
                residual_time=self.norm(residual_time)
            enc_out_time=self.norm(hidden_states_time+residual_time)
            enc_out_time=rearrange(enc_out_time, '(b c) l d -> b c l d',c=n_vars,l=self.patch_num,d=self.d_model)
        if self.d_layers!=0:
            hidden_states_var=enc_in_var
            residual_var=None
            for idx,layer in enumerate(self.encoder_var):
                hidden_states_var,residual_var=layer(hidden_states_var,residual_var)
                residual_var=self.norm_v(residual_var)
            enc_out_var=self.norm_v(hidden_states_var+residual_var)
            enc_out_var=rearrange(enc_out_var, '(b l) c d-> b c l d',c=n_vars,l=self.patch_num,d=self.d_model)
       
        if self.e_layers==0:
            enc_out=enc_out_var
        elif self.d_layers==0:
            enc_out=enc_out_time
        else:
            enc_out=torch.cat((enc_out_time,enc_out_var),dim=-1)
            enc_out=self.ff(enc_out)
        enc_out=rearrange(enc_out, 'b c l d -> b c d l')
        dec_out = self.head(enc_out)
        dec_out=dec_out.permute(0,2,1)
        # De-Normalization from RevIN
        dec_out = dec_out * (stdev[:, [0], :].repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, [0], :].repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

