"""
Taken from : https://github.com/JulesBelveze/time-series-autoencoder.git
"""
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf

def init_hidden(x: torch.Tensor, hidden_size: int, num_dir: int = 1):
    return nn.init.xavier_normal_(torch.zeros(num_dir, x.size(0), hidden_size))


###########################################################################
################################ ENCODERS #################################
###### #####################################################################

class Encoder(nn.Module):
    def __init__(self, config, input_size: int):
        """
        Initialize the model.

        Args:
            config:
            input_size: (int): size of the input
        """
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=config['n_features'],
                            hidden_size=config['h_size_encoder'],
                            num_layers= 2,
                            batch_first=True)

        self.linear = nn.Linear(config['h_size_encoder'], config['n_features'])

    def forward(self, input_data: torch.Tensor):
        """
        Run forward computation.

        Args:
            input_data: (torch.Tensor): tensor of input daa
        """
        lstm_out, _ = self.lstm(input_data)
        y_pred = self.linear(lstm_out[:, -1])
        return y_pred, None


class AttnEncoder(nn.Module):
    def __init__(self, config, input_size: int):
        """
        Initialize the network.

        Args:
            config:
            input_size: (int): size of the input
        """
        super(AttnEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = config['h_size_encoder']
        self.seq_len = config['seq_len']
        self.add_noise = config['denoising']
        self.directions = 1
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1
        )
        self.attn = nn.Linear(
            in_features=2 * self.hidden_size + self.seq_len,
            out_features=1
        )
        self.softmax = nn.Softmax(dim=1)

    def _get_noise(self, input_data: torch.Tensor, sigma=0.01, p=0.1):
        """
        Get noise.

        Args:
            input_data: (torch.Tensor): tensor of input data
            sigma: (float): variance of the generated noise
            p: (float): probability to add noise
        """
        normal = sigma * torch.randn(input_data.shape)
        mask = np.random.uniform(size=(input_data.shape))
        mask = (mask < p).astype(int)
        noise = normal * torch.tensor(mask)
        return noise

    def forward(self, input_data: torch.Tensor):
        """
        Forward computation.

        Args:
            input_data: (torch.Tensor): tensor of input data
        """
        h_t, c_t = (init_hidden(input_data, self.hidden_size, num_dir=self.directions),
                    init_hidden(input_data, self.hidden_size, num_dir=self.directions))

        attentions, input_encoded = (Variable(torch.zeros(input_data.size(0), self.seq_len, self.input_size)),
                                     Variable(torch.zeros(input_data.size(0), self.seq_len, self.hidden_size)))

        if self.add_noise and self.training:
            input_data += self._get_noise(input_data)

        for t in range(self.seq_len):
            x = torch.cat((h_t.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           c_t.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim=2)  # bs * input_size * (2 * hidden_dim + seq_len)

            e_t = self.attn(x.view(-1, self.hidden_size * 2 + self.seq_len))  # (bs * input_size) * 1
            a_t = self.softmax(e_t.view(-1, self.input_size))  # (bs, input_size)

            weighted_input = torch.mul(a_t, input_data[:, t, :])  # (bs * input_size)
            self.lstm.flatten_parameters()
            _, (h_t, c_t) = self.lstm(weighted_input.unsqueeze(0), (h_t, c_t))

            input_encoded[:, t, :] = h_t
            attentions[:, t, :] = a_t

        return attentions, input_encoded


###########################################################################
################################ DECODERS #################################
###########################################################################

class Decoder(nn.Module):
    def __init__(self, config):
        """
        Initialize the network.

        Args:
            config:
        """
        super(Decoder, self).__init__()
        self.seq_len = config['seq_len']
        self.lstm = nn.LSTM(input_size=config['n_features'],
                            hidden_size=config['h_size_decoder'],
                            num_layers=2,
                            batch_first=True)
        self.fc = nn.Linear(config['h_size_decoder'], config['output_size'])

    def forward(self, _,x: torch.Tensor):
        """
        Forward pass

        Args:
            _:
            y_hist: (torch.Tensor): shifted target
        """

        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)


class AttnDecoder(nn.Module):
    def __init__(self, config):
        """
        Initialize the network.

        Args:
            config:
        """
        super(AttnDecoder, self).__init__()
        self.seq_len = config['seq_len']
        self.encoder_hidden_size = config['h_size_encoder']
        self.decoder_hidden_size = config['h_size_decoder']
        self.out_feats = config['n_features']

        self.attn = nn.Sequential(
            nn.Linear(2 * self.decoder_hidden_size + self.encoder_hidden_size, self.encoder_hidden_size),
            nn.Tanh(),
            nn.Linear(self.encoder_hidden_size, 1)
        )
        self.lstm = nn.LSTM(input_size=self.out_feats, hidden_size=self.decoder_hidden_size)
        self.fc = nn.Linear(self.encoder_hidden_size + self.out_feats, self.out_feats)
        self.fc_out = nn.Linear(self.decoder_hidden_size + self.encoder_hidden_size, self.out_feats)
        self.fc.weight.data.normal_()

    def forward(self, input_encoded: torch.Tensor, y_history: torch.Tensor):
        """
        Perform forward computation.

        Args:
            input_encoded: (torch.Tensor): tensor of encoded input
            y_history: (torch.Tensor): shifted target
        """
        h_t, c_t = (
            init_hidden(input_encoded, self.decoder_hidden_size), init_hidden(input_encoded, self.decoder_hidden_size))
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))

        for t in range(self.seq_len):
            x = torch.cat((h_t.repeat(self.seq_len, 1, 1).permute(1, 0, 2),
                           c_t.repeat(self.seq_len, 1, 1).permute(1, 0, 2),
                           input_encoded), dim=2)

            x = tf.softmax(
                self.attn(
                    x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                ).view(-1, self.seq_len),
                dim=1)

            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]  # (batch_size, encoder_hidden_size)

            y_tilde = self.fc(torch.cat((context, y_history[:, t]), dim=1))  # (batch_size, out_size)

            self.lstm.flatten_parameters()
            _, (h_t, c_t) = self.lstm(y_tilde.unsqueeze(0), (h_t, c_t))

        return self.fc_out(torch.cat((h_t[0], context), dim=1))  # predicting value at t=self.seq_length+1