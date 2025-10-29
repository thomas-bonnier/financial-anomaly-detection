import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, window_size: int, num_features: int, 
                 sector_dim: int, bottleneck_dim: int = 16,
                hidden_dims: list = None):
        super().__init__()
        self.window_size = window_size
        self.num_features = num_features
        self.sector_dim = sector_dim
        self.input_dim = window_size * num_features + sector_dim
        self.recon_dim = window_size * num_features

        if hidden_dims is None:
            hidden_dims = [256, 128]

        # Encoder: input = flattened(window*features) + sector vector
        enc_layers = []
        in_dim = self.input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU(inplace=True))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        in_dim = bottleneck_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU(inplace=True))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, self.recon_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        """
        Args:
            x: (batch, window_size, num_features)
            s: (batch, sector_dim)
        Returns: 
            out: (batch, window_size, num_features) -- reconstruction
        """
        batch = x.size(0)
        flat = x.view(batch, -1) # (batch, window*features)
        enc_in = torch.cat([flat, s], dim = 1) # (batch, input_dim)
        z = self.encoder(enc_in) # (batch, bottleneck_dim)
        out_flat = self.decoder(z) # (batch, recon_dim)
        out = out_flat.view(batch, self.window_size, self.num_features)
        return out



