import torch
import torch.nn as nn

class MNISTAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 64, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 3, stride=2),  # b, 16, 5, 5
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.BatchNorm2d(1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class FourierAutoencoder(nn.Module):
    def __init__(self):
        super(FourierAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DualEncoderAutoencoder(nn.Module):
    def __init__(self):
        super(DualEncoderAutoencoder, self).__init__()
        
        # First Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),  # [B, 3, H, W] -> [B, 16, H/2, W/2]
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), # [B, 16, H/2, W/2] -> [B, 32, H/4, W/4]
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # [B, 32, H/4, W/4] -> [B, 64, H/8, W/8]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # [B, 64, H/8, W/8] -> [B, 128, H/16, W/16]
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        # Second Encoder
        self.encoder2 = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride=2, padding=1),  # Same structure as encoder1
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        # Decoder
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # Input channels are doubled after concatenation
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
        #     nn.Tanh()
        # )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # [B, 256, 2, 2] -> [B, 128, 4, 4]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # [B, 128, 4, 4] -> [B, 64, 8, 8]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # [B, 64, 8, 8] -> [B, 32, 16, 16]
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),    # [B, 32, 16, 16] -> [B, 16, 32, 32]
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=1, padding=1),     # [B, 16, 32, 32] -> [B, 3, 32, 32]
            nn.Tanh()
        )

    def forward(self, x1, x2):
        encoded1 = self.encoder1(x1)
        encoded2 = self.encoder2(x2)
        
        encoded = torch.cat((encoded1, encoded2), dim=1)
        decoded = self.decoder(encoded)
        return decoded


class MNISTConditionalAutoencoder(nn.Module):
    def __init__(self, n_classes=10, input_dim=28):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.linear_c = nn.Linear(n_classes, 1 * input_dim * input_dim)
        self.n_classes = n_classes
        self.input_dim = input_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1+1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 64, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 3, stride=2),  # b, 16, 5, 5
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.BatchNorm2d(1),
            nn.Tanh()
        )

    def forward(self, x, c):
        c = self.linear_c(self.label_emb(c)).view(-1, 1, self.input_dim, self.input_dim)
        x = torch.cat([x, c], dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x    
    
class ConditionalAutoencoder(nn.Module):
    def __init__(self, n_classes, input_dim):
        super().__init__()
        
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.linear_c = nn.Linear(n_classes, 1 * input_dim * input_dim)
        self.n_classes = n_classes
        self.input_dim = input_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3+1, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x, c):
        c = self.linear_c(self.label_emb(c)).view(-1, 1, self.input_dim, self.input_dim)
        x = torch.cat([x, c], dim=1)
        
        x = self.encoder(x)
        x = self.decoder(x)
        return x
