# MIT License
# 
# Copyright (c) 2019 Andrew Tallos
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ================================================================

import torch
from torch.utils.data import DataLoader
from data import DeWatermarkerDataset
from utils import display 
import torch.nn as nn


class BaseAutoencoder(nn.Module):
    """
    Provides some common functionality across the different autoencoder
    model architectures.
    """

    def forward(self, x):
        """
        Perform the forward pass on the given input.
        Args:
            x (Tensor): The input to perform the forward pass on.
        """
        x = self.resize(x)
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)

        return decoded_x

    def load(self):
        """
        Load in any existing weights belonging to this model.
        """
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=ETA,
            weight_decay=1e-5
        )
        loss = None
        try:
            checkpoint = torch.load(self.FPATH)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.load_state_dict(checkpoint)
                
            if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                
            if isinstance(checkpoint, dict) and 'loss' in checkpoint:    
                loss = checkpoint['loss']

            #self.load_state_dict(torch.load(self.FPATH))
            self.eval()
            return optimizer, loss
            
        except FileNotFoundError:
            msg = "No existing model to initialize from. Creating new one ..."
            print(msg)
            return None, None

    def save(self, optimizer, loss):
        """
        Save the current state of this model.
        """
        torch.save({
        'model_state_dict': self.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, self.FPATH)

    def resize(self, sample):
        """
        Resize a sample so that it can be inputted to a PyTorch
        Conv2D layer. We need to do this, as PyTorch our input
        is expected in the following shape:
            (batch_size, n_channels, height, width)
        """
        #print(sample.shape) # torch.Size([1, 2306, 1728, 3]) -> torch.Size([1, 2306, 1728])
        return sample.permute(0, 3, 1, 2).type("torch.FloatTensor")

class ARCH1Autoencoder(BaseAutoencoder):
    """
    Second autoencoder architecture. This will be a 2-layer convolutional
    autoencoder model.
    """
    KERNEL_SIZE = 3
    STRIDE = 1
    FPATH = "arch_1.pt"

    def __init__(self, inpt_shape):
        super().__init__()
        inpt_channels = 1
        
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=inpt_channels,
                out_channels=6,
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=6,
                out_channels=12,
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=12,
                out_channels=24,
                kernel_size=self.KERNEL_SIZE,
                stride=self.STRIDE
            ),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=24,
                out_channels=12,
                kernel_size=self.KERNEL_SIZE
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=12,
                out_channels=6,
                kernel_size=self.KERNEL_SIZE
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=6,
                out_channels=inpt_channels,
                kernel_size=self.KERNEL_SIZE
            ),
            nn.ReLU(inplace=True)
        )


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    
    # Hyperparameters.
    BATCH_SIZE = 1
    SHUFFLE = True
    NUM_WORKERS = 4
    N_EPOCHS = 2000
    ETA = 1e-3


    # Data setup.
    dataset = DeWatermarkerDataset(root_dir="data/training/set.pkl")
    INPT_SHAPE = dataset[0]["watermarked"].shape
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS
    )


    # Model setup. Note, we have the option to load in an existing model.
    model = ARCH1Autoencoder(inpt_shape=INPT_SHAPE)
    saved_optimizer, saved_loss = model.load()

    # Loss function & optimizer setup.
    criterion = torch.nn.MSELoss()
    optimizer = saved_optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=ETA,
            weight_decay=1e-5
        )
    

    # Train.
    # TODO: Make this a method on the Autoencoder class.
    min_loss = saved_loss
    if min_loss is None:
        min_loss = float("inf")

    for epoch in range(N_EPOCHS):
        for i_batch, sample_batched in enumerate(dataloader):
            watermarked, original = sample_batched
            original = model.resize(sample=original)
            # NOTE: This resizing operation is __inverting__ the colours.
            # It's fine given that our task is still to remove watermarks, but
            # we'll want to fix this going forwards.

            output = model(x=watermarked)
            loss = criterion(output, original)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # TODO: Move this into `debug()` method.
        if epoch == 0 or epoch % 100 == 0:
            print(">> epoch # {}: {}".format(epoch, loss.data))
            
            if loss < min_loss:
                print(">> updating weights.")
                model.save(optimizer=optimizer, loss=loss)
                min_loss = loss
