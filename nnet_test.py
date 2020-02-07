import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import DeWatermarkerDataset
from utils import display 
import cv2
import numpy as np

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
        optimizer = None
        loss = None
        try:
            checkpoint = torch.load(self.FPATH)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.load_state_dict(checkpoint)
                
          
            #self.load_state_dict(torch.load(self.FPATH))
            self.eval()
            return optimizer, loss
            
        except FileNotFoundError:
            msg = "No existing model to initialize from. Creating new one ..."
            print(msg)

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
        _, _, inpt_channels = inpt_shape
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
        self.decoder = nn.Sequaential(
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
    BATCH_SIZE = 4
    SHUFFLE = True
    NUM_WORKERS = 4
    N_EPOCHS = 2000
    N_BATCHES = 10
    ETA = 1e-3
    
    # Data setup.
    dataset = DeWatermarkerDataset(root_dir="./tests/set.pkl")
    INPT_SHAPE = dataset[0]["watermarked"].shape
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        num_workers=NUM_WORKERS
    )
    
    # Model setup. Note, we have the option to load in an existing model.
    model = ARCH1Autoencoder(inpt_shape=INPT_SHAPE)
    model.load()
    
    
    for i_batch, sample_batched in enumerate(dataloader):
        watermarked = sample_batched["watermarked"]
        output = model(x=watermarked).detach().numpy()
        output_pic = output.reshape(2306, 1728, 3).astype(int) #tiff float형으로 저장이 안돼서, uint8로 하면 노이즈 안없어짐
        
        cv2.imwrite('C:/Users/soyeon/Desktop/DeWatermarker-master/output.tiff', output_pic)
        
        cpoint1 = int(1728/3)
        cpoint2 = int(1728*2/3)
        rpoint1 = int(2306/3)
        rpoint2 = int(2306*2/3)
        
        img = list()
        
        img.append(output_pic[:rpoint1, :cpoint1, :])
        img.append(output_pic[:rpoint1, cpoint1:cpoint2, :])
        img.append(output_pic[:rpoint1, cpoint2:, :])
        
        img.append(output_pic[rpoint1:rpoint2, :cpoint1, :])
        img.append(output_pic[rpoint1:rpoint2, cpoint1:cpoint2, :])
        img.append(output_pic[rpoint1:rpoint2, cpoint2:, :])
        
        img.append(output_pic[rpoint2:, :cpoint1, :])
        img.append(output_pic[rpoint2:, cpoint1:cpoint2, :])
        img.append(output_pic[rpoint2:, cpoint2:, :])
        
        for i in range(9):
            cv2.imwrite('C:/Users/soyeon/Desktop/DeWatermarker-master/output'+str(i)+'.tiff', img[i])
        
        break
        
    print('Finished!')
