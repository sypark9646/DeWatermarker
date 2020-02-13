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

import _pickle as cPickle

from torch.utils.data import Dataset


class DeWatermarkerDataset(Dataset):
    """
    DeWatermarker project dataset.
    """
    #참고 https://wingnim.tistory.com/33
    def __init__(self, root_dir, transform=None): #download, read data 등등을 하는 파트
        """
        Args:
            root_dir ()
            transform (callable): Optional ransform function to apply
                to samples.
        """
        self.root_dir = root_dir
        self.transform = transform

        watermarked_data = []
        original_data = []

        path_of_wimages = os.path.join(root_dir, '/watermarked')
        path_of_oimages = os.path.join(root_dir, '/original')

        list_of_images = os.listdir(path_of_wimages)
        for image in list_of_images:
            img = cv2.imread(os.path.join(path_of_wimages, image), 0)
            watermarked_data.append(img)
            
        list_of_images = os.listdir(path_of_oimages)
        for image in list_of_images:
            img = cv2.imread(os.path.join(path_of_oimages, image), 0)
            original_data.append(img)

        self.len = watermarked_data.shape[-1]
        self.watermarked_data=watermarked_data
        self.original_data=original_data

    def __len__(self): #data size를 넘겨주는 파트
        return self.len

    def __getitem__(self, index): #인덱스에 해당하는 아이템을 넘겨주는 파트.
        """
        Get a sample from the dataset.
        Args:
            index (int): The index of the dataset element we're accessing.
        """
        # Get the sample, and apply any necessary transform (if any).
        sample_watermark = self.watermarked_data[index]
        sample_original = self.original_data[index]

        if self.transform:
             sample_watermark = self.transform(sample_watermark)
             sample_original = self.transform(sample_original)

        return sample_watermark, sample_original
