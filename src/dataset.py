
import pandas as pd
import numpy as np  
from torch.utils.data import Dataset
from torchvision import transforms
from src.config import TEST_CSV, TRAIN_CSV


class Dataset_Train(Dataset):

    def __init__(self, csv_path=TRAIN_CSV):

        self.df = pd.read_csv(csv_path)
        self.labels = self.df.iloc[:, 0].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        """      
        Args: 
            idx (int): Index of the image and label to get.
        
        Returns:
            (img, label): Transformed image as PyTorch Tensor, label as an int.
        """
        row = self.df.iloc[idx]
        label = row.iloc[0]
        data = row.iloc[1:].values.reshape(28, 28).astype(np.float32)
        data = transforms.ToTensor()(data)
        
        return data, label



class Dataset_Test(Dataset):

    def __init__(self, csv_path=TEST_CSV):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        data = (
            row.to_numpy(dtype=np.float32)
            .reshape(28, 28)
        )

        data = transforms.ToTensor()(data)

        return data


if __name__ == "__main__":
    dataset = Dataset_Train()

    print(dataset[0])