from torch.utils.data import Dataset
import torch

class ImageDataset(Dataset):
    def __init__(self, images, targets):
        super().__init__()
        self.images = images
        self.targets = targets
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):

        image = self.images[idx]
        target = self.targets[idx]

        return {
            "data": torch.tensor(image, dtype = torch.float32),
            "target": torch.tensor(target, dtype = torch.int32)
        }