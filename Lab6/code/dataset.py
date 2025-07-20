import os
from PIL import Image
import json

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class iclevr(Dataset):
    def __init__(self, root='iclevr', image_size=64, mode="train"):
        super().__init__()
        self.root = root   
        self.mode = mode
        self.setup_training_data(mode)

        with open('objects.json', 'r') as json_file:
            self.objects_dict = json.load(json_file)
        self.labels_one_hot = torch.stack([
            torch.sum(
                F.one_hot(
                    torch.tensor([self.objects_dict[obj] for obj in label]), 
                    num_classes=len(self.objects_dict)), 

                dim=0 ) for label in self.labels
        ])
            
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def setup_training_data(self, mode):
        assert mode in ['train', 'test', 'new_test']
        with open(f'{mode}.json', 'r') as json_file:
            self.json_data = json.load(json_file)
            if mode == 'train':
                self.img_paths, self.labels = list(self.json_data.keys()), list(self.json_data.values())
            elif mode in ["test", 'new_test']:
                self.labels = self.json_data
            
    def __len__(self):
        return len(self.labels)      
    
    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = os.path.join(self.root, self.img_paths[index])
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            label_one_hot = self.labels_one_hot[index]
            return img, label_one_hot
        
        elif self.mode in ['test', 'new_test']:
            label_one_hot = self.labels_one_hot[index]
            return label_one_hot

if __name__ == '__main__':
    dataset = iclevr(root='iclevr', mode='train')
    print(len(dataset))
    x, y = dataset[0]
    print(x.shape, y.shape)
    dataset = iclevr(root='iclevr', mode='test')
    print(len(dataset))
    y = dataset[0]
    print(y.shape)