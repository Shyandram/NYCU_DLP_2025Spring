import os
import torch
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve

from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform
        self.as_tensor = A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
                ToTensorV2()
            ]
        )

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        try:
            self.filenames = self._read_split()  # read train/valid/test splits
        except FileNotFoundError:
            self.download(self.root)
            self.filenames = self._read_split()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))
        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask)

        if self.transform is not None:
            sample = self.transform(**sample)      
            
        sample = self.as_tensor(**sample)
        sample['mask'] = sample['mask'].unsqueeze(0)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        if not os.path.exists(filepath):
            download_url(
                url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
                filepath=filepath,
            )
            extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        if not os.path.exists(filepath):
            download_url(
                url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
                filepath=filepath,
            )
            extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

def load_dataset(args, mode):
    if mode == "train":
        transform = A.Compose([
            A.Resize(256, 256),
            
            A.Flip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomResizedCrop(256, 256, scale=(0.8, 1.0)),
            A.ShiftScaleRotate(p=0.5),

            A.GaussianBlur(blur_limit=(3, 7), p=0.5),

            A.ElasticTransform(p=0.5),
        ])
    elif mode == "valid":
        transform = A.Compose([
            A.Resize(256, 256),
        ])
    elif mode == "test":
        transform = A.Compose([
            A.Resize(256, 256),
        ])
    # implement the load dataset function here
    dataset = OxfordPetDataset(args.data_path, mode = mode, transform = transform)
    shuffle = True if mode == "train" else False    
    loader = DataLoader(dataset, batch_size = args.batch_size, shuffle = shuffle)

    # assert False, "Not implemented yet!"
    return loader