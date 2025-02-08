import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class WoodCarvingDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        try:
            image = Image.open(img_name)
            image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_name}: {str(e)}")
            return None

class FutharkLetterDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])
        try:
            image = Image.open(img_name)
            image = image.convert('RGB')
            image = self.transform(image)
            futhark_letter = self.images[idx].split('-')[0]
            return image, futhark_letter
        except Exception as e:
            print(f"Error loading image {img_name}: {str(e)}")
            return None, None

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

wood_carving_dataset = WoodCarvingDataset('wood_carvings', transform)
futhark_letter_dataset = FutharkLetterDataset('futhark_letters', transform)

# Create your DataLoaders
wood_carving_loader = torch.utils.data.DataLoader(wood_carving_dataset, batch_size=4, shuffle=True)
futhark_letter_loader = torch.utils.data.DataLoader(futhark_letter_dataset, batch_size=4, shuffle=True)

