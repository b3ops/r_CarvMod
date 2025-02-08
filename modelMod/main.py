from model import Generator, Discriminator
from datasets import WoodCarvingDataset, FutharkLetterDataset
from train import train_model
from torchvision import transforms

def main():
    # Define hyperparameters
    batch_size = 4
    num_epochs = 50

    # Define data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create datasets and data loaders
    wood_carving_dataset = WoodCarvingDataset('wood_carvings', transform)
    futhark_letter_dataset = FutharkLetterDataset('futhark_letters', transform)
    wood_carving_loader = torch.utils.data.DataLoader(wood_carving_dataset, batch_size=batch_size, shuffle=True)
    futhark_letter_loader = torch.utils.data.DataLoader(futhark_letter_dataset, batch_size=batch_size, shuffle=True)

    # Initialize models, optimizers, and criteria
    G_XtoY = Generator()
    G_YtoX = Generator