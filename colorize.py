import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from skimage.color import rgb2lab, lab2rgb
from skimage import io
import numpy as np
import glob

# Custom dataset to convert images to LAB and separate channels
class ColorizationDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = io.imread(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)

        # Rearrange to (height, width, channels) for rgb2lab
        image = image.permute(1, 2, 0).numpy()

        # Convert image to LAB and split channels
        lab_image = rgb2lab(image).astype("float32")
        L = lab_image[:, :, 0] / 50.0 - 1  # normalize to [-1, 1]
        AB = lab_image[:, :, 1:] / 128.0  # normalize to [-1, 1]

        return torch.from_numpy(L).unsqueeze(0), torch.from_numpy(AB).permute(2, 0, 1)

# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
            nn.Tanh()  # Output AB in range [-1, 1]
        )

    def forward(self, x):
        return self.model(x)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, kernel_size=4)  # Output a single value for each batch
        )

    def forward(self, x):
        # Flatten the output to match the shape of real_labels
        return self.model(x).view(-1)

# Training loop

def train_gan(dataloader, generator, discriminator, num_epochs=100, lr=0.0002):
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for i, (L, AB_real) in enumerate(dataloader):
            # Move data to device (GPU if available)
            L, AB_real = L.cuda(), AB_real.cuda()
            batch_size = L.size(0)

            # Create labels
            real_labels = torch.ones(batch_size).cuda().unsqueeze(1).unsqueeze(1).unsqueeze(1)  # Reshape target

            # Train Discriminator
            optimizer_d.zero_grad()
            real_input = torch.cat((L, AB_real), dim=1)
            real_output = discriminator(real_input)

            d_loss_real = criterion(real_output, real_labels)

            AB_fake = generator(L)
            fake_input = torch.cat((L, AB_fake), dim=1)
            fake_output = discriminator(fake_input.detach())

            d_loss_fake = criterion(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            fake_output = discriminator(fake_input)
            g_loss = criterion(fake_output, real_labels)  # Use reshaped target here too
            g_loss.backward()
            optimizer_g.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
                
# Postprocessing function to convert LAB to RGB
def lab_to_rgb(L, AB):
    L = (L + 1) * 50.0  # De-normalize L to [0, 100]
    AB = AB * 128.0  # De-normalize AB to [-128, 128]
    lab = torch.cat([L, AB], dim=0).permute(1, 2, 0).cpu().detach().numpy()
    rgb = lab2rgb(lab)
    return rgb

# Get all jpg and png images from a specific folder
image_folder = "c:/Users/ahire/Desktop/Mini Project 3/Project/images/"
image_paths = glob.glob(f"{image_folder}/*.jpg") + glob.glob(f"{image_folder}/*.png") + glob.glob(f"{image_folder}/*.JPEG")

# Load dataset
transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.ToTensor()])
dataset = ColorizationDataset(image_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Train GAN
train_gan(dataloader, generator, discriminator, num_epochs=100)

# Generate colorized image
L, _ = dataset[0]  # Example L channel
L = L.unsqueeze(0).to(device)
with torch.no_grad():
    AB_fake = generator(L)
colorized_image = lab_to_rgb(L[0], AB_fake[0])