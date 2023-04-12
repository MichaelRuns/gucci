import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.utils import save_image
import os
import time

class Generator(nn.Module):
    def __init__(self, latent_dim, img_size, channels, features_g):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.channels = channels
        self.features_g = features_g
        self.latent_dim = latent_dim

        self.gen = nn.Sequential(
            self._block(latent_dim, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(features_g * 2, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.view(x.size(0), self.latent_dim, 1, 1)
        return self.gen(x)

class Discriminator(nn.Module):
  def __init__(self, img_size, channels, features_d):
    super(Discriminator, self).__init__()
    self.img_size = img_size
    self.channels = channels
    self.features_d = features_d
    self.disc = nn.Sequential(
        nn.Conv2d(channels, features_d, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        self._block(features_d, features_d*2, 4, 2,1),
        self._block(features_d*2, features_d*4, 4, 2, 1),
        self._block(features_d*4, features_d*8, 4, 2, 1),
        nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
        nn.Sigmoid(),
    )
  
  def _block(self, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )

  def forward(self, x):
      return self.disc(x)
  
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameters
    latent_size = 100
    hidden_size = 64
    image_size = 64
    num_epochs = 50
    batch_size = 64
    lr = 0.0002
    beta1 = 0.5
    num_workers = 2
    transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5])
    ])
    root_dir = 'RingFIR/data/RingFIR'
    dataset = ImageFolder(root_dir, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    generator = Generator(latent_dim=latent_size, img_size=image_size, channels=3, features_g=64)
    discriminator = Discriminator(image_size, channels=3, features_d=64)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_gen = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    criterion = nn.BCELoss().to(device)
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"[Epoch {epoch+1}/{num_epochs}]")
        for i, (images, _) in enumerate(dataloader):
            discriminator.zero_grad()
            generator.zero_grad()
            
            real_images = images.to(device)
            real_output = discriminator(real_images)
            real_labels = torch.ones_like(real_output).to(device)
            discriminator_loss_real = criterion(real_output, real_labels)
            discriminator_loss_real.backward()

            noise = torch.randn(batch_size, latent_size, 1, 1).to(device)
            fake_images = generator(noise)
            fake_output = discriminator(fake_images.detach())
            fake_labels = torch.zeros_like(fake_output).to(device)
            discriminator_loss_fake = criterion(fake_output, fake_labels)
            discriminator_loss_fake.backward()
            optimizer_disc.step()

            generator.zero_grad()
            fake_labels.fill_(1)
            fake_output = discriminator(fake_images)
            g_loss = criterion(fake_output, fake_labels)
            g_loss.backward()
            
            optimizer_gen.step()
            optimizer_disc.step()
            if i % 10 == 0:
                print(f"Batch [{i+1}/{len(dataloader)}] complete")
                elapsed_time = time.time() - start_time
                print(f"elapsed time: {int(elapsed_time//3600)}h {(elapsed_time%3600)//60}m {elapsed_time%60:.2f}s")
        epoch_execution_time = time.time() - epoch_start_time
        print(f"epoch#{epoch} execution time: {int(epoch_execution_time//3600)}h {(epoch_execution_time%3600)//60}m {epoch_execution_time%60:.2f}s")
    torch.save(generator.state_dict(), 'generator.pt')
    torch.save(discriminator.state_dict(), 'discriminator.pt')

if __name__ == '__main__':
    main()