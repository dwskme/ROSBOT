import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from tqdm import tqdm


# U-Net (simple version)
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.enc1 = CBR(3, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.dec2 = CBR(256 + 128, 128)
        self.dec1 = CBR(128 + 64, 64)

        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.final(d1)
        return torch.sigmoid(out)


# Dataset
class LaneDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.tf = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        return self.tf(img), self.tf(mask)


def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device}")

    image_paths = sorted(glob("images/*.png"))
    mask_paths = sorted(glob("masks/*.png"))

    dataset = LaneDataset(image_paths, mask_paths)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for epoch in range(15):
        model.train()
        total_loss = 0
        for img, mask in tqdm(loader):
            img, mask = img.to(device), mask.to(device)
            out = model(img)
            loss = loss_fn(out, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "lane_unet.pth")
    print("Model saved as lane_unet.pth")


if __name__ == "__main__":
    train()
