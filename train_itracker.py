# train_itracker.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from ITrackerModel import ITrackerModel
import csv

# === Custom Dataset ===
class GazeDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.samples = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                self.samples.append(row)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        face_img = Image.open(sample[1]).convert('RGB')
        left_eye_img = Image.open(sample[2]).convert('RGB')
        right_eye_img = Image.open(sample[3]).convert('RGB')
        face_grid = np.load(sample[4])
        gaze = torch.tensor([float(sample[5]), float(sample[6])], dtype=torch.float32)

        if self.transform:
            face_img = self.transform(face_img)
            left_eye_img = self.transform(left_eye_img)
            right_eye_img = self.transform(right_eye_img)

        face_grid = torch.tensor(face_grid, dtype=torch.float32).unsqueeze(0)

        return face_img, left_eye_img, right_eye_img, face_grid, gaze


# === Training Loop ===
def train_model(model, dataloader, device, epochs=10):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        total_loss = 0.0
        for faces, eyesL, eyesR, grids, gazes in dataloader:
            faces = faces.to(device)
            eyesL = eyesL.to(device)
            eyesR = eyesR.to(device)
            grids = grids.to(device)
            gazes = gazes.to(device)

            outputs = model(faces, eyesL, eyesR, grids)
            loss = criterion(outputs, gazes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")


# === Main ===
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = GazeDataset("dataset/metadata.csv", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = ITrackerModel().to(device)

    train_model(model, dataloader, device, epochs=10)

    torch.save(model.state_dict(), "itracker_trained.pth")
    print("\n💾 Model saved as itracker_trained.pth")


if __name__ == "__main__":
    main()
