import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse
import json
from typing import Tuple, Dict, Any
from PIL import Image
import os


class ECGDataset(Dataset):
    def __init__(self, ecg_signals, images=None, transform=None):
        self.ecg_signals = torch.FloatTensor(ecg_signals)
        self.images = images
        self.transform = transform
        
    def __len__(self):
        return len(self.ecg_signals)
    
    def __getitem__(self, idx):
        ecg = self.ecg_signals[idx]
        if self.images is not None:
            img = self.images[idx]
            if self.transform:
                img = self.transform(img)
            return ecg, img
        return ecg


class ECGEncoder(nn.Module):
    def __init__(self, ecg_length=1000, latent_dim=256):
        super(ECGEncoder, self).__init__()
        self.ecg_length = ecg_length
        self.latent_dim = latent_dim
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512 * (ecg_length // 16), latent_dim),
            nn.LeakyReLU(0.2),
        )
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ImageGenerator(nn.Module):
    def __init__(self, latent_dim=256, img_size=64, channels=3):
        super(ImageGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * (img_size // 8) * (img_size // 8)),
            nn.BatchNorm1d(512 * (img_size // 8) * (img_size // 8)),
            nn.ReLU(),
        )
        
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 512, self.img_size // 8, self.img_size // 8)
        x = self.deconv_layers(x)
        return x


class ImageDiscriminator(nn.Module):
    def __init__(self, img_size=64, channels=3):
        super(ImageDiscriminator, self).__init__()
        self.img_size = img_size
        self.channels = channels
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512 * (img_size // 16) * (img_size // 16), 1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Dream2Image(nn.Module):
    def __init__(self, ecg_length=1000, latent_dim=256, img_size=64, channels=3):
        super(Dream2Image, self).__init__()
        self.encoder = ECGEncoder(ecg_length, latent_dim)
        self.generator = ImageGenerator(latent_dim, img_size, channels)
        
    def forward(self, ecg_signal):
        z = self.encoder(ecg_signal)
        img = self.generator(z)
        return img


def train_epoch(model, discriminator, dataloader, optimizer_G, optimizer_D, criterion, device, epoch):
    model.train()
    discriminator.train()
    total_loss_G = 0
    total_loss_D = 0
    
    for batch_idx, (ecg, real_img) in enumerate(dataloader):
        ecg = ecg.to(device)
        real_img = real_img.to(device)
        batch_size = ecg.size(0)
        
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)
        
        optimizer_G.zero_grad()
        fake_img = model(ecg)
        pred_fake = discriminator(fake_img)
        loss_G = criterion(pred_fake, valid)
        loss_G.backward()
        optimizer_G.step()
        
        optimizer_D.zero_grad()
        pred_real = discriminator(real_img)
        loss_real = criterion(pred_real, valid)
        pred_fake_detached = discriminator(fake_img.detach())
        loss_fake = criterion(pred_fake_detached, fake)
        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()
        
        total_loss_G += loss_G.item()
        total_loss_D += loss_D.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, G Loss: {loss_G.item():.4f}, D Loss: {loss_D.item():.4f}')
    
    return total_loss_G / len(dataloader), total_loss_D / len(dataloader)


def train(ecg_data_path, image_data_path, model_save_path, epochs=100, batch_size=32, 
          lr_G=0.0002, lr_D=0.0002, ecg_length=1000, img_size=64, channels=3, 
          latent_dim=256, device='cuda'):
    
    ecg_data = np.load(ecg_data_path)
    image_data = np.load(image_data_path)
    
    if image_data.ndim == 4:
        image_data = (image_data.astype(np.float32) / 255.0) * 2.0 - 1.0
    else:
        image_data = np.random.rand(len(ecg_data), channels, img_size, img_size) * 2.0 - 1.0
    
    dataset = ECGDataset(ecg_data, image_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = Dream2Image(ecg_length, latent_dim, img_size, channels).to(device)
    discriminator = ImageDiscriminator(img_size, channels).to(device)
    
    optimizer_G = optim.Adam(model.parameters(), lr=lr_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        loss_G, loss_D = train_epoch(model, discriminator, dataloader, optimizer_G, 
                                     optimizer_D, criterion, device, epoch)
        print(f'Epoch {epoch+1}/{epochs}, Avg G Loss: {loss_G:.4f}, Avg D Loss: {loss_D:.4f}')
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'epoch': epoch,
            }, model_save_path)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'epoch': epochs,
    }, model_save_path)
    
    return model, discriminator


def predict(model, ecg_signal, device='cuda'):
    model.eval()
    with torch.no_grad():
        if isinstance(ecg_signal, np.ndarray):
            ecg_signal = torch.FloatTensor(ecg_signal)
        if ecg_signal.dim() == 1:
            ecg_signal = ecg_signal.unsqueeze(0)
        ecg_signal = ecg_signal.to(device)
        generated_image = model(ecg_signal)
        generated_image = (generated_image + 1) / 2.0
        generated_image = torch.clamp(generated_image, 0, 1)
        return generated_image.cpu().numpy()


def save_image(image_array, save_path):
    if image_array.ndim == 4:
        image_array = image_array[0]
    if image_array.shape[0] == 3:
        image_array = np.transpose(image_array, (1, 2, 0))
    image_array = (image_array * 255).astype(np.uint8)
    img = Image.fromarray(image_array)
    img.save(save_path)


def load_model(model_path, ecg_length=1000, latent_dim=256, img_size=64, channels=3, device='cuda'):
    model = Dream2Image(ecg_length, latent_dim, img_size, channels).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def cli_train(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    train(
        ecg_data_path=args.ecg_data,
        image_data_path=args.image_data,
        model_save_path=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_G=args.lr_g,
        lr_D=args.lr_d,
        ecg_length=args.ecg_length,
        img_size=args.img_size,
        channels=args.channels,
        latent_dim=args.latent_dim,
        device=device
    )


def cli_predict(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model = load_model(
        args.model,
        ecg_length=args.ecg_length,
        latent_dim=args.latent_dim,
        img_size=args.img_size,
        channels=args.channels,
        device=device
    )
    
    if args.ecg_file:
        ecg_signal = np.load(args.ecg_file)
    elif args.ecg_json:
        ecg_data = json.loads(args.ecg_json)
        ecg_signal = np.array(ecg_data['signal'])
    else:
        raise ValueError("Either --ecg-file or --ecg-json must be provided")
    
    generated_image = predict(model, ecg_signal, device)
    save_image(generated_image, args.output)
    print(f"Image saved to {args.output}")


def build_arg_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--ecg-data", required=True)
    train_parser.add_argument("--image-data", required=True)
    train_parser.add_argument("--model", default="dream2image_model.pth")
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr-g", type=float, default=0.0002)
    train_parser.add_argument("--lr-d", type=float, default=0.0002)
    train_parser.add_argument("--ecg-length", type=int, default=1000)
    train_parser.add_argument("--img-size", type=int, default=64)
    train_parser.add_argument("--channels", type=int, default=3)
    train_parser.add_argument("--latent-dim", type=int, default=256)
    train_parser.add_argument("--cpu", action="store_true")
    train_parser.set_defaults(func=cli_train)
    
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--ecg-file")
    predict_parser.add_argument("--ecg-json")
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--ecg-length", type=int, default=1000)
    predict_parser.add_argument("--img-size", type=int, default=64)
    predict_parser.add_argument("--channels", type=int, default=3)
    predict_parser.add_argument("--latent-dim", type=int, default=256)
    predict_parser.add_argument("--cpu", action="store_true")
    predict_parser.set_defaults(func=cli_predict)
    
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

