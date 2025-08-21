import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import kagglehub

# Kagglehub ile veri setini indir
path = kagglehub.dataset_download("akash2sharma/tiny-imagenet")
path = path + "/tiny-imagenet-200"

class HyperUNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=2):
        super(HyperUNet, self).__init__()
        
        # Encoder (Downsampling)
        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (Upsampling)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)  
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)   
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)  
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)    
        
        # Hyper-column connections
        self.hyper_conv1 = nn.Conv2d(64, 32, 1)
        self.hyper_conv2 = nn.Conv2d(128, 32, 1)
        self.hyper_conv3 = nn.Conv2d(256, 32, 1)
        self.hyper_conv4 = nn.Conv2d(512, 32, 1)
        
        # Final layer - hypercolumn features + final decoder features
        self.final_conv = nn.Conv2d(64 + 32*4, output_channels, 1)  # 64 + 128 = 192
        
        self.pool = nn.MaxPool2d(2, 2)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e1_pool = self.pool(e1)
        
        e2 = self.enc2(e1_pool)
        e2_pool = self.pool(e2)
        
        e3 = self.enc3(e2_pool)
        e3_pool = self.pool(e3)
        
        e4 = self.enc4(e3_pool)
        e4_pool = self.pool(e4)
        
        bottleneck = self.bottleneck(e4_pool)
        
        d4 = self.up4(bottleneck)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        h1 = self.hyper_conv1(e1)  
        h2 = F.interpolate(self.hyper_conv2(e2), size=(64, 64), mode='bilinear', align_corners=False)
        h3 = F.interpolate(self.hyper_conv3(e3), size=(64, 64), mode='bilinear', align_corners=False)
        h4 = F.interpolate(self.hyper_conv4(e4), size=(64, 64), mode='bilinear', align_corners=False)
        
        hyper_features = torch.cat([h1, h2, h3, h4], dim=1)
        
        final_features = torch.cat([d1, hyper_features], dim=1)
        
        output = self.final_conv(final_features)
        return torch.tanh(output)  # a,b channels için -1 ile 1 arası

class ColorizationDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        rgb_image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            rgb_image = self.transform(rgb_image)
        
        rgb_tensor = transforms.ToTensor()(rgb_image)
        
        rgb_np = rgb_tensor.permute(1, 2, 0).numpy()
        rgb_np = (rgb_np * 255).astype(np.uint8)
        
        lab_image = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
        lab_image = lab_image.astype(np.float32) / 255.0
        
        # L kanalı input (grayscale)
        L = lab_image[:, :, 0:1]  # Shape: (64, 64, 1)
        L = torch.from_numpy(L).permute(2, 0, 1)  # (1, 64, 64)
        
        # a,b kanalları target
        ab = lab_image[:, :, 1:3]  # Shape: (64, 64, 2)
        ab = torch.from_numpy(ab).permute(2, 0, 1)  # (2, 64, 64)
        ab = ab * 2.0 - 1.0  # [-1, 1] aralığına normalize et
        
        return L, ab

def get_image_paths(data_path):
    image_paths = []
    
    # Train klasöründeki resimleri al
    train_path = os.path.join(data_path, 'train')
    if os.path.exists(train_path):
        for class_folder in os.listdir(train_path):
            class_path = os.path.join(train_path, class_folder, 'images')
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.JPEG', '.jpg', '.png')):
                        image_paths.append(os.path.join(class_path, img_name))
    
    return image_paths

def lab_to_rgb(L, ab):
    L = L.squeeze(0).cpu().numpy()  
    ab = ab.cpu().numpy() 
    
    ab = (ab + 1.0) / 2.0
    
    lab = np.zeros((64, 64, 3))
    lab[:, :, 0] = L
    lab[:, :, 1] = ab[0]
    lab[:, :, 2] = ab[1]
    
    lab = (lab * 255).astype(np.uint8)
    
    # RGB'ye çevir
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return rgb

def train_model():
    # Device seçimi
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    image_paths = get_image_paths(path)
    print(f"Total images found: {len(image_paths)}")
    
    # Train/validation split
    train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
    ])
    
    # Dataset ve DataLoader
    # For Tiny-ImageNet 5000 train size
    # train_dataset = ColorizationDataset(train_paths[:5000], transform=transform)  # İlk 5000 resim
    # val_dataset = ColorizationDataset(val_paths[:1000], transform=val_transform)   # İlk 1000 resim

    # For Tiny-Image-Net and Full train
    train_dataset = ColorizationDataset(train_paths, transform=transform) 
    val_dataset = ColorizationDataset(val_paths, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Model
    model = HyperUNet(input_channels=1, output_channels=2).to(device)
    
    # Loss ve Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
    # Training loop
    num_epochs = 30
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for i, (L, ab) in enumerate(train_loader):
            L, ab = L.to(device), ab.to(device)
            
            optimizer.zero_grad()
            outputs = model(L)
            loss = criterion(outputs, ab)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if i % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for L, ab in val_loader:
                L, ab = L.to(device), ab.to(device)
                outputs = model(L)
                loss = criterion(outputs, ab)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        scheduler.step()
        
        # Her 5 epoch'ta bir örnek sonuçları kaydet
        if (epoch + 1) % 5 == 0:
            save_sample_results(model, val_loader, device, epoch+1)
    
    # Loss grafiği
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_curve.png')
    plt.show()
    
    # Modeli kaydet
    torch.save(model.state_dict(), 'hyperunet_colorization.pth')
    print("Model saved!")
    
    return model

def save_sample_results(model, val_loader, device, epoch):
    model.eval()
    with torch.no_grad():
        for i, (L, ab_true) in enumerate(val_loader):
            if i > 0:  # Sadece ilk batch
                break
                
            L, ab_true = L.to(device), ab_true.to(device)
            ab_pred = model(L)
            
            for j in range(min(4, L.size(0))):
                gray_img = L[j].cpu().numpy().squeeze()
                
                true_rgb = lab_to_rgb(L[j], ab_true[j])
                
                pred_rgb = lab_to_rgb(L[j], ab_pred[j])
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(gray_img, cmap='gray')
                axes[0].set_title('Grayscale Input')
                axes[0].axis('off')
                
                axes[1].imshow(true_rgb)
                axes[1].set_title('True Color')
                axes[1].axis('off')
                
                axes[2].imshow(pred_rgb)
                axes[2].set_title('Predicted Color')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(f'sample_epoch_{epoch}_img_{j}.png', dpi=150, bbox_inches='tight')
                plt.close()

def test_model(model_path, test_image_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = HyperUNet(input_channels=1, output_channels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    if test_image_path and os.path.exists(test_image_path):
        rgb_image = Image.open(test_image_path).convert('RGB')
        rgb_image = rgb_image.resize((64, 64))
        
        # LAB'a çevir
        rgb_np = np.array(rgb_image)
        lab_image = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2LAB)
        lab_image = lab_image.astype(np.float32) / 255.0
        
        L = lab_image[:, :, 0:1]
        L_tensor = torch.from_numpy(L).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            ab_pred = model(L_tensor)
        
        pred_rgb = lab_to_rgb(L_tensor.squeeze(0), ab_pred.squeeze(0))
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(L.squeeze(), cmap='gray')
        plt.title('Grayscale Input')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(rgb_np)
        plt.title('Original Color')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(pred_rgb)
        plt.title('Predicted Color')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('test_result.png', dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    model = train_model()
    test_model('hyperunet_colorization.pth', 'tiny-imagenet/tiny-imagenet-200/test/images/test_38.JPEG')
    
