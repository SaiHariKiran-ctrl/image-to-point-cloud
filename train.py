import numpy as np
import glob
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from model import ImageToPointCloud
from loss import ChamferLoss
from tqdm import tqdm
import time


class PointCloudDataset:
    def __init__(self, root_dir, transform=None, use_augmentation=True):
        self.root_dir = root_dir
        self.use_augmentation = use_augmentation
        self.transform = transform if transform is not None else self._default_transform()
        self.data_pairs = self._build_dataset()

    def _default_transform(self):
        transform_list = []
        
        transform_list.extend([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        if self.use_augmentation:
            augment_list = [
                transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0,
                    hue=0
                ),
                transforms.RandomPerspective(
                    distortion_scale=0.2,
                    p=0.3
                )
            ]
            transform_list = [transform_list[0]] + augment_list + transform_list[1:]

        return transforms.Compose(transform_list)

    def _build_dataset(self):
        data_pairs = []
        model_dirs = glob.glob(os.path.join(self.root_dir, "model_*"))

        for model_dir in model_dirs:
            pc_path = os.path.join(model_dir, "point_cloud.xyz")
            if not os.path.exists(pc_path):
                print(f"Warning: No point cloud file found in {model_dir}")
                continue
            
            image_dir = os.path.join(model_dir, "images")
            if not os.path.exists(image_dir):
                print(f"Warning: Missing images directory for {model_dir}")
                continue

            image_paths = glob.glob(os.path.join(image_dir, "view_*.jpg"))
            if not image_paths:
                print(f"Warning: No images found for {model_dir}")
                continue

            for img_path in image_paths:
                if os.path.exists(img_path):
                    data_pairs.append((img_path, pc_path))
                else:
                    print(f"Warning: Image file missing: {img_path}")

        if not data_pairs:
            raise RuntimeError(f"No valid data pairs found in {self.root_dir}")
            
        return data_pairs

    def _load_point_cloud(self, pc_path):
        try:
            points = np.loadtxt(pc_path, delimiter=' ')
            
            center = np.mean(points, axis=0)
            points = points - center
            scale = np.max(np.abs(points))
            points = points / scale if scale != 0 else points
            
            target_points = 5304

            if len(points) > target_points:
                points = points[:target_points]
            elif len(points) < target_points:
                points_needed = target_points - len(points)
                indices = np.arange(points_needed) % len(points)
                extra_points = points[indices]
                points = np.vstack([points, extra_points])
            
            return torch.FloatTensor(points)
        except Exception as e:
            print(f"Error loading point cloud {pc_path}: {str(e)}")
            raise

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        try:
            img_path, pc_path = self.data_pairs[idx]
            
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            point_cloud = self._load_point_cloud(pc_path)
            
            return {
                'image': image,
                'point_cloud': point_cloud,
                'image_path': img_path,
                'pc_path': pc_path
            }
        except Exception as e:
            print(f"Error loading item {idx}: {str(e)}")
            raise



def create_dataloaders(root_dir, batch_size=8, num_workers=0, use_augmentation=True): 
    train_dataset = PointCloudDataset(root_dir, use_augmentation=use_augmentation)
    val_dataset = PointCloudDataset(root_dir, use_augmentation=False)
    print(train_dataset[0])

    total_size = len(train_dataset)
    
    if total_size < 2 * batch_size:
        batch_size = max(1, total_size // 4)
        print(f"Dataset too small, reducing batch size to {batch_size}")
    
    train_size = int(0.8 * total_size)
    
    if train_size < batch_size:
        train_size = min(batch_size, total_size - batch_size)
    
    indices = list(range(total_size))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]  

    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False  
    )

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, num_epochs=100, device='cuda'):
    """
    Training loop with validation and debug information
    """
    if len(train_loader) == 0:
        raise ValueError("Training loader is empty!")
    if len(val_loader) == 0:
        raise ValueError("Validation loader is empty!")
        
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    model = model.to(device)
    criterion = ChamferLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        model.train()
        train_loss = 0.0
        train_batch_count = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] Training')
        for batch in train_pbar:
            if len(batch['image']) == 0:
                print("Warning: Empty batch encountered in training")
                continue
                
            images = batch['image'].to(device)
            point_clouds = batch['point_cloud'].to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, point_clouds)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batch_count += 1
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss = train_loss / train_batch_count if train_batch_count > 0 else float('inf')
        
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] Validation')
        with torch.no_grad():
            for batch in val_pbar:
                if len(batch['image']) == 0:
                    print("Warning: Empty batch encountered in validation")
                    continue
                    
                images = batch['image'].to(device)
                point_clouds = batch['point_cloud'].to(device)
                
                predictions = model(images)
                loss = criterion(predictions, point_clouds)
                val_loss += loss.item()
                val_batch_count += 1
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved! Best val loss: {best_val_loss:.4f}")
        
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        avg_time_per_epoch = total_time / (epoch + 1)
        estimated_time_remaining = avg_time_per_epoch * (num_epochs - epoch - 1)
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}] ({(epoch + 1) / num_epochs * 100:.1f}%)')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'Epoch Time: {epoch_time:.1f}s')
        print(f'Estimated Time Remaining: {estimated_time_remaining/60:.1f} minutes')
        print('-' * 50)


if __name__ == "__main__":
    root_dir = "D:/Desktop/3D/img-to-pointcloud/dataset/models"

    # model = ImageToPointCloud(num_points=5304)

    train_loader, val_loader = create_dataloaders(root_dir)
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # train_model(model, train_loader, val_loader, num_epochs=100, device=device)