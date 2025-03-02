import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from model import ImageToPointCloud

def load_trained_model(model_path, device='cuda'):
    """
    Load a trained ImageToPointCloud model from a .pth file
    """
    # Initialize the model with the same parameters used during training
    model = ImageToPointCloud(num_points=5304)
    
    # Load the state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Move model to appropriate device
    model = model.to(device)
    
    # Set to evaluation mode
    model.eval()
    
    return model

def process_image(image_path, model, device='cuda'):
    """
    Process a single image through the model
    """
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    # Get prediction
    with torch.no_grad():
        point_cloud = model(image)
    
    # Convert to numpy array
    point_cloud = point_cloud.squeeze(0).cpu().numpy()
    
    return point_cloud

def save_point_cloud(point_cloud, output_path):
    """
    Save the generated point cloud to a file
    """
    # Save as .xyz file
    np.savetxt(output_path, point_cloud, delimiter=' ')

# Example usage
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model_path = 'D:/Desktop/3D/img-to-pointcloud/best_model.pth'
    model = load_trained_model(model_path, device)
    
    # Process an image
    image_path = 'D:/Desktop/3D/img-to-pointcloud/dataset/models/model_1/images/view_1.jpg'
    point_cloud = process_image(image_path, model, device)
    
    # Save the result
    output_path = 'output_point_cloud.xyz'
    save_point_cloud(point_cloud, output_path)
    
    print(f"Point cloud generated and saved to {output_path}")
    print(f"Point cloud shape: {point_cloud.shape}")