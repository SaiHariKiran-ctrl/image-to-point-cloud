import torch
import torch.nn as nn
import numpy as np

class ChamferLoss(nn.Module):
    """
    Chamfer Distance loss for comparing point clouds
    Bidirectional distance between predicted and target point clouds
    """
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted point cloud (B, N, 3)
            target: Target point cloud (B, N, 3)
        """
        diff_matrix = torch.cdist(pred, target)  
        
        forward_min = torch.min(diff_matrix, dim=2)[0]  
        backward_min = torch.min(diff_matrix, dim=1)[0] 
        
        loss = torch.mean(forward_min) + torch.mean(backward_min)
        
        return loss
    
def load_xyz(file_path):
    """Load point cloud from XYZ file."""
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            x, y, z = map(float, line.strip().split())
            points.append([x, y, z])
    return np.array(points)


predicted_path = 'model_10_predicted.xyz'
actual_path = 'dataset/models/model_10/point_cloud.xyz'

predicted_points = load_xyz(predicted_path)
actual_points = load_xyz(actual_path)


predicted = torch.tensor(predicted_points, dtype=torch.float32).unsqueeze(0) 
actual = torch.tensor(actual_points, dtype=torch.float32).unsqueeze(0) 

chamfer_loss = ChamferLoss()

loss = chamfer_loss(predicted, predicted)

print(f"Number of points in predicted cloud: {len(actual_points)}")
print(f"Number of points in actual cloud: {len(actual_points)}")
print(f"Chamfer Loss: {loss.item():.6f}")