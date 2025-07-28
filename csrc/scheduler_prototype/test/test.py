import torch
import torch.nn as nn
import torch.nn.functional as F

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create some tensors on GPU
x = torch.randn(1024, 1024, device=device)
y = torch.randn(1024, 1024, device=device)

# Perform matrix multiplication (this will launch CUDA kernels)
z = torch.matmul(x, y)

# Verify the result is correct
assert z.shape == (1024, 1024), "Matrix multiplication result shape is incorrect"

# # Perform a more complex operation (convolution) that will also launch kernels
# if True:
#     # Create input tensor: (batch_size, channels, height, width)
#     input_tensor = torch.randn(1, 3, 256, 256, device=device)
    
#     # Create a simple CNN
#     class SimpleCNN(nn.Module):
#         def __init__(self):
#             super(SimpleCNN, self).__init__()
#             self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
#             self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
#         def forward(self, x):
#             x = F.relu(self.conv1(x))
#             x = F.max_pool2d(x, 2)
#             x = F.relu(self.conv2(x))
#             x = F.max_pool2d(x, 2)
#             return x
    
#     model = SimpleCNN().to(device)
#     output = model(input_tensor)
#     print("Output shape:", output.shape)

print("Done!")