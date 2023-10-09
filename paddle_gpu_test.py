'''
python paddle_gpu_test.py
'''
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.vision.transforms as T
from paddle.vision.datasets import RandomDataset
from paddle.io import DataLoader

# Check if GPU is available
if paddle.is_compiled_with_cuda():
    place = paddle.CUDAPlace(0)  # Use the first GPU (change the index if needed)
else:
    print("GPU support is not available.")
    exit()

# Define a simple neural network
class Net(nn.Layer):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

# Create a synthetic dataset and data loader
transform = T.Compose([T.Normalize(mean=[0.5], std=[0.5])])
dataset = RandomDataset(num_samples=1000, num_classes=2, transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize the model and optimizer
model = Net()
optimizer = optim.Adam(parameters=model.parameters(), learning_rate=0.01)

# Training loop
epochs = 10
for epoch in range(epochs):
    for batch_id, data in enumerate(data_loader()):
        x_data = data[0]
        y_data = data[1]
        
        # Move data to GPU
        x_data = paddle.to_tensor(x_data, place)
        y_data = paddle.to_tensor(y_data, place)
        
        # Forward pass
        output = model(x_data)
        loss = paddle.nn.functional.mean(paddle.nn.functional.square(output - y_data))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.numpy()}")

print("Training complete!")
