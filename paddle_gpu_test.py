'''
python paddle_gpu_test.py
'''
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.vision.transforms as T
from paddle.vision.datasets import MNIST

# Define a CNN model
class Net(nn.Layer):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2D(1, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 26 * 26, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = paddle.relu(self.conv1(x))
        x = paddle.flatten(x, 1)
        x = paddle.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create a dataset and data loader
transform = T.Compose([T.Resize((28, 28)), T.ToTensor()])
train_dataset = MNIST(mode='train', transform=transform)
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the model and optimizer
model = Net()
optimizer = optim.Adam(parameters=model.parameters(), learning_rate=0.001)

# Train the model
for epoch in range(10):
    for batch_id, data in enumerate(train_loader()):
        x_data = data[0]
        y_data = data[1]
        logits = model(x_data)
        loss = paddle.nn.functional.cross_entropy(logits, y_data)
        avg_loss = paddle.mean(loss)

        avg_loss.backward()
        optimizer.step()
        optimizer.clear_grad()

    print(f"Epoch [{epoch + 1}/10], Loss: {avg_loss.numpy()}")
