import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import ast
import astunparse

# Neural network for predicting algorithm optimizations
class AlgorithmOptimizerNN(nn.Module):
    def __init__(self):
        super(AlgorithmOptimizerNN, self).__init__()
        self.fc1 = nn.Linear(20, 256)  # Input: 20 features -> Hidden layer: 256 neurons
        self.fc2 = nn.Linear(256, 128)  # Hidden layer: 256 neurons -> Hidden layer: 128 neurons
        self.fc3 = nn.Linear(128, 64)   # Hidden layer: 128 neurons -> Hidden layer: 64 neurons
        self.fc4 = nn.Linear(64, 32)    # Hidden layer: 64 neurons -> Hidden layer: 32 neurons
        self.fc5 = nn.Linear(32, 1)     # Hidden layer: 32 neurons -> Output: 1 neuron (optimization decision)
        self.sigmoid = nn.Sigmoid()     # Sigmoid activation for binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = torch.relu(self.fc2(x))  # Apply ReLU activation
        x = torch.relu(self.fc3(x))  # Apply ReLU activation
        x = torch.relu(self.fc4(x))  # Apply ReLU activation
        x = self.sigmoid(self.fc5(x))  # Output probability (0 = not optimized, 1 = optimized)
        return x

# Experience replay memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Function for encoding an AST node into a vector
def encode_node(node):
    """Convert AST node into a fixed-size feature vector"""
    feature_vector = [0] * 20
    if isinstance(node, ast.FunctionDef):
        feature_vector[0] = 1
        feature_vector[5] = len(node.body)
    elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
        feature_vector[1] = 1
    elif isinstance(node, ast.Assign):
        feature_vector[2] = 1
    elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Str):
        comment_length = len(node.value.s)
        if comment_length > 40:
            feature_vector[3] = 1
            feature_vector[4] = comment_length
    elif isinstance(node, (ast.For, ast.While)):
        feature_vector[6] = 1  # Indicate presence of a loop
    elif isinstance(node, (ast.If, ast.Compare)):
        feature_vector[7] = 1  # Indicate presence of a conditional statement
    elif isinstance(node, ast.BinOp):
        feature_vector[8] = 1  # Indicate presence of a binary operation
    return torch.tensor(feature_vector, dtype=torch.float32)

# Function to extract features from the code
def extract_features_from_code(code):
    try:
        tree = ast.parse(code)
        features = [encode_node(node) for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Assign, ast.Expr, ast.For, ast.While, ast.If, ast.Compare, ast.BinOp))]
        return features
    except Exception as e:
        print(f"[!] AST parsing error: {e}")
        return []

# Function to generate random code for training
def generate_random_code():
    # This function should be implemented to generate random Python code snippets
    # For simplicity, we return a fixed code snippet here
    code = """
import torch
import numpy as np

def example_function(x):
    y = np.sin(x)
    return y
"""
    return code

# Training with Reinforcement Learning
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_episodes = 1000
    max_steps = 20
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 64
    memory_capacity = 10000

    model = AlgorithmOptimizerNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    memory = ReplayMemory(memory_capacity)

    rewards = []
    losses = []

    for episode in range(num_episodes):
        code = generate_random_code()
        state = extract_features_from_code(code)
        total_reward = 0
        episode_loss = 0

        for step in range(max_steps):
            if not state:
                break

            state_tensor = torch.stack(state).to(device)
            if random.random() < epsilon:
                action = random.choice([0, 1])  # Random action (0 = not optimized, 1 = optimized)
            else:
                with torch.no_grad():
                    q_values = model(state_tensor)
                    action = 1 if q_values.mean().item() >= 0.5 else 0

            next_code = generate_random_code()  # Generate next code (replace with actual next code generation)
            next_state = extract_features_from_code(next_code)
            reward = random.random()  # Random reward (replace with actual reward calculation)
            memory.push(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            if len(memory) >= batch_size:
                batch = memory.sample(batch_size)
                state_batch = torch.stack([torch.stack(b[0]) for b in batch if b[0]]).to(device)
                action_batch = torch.tensor([b[1] for b in batch], dtype=torch.float32).unsqueeze(1).to(device)
                reward_batch = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1).to(device)
                next_state_batch = torch.stack([torch.stack(b[3]) for b in batch if b[3]]).to(device)

                current_q_values = model(state_batch).mean(dim=1, keepdim=True)
                max_next_q_values = model(next_state_batch).mean(dim=1, keepdim=True).detach()
                expected_q_values = reward_batch + (gamma * max_next_q_values)

                loss = criterion(current_q_values, expected_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_loss += loss.item()

        epsilon = max(epsilon_min, epsilon_decay * epsilon)
        rewards.append(total_reward)
        losses.append(episode_loss)
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}, Loss: {episode_loss:.4f}")

    torch.save(model.state_dict(), "algorithm_optimizer.pth")
    print("[+] Model trained and saved as 'algorithm_optimizer.pth'")

    # Plot training schedule
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title('Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('training_schedule.png')
    plt.show()

if __name__ == "__main__":
    train_model()