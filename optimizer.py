import ast
import os
import sys
import torch
import torch.nn as nn
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

# Class for AST optimization using a neural network
class NeuralCodeOptimizer(ast.NodeTransformer):
    def __init__(self, model):
        self.model = model

    def visit(self, node):
        """Apply the neural network to decide whether to optimize the node"""
        if isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Assign, ast.Expr, ast.For, ast.While, ast.If, ast.Compare, ast.BinOp)):
            features = encode_node(node).unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = self.model(features).item()
            if prediction < 0.5:  # If the neural network considers the node unnecessary, remove it
                return None
        return self.generic_visit(node)

# Main function for code optimization
def optimize_code(filename, model):
    """Optimize Python script using AST and a neural network"""
    with open(filename, "r", encoding="utf-8") as f:
        source_code = f.read()

    try:
        tree = ast.parse(source_code)
        optimizer = NeuralCodeOptimizer(model)
        optimized_tree = optimizer.visit(tree)

        # Remove `None` from `Module.body` to avoid errors
        optimized_tree.body = [node for node in optimized_tree.body if node is not None]
        optimized_tree = ast.fix_missing_locations(optimized_tree)  # Fix AST

        optimized_code = astunparse.unparse(optimized_tree)  # Convert AST back to source
    except Exception as e:
        print(f"[!] AST parsing error: {e}")
        return False

    if optimized_code.strip() != source_code.strip():
        with open(filename, "w", encoding="utf-8") as f:
            f.write(optimized_code)
        print("[+] Code optimized successfully!")
        return True
    else:
        print("[=] No improvements found.")
        return False

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the pre-trained model
    model = AlgorithmOptimizerNN().to(device)
    model.load_state_dict(torch.load("algorithm_optimizer.pth", map_location=device))

    if len(sys.argv) != 2:
        print("Usage: python optimizer.py <path_to_python_file>")
        sys.exit(1)

    script_path = sys.argv[1]
    if optimize_code(script_path, model):
        print(f"[+] Optimization completed for {script_path}")
    else:
        print(f"[=] No improvements found for {script_path}")