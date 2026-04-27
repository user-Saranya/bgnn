import torch
import torch.nn as nn


class NeuralDecisionTree(nn.Module):
    def __init__(self, input_dim, output_dim, depth=3):
        super().__init__()

        self.depth = depth
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.num_internal = 2 ** depth - 1
        self.num_leaves = 2 ** depth

        # Decision nodes
        self.decision = nn.Linear(input_dim, self.num_internal)

        # Leaf values
        self.leaf_values = nn.Parameter(
            torch.randn(self.num_leaves, output_dim) * 0.1  # scaled init for stability
        )

    def forward(self, x):
        """
        x: [N, input_dim]
        returns: [N, output_dim]
        """

        batch_size = x.size(0)

        decision_probs = torch.sigmoid(self.decision(x))  # [N, num_internal]

        mu = x.new_ones(batch_size, 1)

        idx = 0  # correct index tracker

        for d in range(self.depth):
            nodes = decision_probs[:, idx: idx + 2 ** d]  # correct slicing
            idx += 2 ** d

            mu = mu.unsqueeze(-1)  # [N, current_leaves, 1]

            mu = torch.cat([
                mu * nodes.unsqueeze(-1),          # left
                mu * (1 - nodes.unsqueeze(-1))     # right
            ], dim=-1)

            mu = mu.view(batch_size, -1)

        # mu: [N, num_leaves]

        # Normalize (important for numerical stability)
        mu = mu / (mu.sum(dim=1, keepdim=True) + 1e-8)

        out = torch.matmul(mu, self.leaf_values)

        return out


class NeuralDecisionForest(nn.Module):
    def __init__(self, input_dim, output_dim, num_trees=5, depth=3):
        super().__init__()

        self.trees = nn.ModuleList([
            NeuralDecisionTree(input_dim, output_dim, depth)
            for _ in range(num_trees)
        ])

    def forward(self, x):
        outputs = [tree(x) for tree in self.trees]
        return torch.stack(outputs, dim=0).mean(dim=0)
