import torch

# totally subjective, we can change later if we want
COST_MATRIX = torch.tensor([
# predicted:  D    C    B    A
            [0.0, 0.5, 2.0, 4.0], # D
            [0.5, 0.0, 0.5, 2.0], # C
            [2.0, 0.5, 0.0, 0.5], # B
            [4.0, 2.0, 0.5, 0.0], # A
])