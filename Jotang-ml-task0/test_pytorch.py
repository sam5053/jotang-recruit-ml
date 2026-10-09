import sys

import torch


def main():
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    tensor_a = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        device=device,
    )
    tensor_b = torch.tensor(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        device=device,
    )
    tensor_c = tensor_a @ tensor_b

    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {cuda_available}")
    print(f"Computation device: {device}")
    print(f"Tensor A shape: {tuple(tensor_a.shape)}")
    print(f"Tensor B shape: {tuple(tensor_b.shape)}")
    print(f"Tensor C = A @ B:\n{tensor_c}")
    print(f"Tensor C shape: {tuple(tensor_c.shape)}")


if __name__ == "__main__":
    main()
