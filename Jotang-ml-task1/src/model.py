import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        hidden_dim=None,
        hidden_dims=(32,),
        activation="relu",
        input_dim=2,
        output_dim=1,
    ):
        super().__init__()

        if hidden_dim is not None:
            hidden_dims = (hidden_dim,)
        elif isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)

        activation_layers = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        activation_name = activation.lower()
        if activation_name not in activation_layers:
            raise ValueError(
                "activation must be one of: relu, tanh, sigmoid"
            )

        self.hidden_dims = tuple(hidden_dims)
        self.activation = activation_name
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        previous_dim = input_dim
        for width in self.hidden_dims:
            layers.append(nn.Linear(previous_dim, width))
            layers.append(activation_layers[self.activation]())
            previous_dim = width

        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_config(self):
        return {
            "hidden_dims": self.hidden_dims,
            "activation": self.activation,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }
