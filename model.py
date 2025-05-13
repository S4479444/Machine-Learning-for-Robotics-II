import torch.nn as nn
import torch

class ShallowLinearNetwork(nn.Module):
    def __init__(self, n_input: int,  hidden_dims: list, n_output: int):
        super().__init__()

        self.input_layer = nn.Linear(n_input, hidden_dims[0])
        self.act1 = nn.ReLU()

        self.hidden_layers = []
        for i in range(1, len(hidden_dims)):
            self.hidden_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Dropout(0.1))

        self.out_layer = nn.Linear(hidden_dims[-1], n_output)
        self.act2 = nn.Softmax()

    def forward(self, x):
        x = self.act1(self.input_layer(x))
        for layer in self.hidden_layers:
            x = layer(x)
        return self.act2(self.out_layer(x))

        
class DeepLinearNetwork(nn.Module):
    def __init__(self, n_input: int, layers_dims: list, n_output: int, name: str):
        super().__init__()

        self.layers_dims = layers_dims

        self.in_layer = nn.Linear(n_input, layers_dims[0])
        self.in_act = nn.ReLU()
        self.in_dropout = nn.Dropout(0.1)
        self.name = name

        self.layers = []
        for i in range(len(layers_dims) - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(layers_dims[i], layers_dims[i+1]),
                    nn.ReLU(),
                    nn.Dropout(0.1))
                )
        
        self.out_layer = nn.Linear(layers_dims[-1], n_output)
        self.out_act = nn.Softmax()

    def forward(self, x):
        x = self.in_dropout(self.in_act(self.in_layer(x)))

        prev_xs = []

        H = len(self.layers_dims) - 1

        for i in range(len(self.layers)):
            curr_layer = self.layers[i]

            if i < H:
                prev_xs.append(x)
                x = curr_layer(x)
                
            else:
                prev_x = prev_xs[H - 1]
                x = curr_layer(x) + prev_x
            
            H -= 1

        return self.out_act(self.out_layer(x))
    
