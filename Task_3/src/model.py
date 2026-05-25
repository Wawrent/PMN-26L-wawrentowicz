import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, kernel_size=3, pooling="max"):

        super(CNN, self).__init__()
        padding = kernel_size // 2
        if pooling == "max":
            pool_layer = nn.MaxPool2d(2, 2)

        elif pooling == "avg":
            pool_layer = nn.AvgPool2d(2, 2)

        else:
            pool_layer = nn.Identity()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=kernel_size,
                padding=padding
            ),

            nn.ReLU(),
            pool_layer,

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=kernel_size,
                padding=padding
            ),

            nn.ReLU(),
            pool_layer
        )

        if pooling == "none":
            fc_input = 64 * 32 * 32
        else:
            fc_input = 64 * 8 * 8

        self.classifier = nn.Sequential(

            nn.Flatten(),
            nn.Linear(fc_input, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)
        return x