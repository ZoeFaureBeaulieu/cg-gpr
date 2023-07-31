import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback


class TrainingLossCallback(Callback):
    def __init__(self):
        super().__init__()
        self.training_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.training_losses.append(outputs["loss"].item())


class MLP(pl.LightningModule):
    """
    A simple MLP model.
    Args
    ----------
        *layer_widths: List[int]
            A list of integers representing the width of each layer.
            The length of this list determines the number of layers in the MLP.
        activation: nn.Module, optional
            The activation function to use. Defaults to ReLU
    """

    def __init__(self, *layer_widths, activation=None):
        super().__init__()
        self.activation = nn.ReLU() if activation is None else activation

        # we have to store each layer in a ModuleList so that
        # pytorch can find the parameters of the model
        # when we pass model.parameters() to an optimizer
        self.layers = nn.ModuleList()
        for i in range(len(layer_widths) - 1):
            self.layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))

    def forward(self, x):
        """
        Forward pass of the MLP.
        Args
        ----------
            x:
                the input tensor
        Returns
        ----------
            _type_:
                the output tensor, this corresponds to the latent representation
        """
        # sequentially pass the output of each layer to the next layer
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        # don't apply the activation to the last layer
        x = self.layers[-1](x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)

        loss = F.mse_loss(output, y)

        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer
