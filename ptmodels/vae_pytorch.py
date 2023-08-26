import torch
from typing import Iterable
from math import sqrt


class BaseAE(torch.nn.Module):
    """ Torch.nn.Module subclass with autoencoder functionality.
        The final layer is assumed to be a sigmoid activation.
    """
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Identity()
        self.decoder = torch.nn.Identity()

        # Define final sigmoid activation that will be used after
        # decoding.
        self.last_activation = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward call for autoencoder.

        Args:
            x (torch.Tensor): Input batch tensor.

        Returns:
            torch.Tensor: Predicted batch tensor.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return self.last_activation(x)

    def get_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """ Returns the latent encoded representation for the batch 'x'.

        Args:
            x (torch.tensor): Batch of images that will be encoded.

        Returns:
            torch.Tensor: Latent representation for each element from batch
                'x'.
        """
        x = self.encoder(x)
        return x

    def get_decoding(self, x: torch.Tensor) -> torch.Tensor:
        """ Returns decoding prediction from encoded representation for the
            batch 'x'.

        Args:
            x (torch.tensor): Batch of encodings that will be decoded.

        Returns:
            torch.Tensor: Predicted representation for each element from batch
                'x'.
        """
        x = self.decoder(x)
        x = self.last_activation(x)
        return x

    @staticmethod
    def normalize_encoding(x: torch.Tensor) -> torch.Tensor:
        """ Normalizes each of the latent components to have mean of
            zero and standard deviation of one.

        Args:
            x (torch.Tensor): Tensor of encoding vectors.

        Returns:
            torch.Tensor: Normalized tensor of encoding vectors.
        """
        # We want to retain the dimensionality of the mean and std vectors
        # --- this will allow easy broadcasting of element-wise addition
        # and division.
        mu = torch.mean(x, -1, keepdim=True)
        sig = torch.std(x, -1, keepdim=True)
        return (x - mu) / sig


class DenseAE(BaseAE):
    """ Five-layer linear autoencoder model. Batch normalization is used
        between layers for both the encoder and decoder.
    """
    def __init__(self, n_layers: Iterable[int], n_latent: int) -> None:
        """
        Args:
            n_layers (Iterable[int]): Starting and two intermediate layer
                sizes. n_layers[0] is the flattened input layer
                dimension.
            n_latent (int): Latent space dimension.
        """
        super().__init__()
        # The encoder and decoder has duplicated blocks, so these
        # have been iterated and unpacked as their own Sequential instances.
        # This also allows for an arbitrary number of hidden layers.
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            *[
                DenseAE.ln_block(n_layers[i], n_layers[i + 1])
                for i in range(len(n_layers) - 1)
            ],
            torch.nn.Linear(n_layers[-1], n_latent, bias=False),
            torch.nn.BatchNorm1d(n_latent),
            torch.nn.Tanh()
        )

        self.decoder = torch.nn.Sequential(
            DenseAE.ln_block(n_latent, n_layers[-1]),
            *[
                DenseAE.ln_block(n_layers[i + 1], n_layers[i])
                for i in range(len(n_layers) - 2, 0, -1)
            ],
            torch.nn.Linear(n_layers[1], n_layers[0], bias=False),
            torch.nn.BatchNorm1d(n_layers[0])
        )

    @staticmethod
    def ln_block(n_in: int, n_out: int) -> torch.nn.Sequential:
        """ Linear network block with batch normalization and
            ReLU activation

        Args:
            n_in (int): Input size.
            n_out (int): Output size

        Returns:
            torch.nn.Sequential: Dense network, batch normalization,
                and ReLU activation sequence.
        """
        return torch.nn.Sequential(
            torch.nn.Linear(n_in, n_out, bias=False),
            torch.nn.BatchNorm1d(n_out),
            torch.nn.ReLU()
        )


class ConvNetAE(BaseAE):
    """ Image autoencoder roughly mirroring AlexNet.  Square kernels
        are used for each convnet layer.  Max pooling decreases
        dimensionality by a factor of two per-axis.
    """
    def __init__(
            self, channels: Iterable[int], k_layers: Iterable[int],
            n_flat: int, n_latent: int
    ) -> None:
        """
        Args:
            channels (Iterable[int]): List of input and output channels.
                These are chained together.
            k_layers (Iterable[int]): List of square kernel sizes.
            n_flat (int): flattened linear layer input size.  Must be
                the square of a non-zero integer, which will necessarily
                be true if images are square.
            n_latent (int): Latent space dimension.

        Raises:
            ValueError: channels and k_layers inconsistent.
        """
        super().__init__()
        # Make sure number of channels and kernels are consistent.
        try:
            assert len(channels) == len(k_layers) + 1

        except AssertionError:
            raise ValueError('channels and k_layers lengths inconsistent.')

        # Make sure n_latent is the square of a non-zero integer.
        try:
            assert sqrt(n_flat) % 1. == 0.

        except AssertionError:
            raise ValueError(
                'n_flat must be the square of a non-zero integer.'
            )

        # This is a bit obtuse, but a method in the madness is present.
        # The first list creates len(k_layer) ConvNet blocks per user request.
        # The second list adds the dense linear portion of the encoder network.
        # The two combined lists are then unpacked as arguments for
        # Sequential().
        encoder_list = [
            ConvNetAE.cn_block(
                channels[i], channels[i + 1], k_layers[i], 2
            ) for i in range(len(k_layers))

        ] + [
            torch.nn.Flatten(),
            torch.nn.Linear(n_flat, n_latent, bias=False),
            torch.nn.BatchNorm1d(n_latent),
            torch.nn.Tanh()
        ]

        self.encoder = torch.nn.Sequential(*encoder_list)

        # Same reasoning as that provided above. Note we need to unflatten
        # the n_flat into the first ConvNet layer of the decoder.
        nl_in = int(sqrt(n_flat))
        decoder_list = [
            torch.nn.Linear(n_latent, n_flat, bias=False),
            torch.nn.BatchNorm1d(n_flat),
            torch.nn.ReLU(),
            torch.nn.Unflatten(-1, (nl_in, nl_in))
        ] + [
            ConvNetAE.tcn_block(
                channels[i + 1], channels[i], 3
            ) for i in range(len(k_layers) - 2, 0, -1)
        ] + [
            torch.nn.ConvTranspose2d(channels[1], channels[0], 3)
        ]

        self.decoder = torch.nn.Sequential(*decoder_list)

    @staticmethod
    def cn_block(
        c_in: int, c_out: int, k: int, p: int
    ) -> torch.nn.Sequential:
        """ ConvNet block that includes square kernel convolution
            with stride=1, ReLU activation, and max pooling.

        Args:
            c_in (int): Number of input channels.
            c_out (int): Number of output channels.
            k (int): Square kernel size.
            p (int): Pooling size and stride.

        Returns:
            torch.nn.Sequential: Convnet block.
        """
        return torch.nn.Sequential(
            torch.nn.Conv2d(c_in, c_out, k, stride=1, padding='same'),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(p, p)
        )

    @staticmethod
    def tcn_block(
        c_in: int, c_out: int, k: int
    ) -> torch.nn.Sequential:
        """ Transpose convnet block with an ReLU activation and stride=1.
            Note that the pooling size/stride size + 1 should be used for the
            transpose kernel.

        Args:
            c_in (int): Number of input channels.
            c_out (int): Number of output channels.
            k (int): Square kernel size.

        Returns:
            torch.nn.Sequential: Transpose convent block.
        """
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(c_in, c_out, k, stride=1),
            torch.nn.ReLU()
        )


def train_AE(
        train_dl: torch.utils.data.DataLoader,
        valid_dl: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        opt: torch.optim.Optimizer,
        loss_fn: callable,
        n_epoch: int,
        device: str
) -> float:
    """ Trains an autoencoder model 'model'. 'opt' should be a
        PyTorch Optimizer object, while loss must be a callable scalar
        function. Includes basic loss logging during training and validation.

    Args:
        train_dl (torch.utils.data.DataLoader): Training DataLoader object.
        train_dl (torch.utils.data.DataLoader): Validation DataLoader object.
        model (torch.nn.Module): PyTorch model assumed to be sequential. More
            importantly, it requires a scalar tensor output with values 0 or 1.
        opt (torch.optim.Optimizer): Optimizer object instance.
        loss_fn (callable): Scalar function used to calculate loss.
        n_epoch (int): Number of epochs.
        device (str): Device to run Tensors on.

    Returns:
        float: Validation loss for final epoch.
    """
    # This extracts the DataLoader's input DataSet object and returns its
    # length.
    fs = len(train_dl.dataset)

    for epoch in range(n_epoch):
        # We need to set the model to 'train mode'.
        model.train()

        # Train each batch.
        for i, (xb, _) in enumerate(train_dl):

            # Assign device to input/label tensors.
            xb = xb.to(device)

            # Get AE output vector.
            reconstruction = model(xb)

            # Calculate loss.
            loss = loss_fn(reconstruction, torch.nn.Flatten()(xb))

            # Backpropagate.
            loss.backward()

            # Update parameters with optimizer.
            opt.step()

            # Log training loss every 25 batches.
            if i % 200 == 0:
                # This converts the current batch index to the sample size.
                s = (i + 1) * len(xb)
                print(
                    f'Batch [{s:>5d}/{fs:>5d}] loss: {loss.item():7f}.'
                )

            # After each batch, zero the gradients.
            opt.zero_grad()

        # Now that the model has been training, we can evaluate its performance
        # using the validation set by setting the model to 'evalucation model'.
        model.eval()

        # We don't want to keep track of gradients for validation. It's a
        # waste of resources!
        with torch.no_grad():
            # We don't need to call the optimizer here since we are just
            # evaluating.
            losses, n_b = map(
                torch.tensor,
                zip(
                    *[
                        (loss_fn(
                            model(xv.to(device)),
                            torch.nn.Flatten()(xv.to(device))
                        ),
                            xv.shape[0])
                        for xv, _ in valid_dl
                    ]
                )
            )

            # The code above is a bit dense, so I've broken it up into a
            # for-loop below for a more verbose equivalence for reference.
            # losses = []
            # n_b = []
            # for xv, _ in valid_dl:
            #     pred = model(xv.to(device))
            #     loss = loss_fn(pred, xv.to(device))
            #     losses.append(loss)
            #     n_b.append(len(xv.shape[0]))
            # losses = torch.tensor(losses)
            # n_b = torch.tensor(n_b)

            validation_loss = torch.sum(losses) / len(n_b)

        # Log validation loss after each training epoch.
        print(
            f'Epoch {epoch + 1} validation loss: {validation_loss.item():>7f}.'
        )
        print('---------------------------------------')
    return validation_loss
