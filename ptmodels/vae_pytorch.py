import torch


class SymmetricLinearAE(torch.nn.Module):
    """ Symmetric five-layer linear autoencoder model. Batch
        normalization is used between layers for both the encoder
        and decoder.
    """
    def __init__(self, n_layers: list, n_latent: int) -> None:
        """
        Args:
            n_layers (list): Starting and two intermediate layer
                sizes. n_layers[0] is the flattened input layer
                dimensionality.
            n_latent (int): Latent space dimensionality.
        """
        super().__init__()
        n1, n2, n3 = n_layers

        # Specify model flattener.
        self.flatten = torch.nn.Flatten()

        # Define five-layer encoder with batch normalization.
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n1, n2, bias=False),
            torch.nn.BatchNorm1d(n2),
            torch.nn.ReLU(),
            torch.nn.Linear(n2, n3, bias=False),
            torch.nn.BatchNorm1d(n3),
            torch.nn.ReLU(),
            torch.nn.Linear(n3, n_latent),
            torch.nn.BatchNorm1d(n_latent),
            torch.nn.ReLU()
        )

        # Define five-layer decoder with batch normalization.
        # It is a mirror of the encoder architecture apart from
        # the final activation being omitted.
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_latent, n3, bias=False),
            torch.nn.BatchNorm1d(n3),
            torch.nn.ReLU(),
            torch.nn.Linear(n3, n2, bias=False),
            torch.nn.BatchNorm1d(n2),
            torch.nn.ReLU(),
            torch.nn.Linear(n2, n1),
            torch.nn.BatchNorm1d(n1)
        )

        # Define final sigmoid activation layer.
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward call for autoencoder.

        Args:
            x (torch.Tensor): Input batch tensor.

        Returns:
            torch.Tensor: Predicted batch tensor.
        """
        x = self.flatten(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return self.sigmoid(x)


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
            loss = loss_fn(reconstruction, model.flatten(xb))

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
                            model.flatten(xv.to(device))
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
