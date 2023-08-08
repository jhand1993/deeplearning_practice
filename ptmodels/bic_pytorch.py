import torch

class Logistic(torch.nn.Module):
    """ This is a simple logistics regression implemented with PyTorch. 
    """
    def __init__(self, n_in: int):
        """ 
        Args:
            n_in (int): Dimension of input. 
        """
        super().__init__()
        self.logistic = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_in, 1), 
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        """ Forward call.

        Args:
            x (tensor.Tensor): Input tensor of dimension 4.

        Returns:
            tesnor.Tesnor: output logits tensor of dimension 1. 
        """
        return torch.squeeze(self.logistic(x).T) # return logits.

class Perceptron(torch.nn.Module):
    """ Three layer perceptron NN model.
    """
    def __init__(self, n_layers: list):
        """
        Args:
            n_layers (sequential): List of layer dimensions.
        """
        super().__init__()
        n0, n1, n2 = n_layers
        self.ml_percep = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n0, n1),
            torch.nn.ReLU(),
            torch.nn.Linear(n1, n2),
            torch.nn.ReLU(),
            torch.nn.Linear(n2, 1)
        )
    
    def forward(self, x):
        return torch.squeeze(self.logistic(x).T) # return logits.
    

def train_bic_model(
        train_dl: torch.utils.data.DataLoader,
        valid_dl: torch.utils.data.DataLoader,
        model: torch.nn.Module, 
        opt: torch.optim.Optimizer,
        loss_fn: callable, 
        n_epoch: int
    ) -> float:
    """ Trains binary classification model 'model'. 'opt' should be a 
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
    
    Returns:
        float: Validation loss for final epoch. 
    """
    # This extracts the DataLoader's input DataSet object and returns its length. 
    full_size = len(train_dl.dataset)

    for epoch in range(n_epoch):
        # We need to set the model to 'train mode'.
        model.train()

        # Iterate throught he batched training data. We use enumerate to keep track 
        # of the batch number for logging purposes
        for batch, (xb, yb) in enumerate(train_dl):
            # First, feed forward data through the model and get prediction.
            pred = model(xb)

            # Second, calculate the loss between prediction and dependent variable.
            loss = loss_fn(pred, yb.float())

            # Third, propagate the backward gradient calculations.
            loss.backward()

            # Fourth, use loss gradient to update parameters. 
            opt.step()

            # Fifth, after each batch, zero the gradients. 
            opt.zero_grad()

            # Log training loss every 25 batches. 
            if batch % 25 == 0:
                # This converts the current batch index to the sample size. 
                current_size = (batch + 1) * len(xb)
                print(
                    f'Batch [{current_size:>5d}/{full_size:>5d}] loss: {loss.item():7f}.'
                )
        
        # Now that the model has been training, we can evaluate its performance
        # using the validation set by setting the model to 'evalucation model'.
        model.eval()

        # We don't want to keep track of gradients for validation. It's a waste of
        # resources!

        with torch.no_grad():
            # We don't need to call the optimizer here since we are just 
            # evaluating. 
            losses, n_b = map(
                torch.tensor,
                zip(*[(loss_fn(model(xv), yv.float()), len(yv)) for xv, yv in valid_dl])
            )

            # The code above is a bit dense, so I've broken it up into a for-loop 
            # below for a more verbose equivalence for reference. 
            # losses = []
            # n_b = []
            # for xv, yv in valid_dl:
            #     pred = model(xv)
            #     loss = loss_fn(pred, yv)
            #     losses.append(loss)
            #     n_b.append(len(yv))
            # losses = torch.tensor(losses)
            # n_b = torch.tensor(n_b)

            validation_loss = torch.sum(losses) / len(n_b)
        
        # Log validation loss after each training epoch. 
        print(f'Epoch {epoch} validation loss: {validation_loss.item():>7f}.')
        print('---------------------------------------')
    return validation_loss


def reset_params(model: torch.nn.Module) -> None:
    """ Resets parameters for valid layers. 

    Args:
        model (torch.nn.Module): Model with parameters to reset. 
    """
    for layer in model.children():
        # Not all layers have parameters (e.g. Flatten()).
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()