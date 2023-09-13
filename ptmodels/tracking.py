""" Tracker objects for keeping track of training/validation loss. """
from typing import List
from torch import tensor, Tensor, sum


class BaseTracker(object):
    """ Base tracker object.  Each list tracks the epoch, iteration,
        and loss per entry.
    """
    def __init__(
        self,
    ) -> None:
        self._train_loss = []
        self._validation_loss = []

    def track_train_loss(
        self,
        record: List[int | float]
    ) -> None:
        """ Appends record to _train_loss.

        Args:
            record (tuple[int  |  float]): Record to append to _train_loss
        """
        self._train_loss.append(record)

    def track_validation_loss(
        self,
        record: List[int | float]
    ) -> None:
        """ Appends record to _validation_loss.

        Args:
            record (tuple[int  |  float]): Record to append to _validation_loss
        """
        self._validation_loss.append(record)

    def get_train_tracker(self) -> Tensor:
        """ Returns _train_loss

        Returns:
            List: Copy of _train_loss
        """
        return self._train_loss.copy()

    def get_validation_tracker(self) -> Tensor:
        """ Returns _validation_loss

        Returns:
            List: Copy of _validation_loss
        """
        return self._validation_loss.copy()

    @staticmethod
    def stratify_loss_list(x: List) -> List:
        """ Takes either _train_loss or _validation_loss and splits stratifies
            records into sublists for each epoch.

        Args:
            x (list): Either _train_loss or _validation_loss

        Returns:
            list: Stratified list of _train_loss or _validation_loss lists.
        """
        # Get final epoch index from last entry in x.
        n = x[-1][0]

        return [
            [record[1:] for record in x if record[0] == i]
            for i in range(1, n + 1)
        ]

    @staticmethod
    def get_epoch_loss(x: List) -> Tensor:
        """ calculates epoch loss for list x.  X should be instances of
            _train_loss or _validation_loss attributes.

        Args:
            x (list): Either _train_loss or _validation_loss

        Returns:
            Tensor: Tensor of epoch losses.
        """
        x_strat = BaseTracker.stratify_loss_list(x)
        return tensor(
            [sum(tensor(row)[:, 1]) / len(row) for row in x_strat]
        )

    def get_train_epoch_loss(self) -> Tensor:
        """ Calls get_epoch_loss on _train_loss

        Returns:
            Tensor: Tensor of training epoch losses.
        """
        return BaseTracker.get_epoch_loss(self._train_loss)

    def get_validation_epoch_loss(self) -> Tensor:
        """ Calls get_epoch_loss on _validation_loss

        Returns:
            Tensor: Tensor of validation epoch losses.
        """
        return BaseTracker.get_epoch_loss(self._validation_loss)
