import time
import torch


class Timer:

    def __init__(self, name="Timer"):
        self.name = name

    def __enter__(self):
        self.begin = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # print "in __exit__"
        print("{}: elapsed {:.2f}s.".format(
            self.name, time.time() - self.begin))


def accuracy(output, labels):
    """Return accuracy of output compared to labels.
    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels
    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels).to(output.device)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
