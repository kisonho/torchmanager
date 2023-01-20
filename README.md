# torchmanager
### A highly-wrapped PyTorch training and testing manager

*This branch is used for beta unstable release. Please check **main** branch for the latest main release version*

## Pre-request
* Python 3.8+
* PyTorch
* tqdm
* tensorboard (Optional)

## Installation
* PyPi: `pip install torchmanager-nightly`
* Conda: `conda install -c kisonho torchmanager-nightly`

## The Manager
- Initialize the manager with target model, optimizer, loss function, and metrics:
```
import torch, torchmanager

# define model
class PytorchModel(torch.nn.Module):
    ...

# initialize model, optimizer, loss function, and metrics
model = PytorchModel(...)
optimizer = torch.optim.SGD(model.parameters())
loss_fn = torchmanager.losses.CrossEntropy()
metrics = {'accuracy': torchmanager.metrics.SparseCategoricalAccuracy()}

# initialize manager
manager = torchmanager.Manager(model, optimizer, loss_fn=loss_fn, metrics=metrics)
```

- Train the model with fit method:
```
from torchmanager.data import Dataset

# get datasets
training_dataset: Dataset = ...
val_dataset: Dataset = ...

# train with fit method
manager.fit(training_dataset, epochs=10, val_dataset=val_dataset)
```

* Test the model with test method:
```
# get dataset
testing_dataset: Dataset = ...

# test with test method
manager.test(testing_dataset)
```

- There are also some other callbacks to use:
```
...

tensorboard_callback = torchmanager.callbacks.TensorBoard('logs')
last_ckpt_callback = torchmanager.callbacks.LastCheckpoint(model, 'last.model')
manager.fit(..., callbacks_list=[tensorboard_callback, last_ckpt_callback])
```

## Custom your training loop
1. Create your own manager class by extending the `Manager` class:
```
...

class CustomManager(Manager):
    ...
```

2. Override the `train_step` method:
```
class CustomManager(Manager):
    ...
    
    def train_step(x_train: torch.Tensor, y_train: torch.Tensor) -> Dict[str, float]:
        ...
```
