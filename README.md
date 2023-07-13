# torchmanager
### A highly-wrapped PyTorch training and testing manager

## Pre-request
* Python 3.9+
* PyTorch
* tqdm
* scipy (Optional)
* tensorboard (Optional)

## Installation
* PyPi: `pip install torchmanager-nightly`
* Conda: `conda install -c kisonho torchmanager-nightly`

## The Manager
1. Start with configurations:
```
from torchmanager.configs import Configs as _Configs

# define necessary configurations
class Configs(_Configs):
    epochs: int
    lr: float

    ...

    def get_arguments(parser: Union[argparse.ArgumentParser, argparse._ArgumentGroup] = argparse.ArgumentParser()) -> Union[argparse.ArgumentParser, argparse._ArgumentGroup]:
        '''Add arguments to argument parser'''
        ...

    def show_settings(self) -> None:
        ...

# get configs from terminal arguments
configs = Configs.from_arguments()
```

2. Initialize the manager with target model, optimizer, loss function, and metrics:
```
import torch, torchmanager

# define model
class PytorchModel(torch.nn.Module):
    ...

# initialize model, optimizer, loss function, and metrics
model = PytorchModel(...)
optimizer = torch.optim.SGD(model.parameters(), lr=configs.lr)
loss_fn = torchmanager.losses.CrossEntropy()
metrics = {'accuracy': torchmanager.metrics.SparseCategoricalAccuracy()}

# initialize manager
manager = torchmanager.Manager(model, optimizer, loss_fn=loss_fn, metrics=metrics)
```

3. Train the model with fit method:
```
from torchmanager.data import Dataset

# get datasets
training_dataset: Dataset = ...
val_dataset: Dataset = ...

# train with fit method
manager.fit(training_dataset, epochs=configs.epochs, val_dataset=val_dataset)
```

- There are also some other callbacks to use:
```
...

tensorboard_callback = torchmanager.callbacks.TensorBoard('logs') # tensorboard dependency required
last_ckpt_callback = torchmanager.callbacks.LastCheckpoint(manager, 'last.model')
model = manager.fit(..., callbacks_list=[tensorboard_callback, last_ckpt_callback])
```

- Or use `callbacks.Experiment` to handle both `callbacks.TensorBoard` and `callbacks.LastCheckpoint`:
```
...

exp_callback = torchmanager.callbacks.Experiment('test.exp', manager) # tensorboard dependency required
model = manager.fit(..., callbacks_list=[exp_callback])
```

4. Test the model with test method:
```
# get dataset
testing_dataset: Dataset = ...

# test with test method
manager.test(testing_dataset)
```

5. Save final model in PyTorch format:
```
torch.save(model, "model.pth")
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
