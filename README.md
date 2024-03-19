# torchmanager
### A generic deep learning training/testing framework for PyTorch
To use this framework, simply initialize a `Manager` object. The `Manager` class provides a generic training/testing loop for PyTorch models. It also provides some useful callbacks to use during training/testing.

## Pre-request
* Python 3.9+
* PyTorch
* tqdm
* PyYAML (Optional for yaml configs)
* scipy (Optional for FID metric)
* tensorboard (Optional for tensorboard recording)

## Installation
* PyPi: `pip install --pre torchmanager`
* Conda: `conda install -c kisonho torchmanager-nightly`

## Start from Configurations
The `Configs` class is designed to be inherited to define necessary configurations. It also provides a method to get configurations from terminal arguments.

```python
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

## Torchmanager Dataset
The `data.Dataset` class is designed to be inherited to define a dataset. It is a combination of `torch.utils.data.Dataset` and `torch.utils.data.DataLoader` with easier usage.

```python
from torchmanager.data import Dataset

# define dataset
class CustomDataset(Dataset):
    def __init__(self, ...):
        ...

    @property
    def unbatched_len(self) -> int:
        ...

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

# initialize datasets
training_dataset = CustomDataset(...)
val_dataset = CustomDataset(...)
testing_dataset = CustomDataset(...)
```

## The Manager
The `Manager` class is the core of the framework. It provides a generic training/testing loop for PyTorch models. The `Manager` class is designed to be inherited to manage the training/testing algorithm. There are also some useful callbacks to use during training/testing.

1. Initialize the manager with target model, optimizer, loss function, and metrics:
```python
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

- Multiple losses can be used by passing a dictionary to `loss_fn`:
```python
loss_fn = {
    'loss1': torchmanager.losses.CrossEntropy(),
    'loss2': torchmanager.losses.Dice(),
    ...
}
```

2. Train the model with `fit`` method:
```python
show_verbose: bool = ... # show progress bar information during training/testing
manager.fit(training_dataset, epochs=configs.epochs, val_dataset=val_dataset, show_verbose=show_verbose)
```

- There are also some other callbacks to use:
```python
tensorboard_callback = torchmanager.callbacks.TensorBoard('logs') # tensorboard dependency required
last_ckpt_callback = torchmanager.callbacks.LastCheckpoint(manager, 'last.model')
model = manager.fit(..., callbacks_list=[tensorboard_callback, last_ckpt_callback])
```

3. Test the model with test method:
```python
manager.test(testing_dataset, show_verbose=show_verbose)
```

4. Save the final trained PyTorch model:
```python
torch.save(model, "model.pth") # The saved PyTorch model can be loaded individually without using torchmanager
```

## Device selection during training/testing
Torchmanager automatically detects available devices to use during training/testing. GPU/MPS will be used in first priority if available. To specify other device to use, simply pass the device to the `fit` method for training and `test` method for testing:

1. Multi-GPU training/testing:
```python
# train on multiple GPUs
model = manager.fit(..., use_multi_gpus=True)

# test on multiple GPUs
manager.test(..., use_multi_gpus=True)
```

2. Use only specified GPUs for training/testing:
```python
# specify devices to use
gpus: Union[list[torch.device], torch.device] = ... # Notice: device id must be specified

# train on specified multiple GPUs
model = manager.fit(..., use_multi_gpus=True, devices=gpus)

# test on specified multiple GPUs
manager.test(..., use_multi_gpus=True, devices=gpus)
```

## Customize training/testing algorithm
The `Manager` class is designed to be inherited to manage the training/testing algorithm. To customize the training/testing algorithm, simply inherit the `Manager` class and override the `train_step` and `test_step` methods.
```python
class CustomManager(Manager):
    ...

    def train_step(x_train: torch.Tensor, y_train: torch.Tensor) -> Dict[str, float]:
        ...

    def test_step(x_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, float]:
        ...
```

## The saved experiment information
The `Experiment` class is designed to be used as a single callback to save experiment information. It is a combination of `torchmanager.callbacks.TensorBoard`, `torchmanager.callbacks.LastCheckpoint`, and `torchmanager.callbacks.BestCheckpoint` with easier usage.
```python
...

exp_callback = torchmanager.callbacks.Experiment('test.exp', manager) # tensorboard dependency required
model = manager.fit(..., callbacks_list=[exp_callback])
```

The information, including full training logs and checkpoints, will be saved in the following structure:
```
experiments
└── <experiment name>.exp
    ├── checkpoints
    │   ├── best-<metric name>.model
    │   └── last.model
    └── data
    │   └── <TensorBoard data file>
    ├── <experiment name>.cfg
    └── <experiment name>.log
```

## Please cite this work if you find it useful
```bibtex
@software{he_2023_10381715,
  author       = {He, Qisheng and
                  Dong, Ming},
  title        = {{TorchManager: A generic deep learning 
                   training/testing framework for PyTorch}},
  month        = dec,
  year         = 2023,
  publisher    = {Zenodo},
  version      = 1,
  doi          = {10.5281/zenodo.10381715},
  url          = {https://doi.org/10.5281/zenodo.10381715}
}
```

## Also checkout our projects implemented with torchmanager
* [MAG-MS/MAGNET](https://github.com/kisonho/magnet) - Modality-Agnostic Learning for Medical Image Segmentation Using Multi-modality Self-distillation
* [tlt](https://github.com/kisonho/tlt) - Transferring Lottery Tickets in Computer Vision Models: a Dynamic Pruning Approach
