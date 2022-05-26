# torchmanager
A highly-wrapped PyTorch training and testing manager

## Pre-request
* Python 3.8+
* PyTorch 1.8.2+
* tensorboard
* tqdm

## Installation
`pip install torchmanager`

## The Manager
- Initialize the manager with target model,  optimizer, loss function, and metrics:
[image:res/torchmanager.png]

* Test the model with test method:
```
# get dataset
testing_dataset: DataLoader = ...

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
