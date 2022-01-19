# **# torchmanager**
A Keras like PyTorch training and testing manager

## The Manager
- Initialize the manager with target model,  optimizer, loss function, and metrics:
```
import torch, torchmanager

# define model
class PytorchModel(torch.nn.Module):
	...

# initialize model, optimizer, loss function, and metrics
model = PytorchModel(...)
optimizer = torch.optim.SGD(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()
metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {'accuracy': lambda y, y_true: float(y.argmax(dim=1).eq(y_true).to(torch.float32).mean())}

# initialize manager
manager = torchmanager.Manager(model, optimizer, loss_fn=loss_fn, metrics=metrics)
```

- Train the model with fit method:
```
from torch.utils.data import DataLoader

# get datasets
training_dataset: DataLoader = ...
val_dataset: DataLoader = ...

# train with fit method
manager.fit(training_dataset, epochs=10, val_dataset=val_dataset)
```

- There are also some Keras-like callbacks to use:
```
...

tensorboard_callback = torchmanager.callbacks.TensorBoard('logs')
last_ckpt_callback = torchmanager.callbacks.Checkpoint(model, 'last.model')
manager.fit(..., callbacks_list=[tensorboard_callback, last_ckpt_callback])
```