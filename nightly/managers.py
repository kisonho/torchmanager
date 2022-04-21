import copy

from .train import configs
from torchmanager import callbacks
from torchmanager.core import os, torch
from torchmanager.core.typing import Iterable, Generic, Module, Tuple
from torchmanager.managers import * # type: ignore

class NightlyManager(Manager, Generic[Module]):
    model: Module

    def debug(self, input_shape: torch.Size, label_shapes: Iterable[torch.Size], dtype: torch.dtype=torch.float) -> Optional[Tuple[torch.Size, ...]]:
        """
        The debug function to debug model with dummy data

        - Parameters:
            input_shspe: A `torch.Size` of input dummy data shape
            dtype: A `torch.dtype` of the input dummy data
        - Returns: An optional `tuple` of `torch.Size` of multi output dummy data shape
        """
        assert self.__compiled is True, "[Debug Error]: Manager has not yet been compiled. Either loss_fn or optimizer, or both, are not given."
        assert self.optimizer is not None, "[Debug Error]: Manager has not yet been compiled. Either loss_fn or optimizer, or both, are not given."

        # initialize dummy inputs
        inputs = torch.randn(input_shape, dtype=dtype)
        labels = tuple([torch.randn(label_shape) for label_shape in label_shapes])
        label = labels[0] if len(labels) == 1 else labels
        raw_model = copy.deepcopy(self.model)

        # parsing to model
        try:
            self.train_step(inputs, label)
            y = self.model(inputs)
            self.model.load_state_dict(raw_model.state_dict())
        except Exception as error:
            raise RuntimeError(f"[Running Failed]: Data with input shape {input_shape} and label shape {label_shapes} does not fit with the given model.") from error

        # collect output shapes
        if isinstance(y, torch.Tensor):
            return tuple([y.shape])
        elif isinstance(y, Iterable):
            y: Iterable[torch.Tensor]
            output_shapes = [output.shape for output in y]
            return tuple(output_shapes)
        else: return None

    def fit_by_config(self, training_dataset: Any, config: configs.TrainingConfig, **kwargs: Any) -> torch.nn.Module:
        """
        Train model with configurations

        - Parameters:
            - training_dataset: `Any` kind of training dataset that performs `SizedIterable` protocol
            - config: A `config.TrainingConfig` of the training configurations
        - Returns: A trained `torch.nn.Module`
        """
        # initialize callbacks
        callbacks_list: List[callbacks.Callback] = []

        # check experiments
        if config.experiment is not None:
            # initialize directory
            experiment_dir = os.path.join("experiments", config.experiment)
            best_ckpt_dir = os.path.join(experiment_dir, "best.model")
            data_dir = os.path.join(experiment_dir, "data")
            last_ckpt_dir = os.path.join(experiment_dir, "last.model")
            os.makedirs(experiment_dir, exist_ok=True)

            # add callbacks
            tensorboard_callback = callbacks.TensorBoard(data_dir)
            last_ckpt_callback = callbacks.LastCheckpoint(self.model, last_ckpt_dir)
            callbacks_list.extend([tensorboard_callback, last_ckpt_callback])
            if config.monitor is not None:
                best_ckpt_callback = callbacks.BestCheckpoint(config.monitor, self.model, best_ckpt_dir)
                callbacks_list.append(best_ckpt_callback)

        # initialize learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.compiled_optimizer, step_size=config.lr_decay_step, gamma=config.lr_decay) if config.lr_decay > 0 else config.default_lr_scheduler

        # train model
        return self.fit(training_dataset, config.epochs, initial_epoch=config.initial_epoch, lr_scheduler=lr_scheduler, show_verbose=config.show_verbose, use_multi_gpus=config.use_multi_gpus, callbacks_list=callbacks_list, **kwargs)

def clone(model: torch.nn.Module, clone_fn: Callable[[str, torch.nn.Module], torch.nn.Module]):
    """
    Clone a given `torch.nn.Module` and modifies the inside module by given function

    - Parameters:
        - model: A target `torch.nn.Module` to clone
        - clone_fn: A function that accept an `str` of layer name and a `torch.nn.Module` of the layer while returns a `torch.nn.Module` of replaced layer
    - Returns: A copied and modified `torch.nn.Module`
    """
    # copy model
    copied_model = copy.deepcopy(model)

    # loop for each module inside the copied model
    for name, m in copied_model.named_modules():
        m = clone_fn(name, m)
        setattr(copied_model, name, m)
    return copied_model