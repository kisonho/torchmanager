from .train import configs
from torchmanager import callbacks
from torchmanager.core import os, torch
from torchmanager.core.typing import Iterable
from torchmanager.managers import * # type: ignore

class NightlyManager(Manager):
    def debug(self, input_shape: torch.Size, dtype: torch.dtype=torch.float) -> Optional[tuple[torch.Size, ...]]:
        """
        The debug function to debug model with dummy data

        - Parameters:
            input_shspe: A `torch.Size` of input dummy data shape
            dtype: A `torch.dtype` of the input dummy data
        - Returns: An optional `tuple` of `torch.Size` of multi output dummy data shape
        """
        # initialize dummy inputs
        inputs = torch.randn(input_shape, dtype=dtype)

        # no gradients
        with torch.no_grad():
            # parsing to model
            try: outputs = self.model(inputs)
            except Exception as error:
                raise RuntimeError(f"[Running Failed]: Data with input shape {input_shape} does not fit with the given model.") from error

            # collect output shapes
            if isinstance(outputs, torch.Tensor):
                return tuple([outputs.shape])
            elif isinstance(outputs, Iterable):
                output_shapes: List[torch.Size] = [output.shape for output in outputs]
                return tuple(output_shapes)
            else: return None

    def fit_by_config(self, training_dataset: data.DataLoader, config: configs.TrainingConfig, val_dataset: Optional[data.DataLoader] = None, **kwargs: Any) -> torch.nn.Module:
        """
        Train model with configurations

        - Parameters:
            training_dataset: A `data.DataLoader` of training dataset
            config: A `config.TrainingConfig` of the training configurations
            val_dataset: An optional `data.DataLoader` of validation dataset
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
        return self.fit(training_dataset, config.epochs, initial_epoch=config.initial_epoch, lr_scheduler=lr_scheduler, show_verbose=config.show_verbose, val_dataset=val_dataset, use_multi_gpus=config.use_multi_gpus, callbacks_list=callbacks_list, **kwargs)
