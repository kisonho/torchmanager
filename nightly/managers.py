# import required modules
import os, torch

# import core modules
from .train import configs
from torchmanager import * # type: ignore

class NightlyManager(Manager):
    def fit_by_config(self, training_dataset: data.DataLoader, config: configs.TrainingConfig, monitor: Optional[str] = None, val_dataset: Optional[data.DataLoader] = None) -> torch.nn.Module:
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
            if monitor is not None:
                best_ckpt_callback = callbacks.BestCheckpoint(monitor, self.model, best_ckpt_dir)
                callbacks_list.append(best_ckpt_callback)

        # initialize learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(self.compiled_optimizer, step_size=config.lr_decay_step, gamma=config.lr_decay) if config.lr_decay > 0 else None

        # train model
        return self.fit(training_dataset, config.epochs, initial_epoch=config.initial_epoch, lr_scheduler=lr_scheduler, show_verbose=config.show_verbose, val_dataset=val_dataset, use_multi_gpus=config.use_multi_gpus, callbacks_list=callbacks_list)
