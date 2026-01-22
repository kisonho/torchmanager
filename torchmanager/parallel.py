import os
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmanager_core import devices, errors, math, torch, view
from torchmanager_core.typing import Any, Collection, Module, cast

from .callbacks import Callback, ProgressBar
from .data import Dataset
from .training import Manager as BaseTrainingManager


class DistributedTrainingManager(BaseTrainingManager[Module]):
    """
    A distributed training manager, used for distributed training

    * extends: `Manager`

    - Properties:
        - backend: A `str` of the backend for distributed training
        - model: A `torch.nn.Module` or `torch.nn.parallel.DistributedDataParallel` model to be trained
        - init_method: A `str` of the initialization method for process group
    """
    backend: str
    init_method: str

    def __init__(self, model: Module, optimizer: torch.optim.Optimizer | None = None, loss_fn: Any | None = None, metrics: dict[str, Any] = {}, *, backend: str = "nccl", init_method: str = "env://") -> None:
        super().__init__(model, optimizer, loss_fn, metrics)
        self.backend = backend
        self.init_method = init_method

    def _ensure_process_group(self, world_size: int | None, rank: int | None, backend: str | None, init_method: str | None) -> tuple[int, int, bool]:
        if not dist.is_available():
            return (1 if world_size is None else world_size, 0 if rank is None else rank, False)
        if dist.is_initialized():
            return dist.get_world_size(), dist.get_rank(), False

        backend = backend or self.backend
        init_method = init_method or self.init_method
        if world_size is None:
            world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if world_size < 1:
            world_size = 1
        if rank is None:
            rank = int(os.environ.get("RANK", 0))
        if world_size <= 1:
            return world_size, rank, False
        dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)
        return world_size, rank, True

    def _cleanup_process_group(self, created: bool) -> None:
        if created and dist.is_initialized():
            dist.destroy_process_group()

    def _resolve_device(self, device: torch.device | list[torch.device] | None, rank: int) -> torch.device:
        if isinstance(device, list):
            assert len(device) > 0, errors._raise(ValueError("Device list must not be empty."))
            return device[rank % len(device)]
        if device is None:
            _, _, targets = devices.search()
            return targets[rank % len(targets)]
        return device

    def _prepare_dataset(self, dataset: Any, world_size: int, rank: int, *, is_training: bool) -> tuple[Any, DistributedSampler | None]:
        if dataset is None:
            return None, None

        if isinstance(dataset, DataLoader):
            sampler = dataset.sampler if isinstance(dataset.sampler, DistributedSampler) else DistributedSampler(dataset.dataset, num_replicas=world_size, rank=rank, shuffle=is_training)
            persistent = getattr(dataset, "persistent_workers", False) if dataset.num_workers > 0 else False
            loader = DataLoader(dataset.dataset, batch_size=dataset.batch_size, sampler=sampler, shuffle=False, num_workers=dataset.num_workers, pin_memory=dataset.pin_memory, pin_memory_device=getattr(dataset, "pin_memory_device", ""), drop_last=dataset.drop_last, collate_fn=dataset.collate_fn, worker_init_fn=getattr(dataset, "worker_init_fn", None), persistent_workers=persistent)
            return loader, sampler

        if isinstance(dataset, Dataset):
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=getattr(dataset, "shuffle", is_training))
            dataset.sampler = sampler
            return dataset, sampler

        return dataset, None

    def data_parallel(self, target_devices: list[torch.device]) -> bool:
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return True
        if not dist.is_available() or not dist.is_initialized():
            self.to(target_devices[0])
            return False
        device_ids = [d.index for d in target_devices]
        self.model = cast(torch.nn.parallel.DataParallel[Module], torch.nn.parallel.DistributedDataParallel(self.raw_model, device_ids=device_ids, output_device=target_devices))
        if self.loss_fn is not None:
            self.loss_fn = self.loss_fn.to(target_devices[0])
        for k, m in self.metric_fns.items():
            self.metric_fns[k] = m.to(target_devices[0])
        return True

    def _distributed_test(self, dataset: Dataset | DataLoader, sampler: DistributedSampler | None, device: torch.device, rank: int, *, show_verbose: bool) -> dict[str, float]:
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(self.current_epoch)

        if len(dataset) == 0:
            return {}
        if isinstance(dataset, Dataset):
            dataset_len = dataset.batched_len
        else:
            dataset_len = len(dataset)

        progress_bar = view.tqdm(total=dataset_len, disable=not show_verbose or rank != 0)
        self.reset_metrics()

        for data in dataset:
            x_test, y_test = self.unpack_data(data)
            x_test = devices.move_to_device(x_test, device)
            y_test = devices.move_to_device(y_test, device)

            step_summary = self.test_step(x_test, y_test)
            if rank == 0:
                progress_bar.set_postfix(step_summary)
                progress_bar.update()

        progress_bar.close()
        return self.summary

    def fit(self, training_dataset: DataLoader[Any] | Dataset[Any], /, epochs: int | None = None, val_dataset: DataLoader[Any] | Dataset[Any] | None = None, callbacks_list: list[Callback] = [], *args, iterations: int | None = None, initial_epoch: int | None = None, return_summary: bool = False, device: torch.device | list[torch.device] | None = None, show_verbose: bool = False, verbose_type: view.VerboseType = view.VerboseType.ALL, backend: str | None = None, init_method: str | None = None, world_size: int | None = None, rank: int | None = None, **kwargs) -> Module | tuple[Module, dict[str, float]]:
        dataset_len = training_dataset.batched_len if isinstance(training_dataset, Dataset) else len(training_dataset)
        assert self.compiled is True, errors._raise(ValueError("Manager has not yet been compiled. Either loss_fn or optimizer, or both, are not given."))

        if epochs is not None:
            assert epochs > 0, errors._raise(ValueError(f"The epochs must be a positive integer, got {epochs}."))
            assert iterations is None, errors._raise(ValueError(f"The iterations must be given as `None` when epochs is given, got {iterations}."))
        else:
            assert iterations is not None, errors._raise(ValueError(f"The iterations must be given if epochs is not given."))
            assert iterations > 0, errors._raise(ValueError(f"The iterations must be a positive integer, got {iterations}."))
            assert epochs is None, errors._raise(ValueError(f"The epochs must be given as `None` when iterations is given, got {epochs}."))
            epochs = math.ceil(iterations / dataset_len)

        if initial_epoch is not None:
            assert initial_epoch >= 0, errors._raise(ValueError(f"The initial_epoch must be a non_negative integer, got {initial_epoch}."))
            assert initial_epoch < epochs, errors._raise(ValueError(f"The initial_epoch must be smaller than total epochs, got epochs={epochs} but initial_epoch={initial_epoch}."))
            self.current_epoch = initial_epoch
        elif self.current_epoch > 0:
            initial_epoch = self.current_epoch + 1
        else:
            initial_epoch = self.current_epoch

        world_size, rank, created_pg = self._ensure_process_group(world_size, rank, backend, init_method)
        device = self._resolve_device(device, rank)

        training_dataset, train_sampler = self._prepare_dataset(training_dataset, world_size, rank, is_training=True)
        val_dataset, val_sampler = self._prepare_dataset(val_dataset, world_size, rank, is_training=False) if val_dataset is not None else (None, None)
        dataset_len = training_dataset.batched_len if isinstance(training_dataset, Dataset) else len(training_dataset)

        if show_verbose and rank == 0:
            callbacks_list.append(ProgressBar(dataset_len, verbose_type=verbose_type))

        summary: dict[str, float] = {}
        cpu = devices.CPU

        try:
            self.data_parallel([device])

            for callback in callbacks_list:
                callback.on_train_start(initial_epoch)

            for self.current_epoch in range(initial_epoch, epochs):
                if isinstance(train_sampler, DistributedSampler):
                    train_sampler.set_epoch(self.current_epoch)
                if isinstance(val_sampler, DistributedSampler):
                    val_sampler.set_epoch(self.current_epoch)

                if iterations is not None:
                    batch_iterations = iterations if dataset_len < iterations else iterations
                else:
                    batch_iterations = None

                iterations_per_epoch = dataset_len if batch_iterations is None else batch_iterations
                for callback in callbacks_list:
                    if isinstance(callback, ProgressBar) and iterations_per_epoch != callback.iterations_per_epoch:
                        callback.iterations_per_epoch = iterations_per_epoch
                    callback.on_epoch_start(self.current_epoch)

                with self:
                    training_summary = self._train(training_dataset, iterations=batch_iterations, *args, device=device, use_multi_gpus=False, callbacks_list=callbacks_list, **kwargs)

                summary |= training_summary
                if iterations is not None and batch_iterations is not None:
                    iterations -= batch_iterations

                val_summary = self._distributed_test(val_dataset, val_sampler, device, rank, show_verbose=show_verbose) if val_dataset is not None else None

                for callback in callbacks_list:
                    callback.on_epoch_end(self.current_epoch, summary=training_summary, val_summary=val_summary)

                if rank == 0:
                    val_message = f"Epoch {self.current_epoch + 1}/{epochs}: "
                    if val_summary is not None:
                        summary |= {f"val_{name}": value for name, value in val_summary.items()}
                    for i, (name, value) in enumerate(summary.items()):
                        if i > 0:
                            val_message += ", "
                        val_message += f"{name}={value:.4f}"
                    view.logger.info(val_message)

            if rank == 0:
                view.logger.info("Training finished.")
        except errors.StopTraining:
            if rank == 0:
                view.logger.info("Training finished.")
        except KeyboardInterrupt:
            if rank == 0:
                view.logger.info("Training interrupted.")
        except Exception as error:
            view.logger.error(error)
            runtime_error = errors.StopTraining(self.current_epoch, "Training failed.")
            raise runtime_error from error
        finally:
            for callback in callbacks_list:
                callback.on_train_end(self.raw_model)

            if show_verbose and rank == 0:
                callbacks_list.pop()

            self.reset(cpu)
            self._cleanup_process_group(created_pg)

        return (self.raw_model, summary) if return_summary else self.raw_model

    @torch.no_grad()
    def test(self, dataset: DataLoader[Any] | Dataset[Any], /, *, device: torch.device | list[torch.device] | None = None, empty_cache: bool = True, show_verbose: bool = False, backend: str | None = None, init_method: str | None = None, world_size: int | None = None, rank: int | None = None) -> dict[str, float]:
        world_size, rank, created_pg = self._ensure_process_group(world_size, rank, backend, init_method)
        device = self._resolve_device(device, rank)
        dataset, sampler = self._prepare_dataset(dataset, world_size, rank, is_training=False)

        cpu = devices.CPU
        try:
            self.data_parallel([device])
            summary = self._distributed_test(dataset, sampler, device, rank, show_verbose=show_verbose)
            return summary
        except KeyboardInterrupt:
            if rank == 0:
                view.logger.info("Testing interrupted.")
            return {}
        except Exception as error:
            view.logger.error(error)
            runtime_error = errors.TestingError()
            raise runtime_error from error
        finally:
            if empty_cache:
                self.reset(cpu)
            self._cleanup_process_group(created_pg)
