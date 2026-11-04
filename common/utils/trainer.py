import statistics
import time
from collections.abc import Callable
from typing import Any, Literal

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from common.utils.checkpoint import Checkpointer, CheckpointState

BatchLossFn = Callable[
    [nn.Module, Any, torch.device],
    torch.Tensor | tuple[torch.Tensor, float],
]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        batch_loss_fn: BatchLossFn,
        scheduler: LRScheduler | None = None,
        scheduler_step_on: Literal["step", "epoch"] = "step",
        max_grad_norm: float | None = None,
        device: torch.device | str | None = None,
        experiment_name: str | None = None,
        save_every_steps: int | None = None,
        run_config: dict[str, Any] | None = None,
        vocab: dict[str, Any] | None = None,
        seed: int = 42,
        profile: bool = False,
        profile_warmup_steps: int = 20,
        profile_log_every: int = 50,
    ) -> None:
        assert scheduler_step_on in ("step", "epoch")
        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.batch_loss_fn = batch_loss_fn
        self.scheduler = scheduler
        self.scheduler_step_on = scheduler_step_on
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        # profile=True: per-step wall time (GPU synced) and peak memory, so a
        # short run can size a long one. Syncing costs a little speed, so off
        # by default. The first profile_warmup_steps of each epoch are left out
        # of the stats (CUDA init, cuDNN autotuning).
        self.profile = profile
        self.profile_warmup_steps = profile_warmup_steps
        self.profile_log_every = profile_log_every
        self.profile_stats: list[dict[str, float]] = []

        self.checkpoint_state = CheckpointState(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=run_config or {},
            vocab=vocab or {},
        )
        self.checkpoint = (
            Checkpointer(
                experiment_name=experiment_name,
                mode="min",
                save_every_steps=save_every_steps,
            )
            if experiment_name
            else None
        )

    def _log(self, message: str) -> None:
        print(message)
        if self.checkpoint:
            self.checkpoint.log(message)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def _peak_memory_gb(self) -> float | None:
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device) / 1e9
        if self.device.type == "mps":
            # MPS has no peak counter; this is memory held right now
            return torch.mps.driver_allocated_memory() / 1e9
        return None

    def _profile_line(self, step_times: list[float], batch_size: int | None) -> str:
        timed = step_times[self.profile_warmup_steps :]
        if not timed:
            return f"profile: {len(step_times)} steps, all warm-up"
        median = statistics.median(timed)
        p90 = statistics.quantiles(timed, n=10)[-1] if len(timed) >= 2 else median
        line = (
            f"profile: {len(timed)} timed steps, median {median:.3f} s/step, "
            f"p90 {p90:.3f} s/step"
        )
        if batch_size:
            line += f", {batch_size / median:.1f} examples/s"
        peak = self._peak_memory_gb()
        if peak is not None:
            line += f", peak mem {peak:.2f} GB"
        return line

    def _run_batch(self, batch: Any) -> tuple[torch.Tensor, float]:
        out = self.batch_loss_fn(self.model, batch, self.device)
        if isinstance(out, tuple):
            loss, weight = out
        else:
            loss, weight = out, 1.0
        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Non-finite loss ({loss.item()}) at epoch {self.checkpoint_state.epoch}, "
                f"batch {self.checkpoint_state.batch_idx}, step {self.checkpoint_state.step}"
            )
        return loss, float(weight)

    def evaluate(self, val_loader: DataLoader[Any]) -> float:
        self.model.eval()
        total_loss = 0.0
        total_weight = 0.0
        with torch.no_grad():
            for batch in val_loader:
                loss, weight = self._run_batch(batch)
                total_loss += loss.item() * weight
                total_weight += weight
        return total_loss / total_weight

    def fit(
        self,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any] | None,
        epochs: int,
    ) -> tuple[list[float], list[float]]:
        state = self.checkpoint_state
        if self.checkpoint and self.checkpoint.load(state=state, device=self.device):
            print(
                f"Resumed at epoch {state.epoch}, batch {state.batch_idx}, "
                f"step {state.step} (best val_loss: {state.best_metric})"
            )

        for epoch in range(state.epoch, epochs):
            self.model.train()
            epoch_loader = Checkpointer.iter_epoch(train_loader, state, seed=self.seed)
            step_times: list[float] = []
            epoch_start = step_start = time.perf_counter()
            if self.profile:
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                self._sync()
                epoch_start = time.perf_counter()
                step_start = epoch_start
            for batch in epoch_loader:
                loss, weight = self._run_batch(batch)
                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_grad_norm
                    )
                self.optimizer.step()
                if self.scheduler is not None and self.scheduler_step_on == "step":
                    self.scheduler.step()

                state.step_batch(total_loss=loss.item() * weight, total_weight=weight)
                if self.checkpoint:
                    self.checkpoint.step(state)

                if self.profile:
                    # wall time from the end of the previous step: data
                    # loading, forward, backward, optimizer and checkpoint I/O
                    self._sync()
                    now = time.perf_counter()
                    step_times.append(now - step_start)
                    step_start = now
                    n = len(step_times)
                    if (
                        n > self.profile_warmup_steps
                        and (n - self.profile_warmup_steps) % self.profile_log_every
                        == 0
                    ):
                        self._log(
                            f"epoch {epoch} step {n}: "
                            + self._profile_line(step_times, train_loader.batch_size)
                        )

            if self.profile:
                train_seconds = time.perf_counter() - epoch_start
                timed = step_times[self.profile_warmup_steps :]
                self.profile_stats.append(
                    {
                        "epoch": float(epoch),
                        "steps": float(len(step_times)),
                        "train_seconds": train_seconds,
                        "median_step_seconds": (
                            statistics.median(timed) if timed else float("nan")
                        ),
                        "peak_memory_gb": self._peak_memory_gb() or float("nan"),
                    }
                )
                self._log(
                    f"epoch {epoch} done: {len(step_times)} steps in {train_seconds:.1f} s; "
                    + self._profile_line(step_times, train_loader.batch_size)
                )

            if self.scheduler is not None and self.scheduler_step_on == "epoch":
                self.scheduler.step()

            avg_train_loss = (
                state.accumulators["total_loss"] / state.accumulators["total_weight"]
            )
            val_start = time.perf_counter()
            val_loss = self.evaluate(val_loader) if val_loader is not None else None
            if self.profile and val_loader is not None:
                self._sync()
                val_seconds = time.perf_counter() - val_start
                self.profile_stats[-1]["val_seconds"] = val_seconds
                self._log(f"epoch {epoch} validation: {val_seconds:.1f} s")
            metric = val_loss if val_loss is not None else avg_train_loss
            state.finish_epoch(train_loss=avg_train_loss, val_loss=val_loss)
            if self.checkpoint:
                self.checkpoint.save(state=state, metric=metric)

            if val_loss is not None:
                print(
                    f"Epoch {epoch}: train loss {avg_train_loss:.4f}, "
                    f"avg validation loss {val_loss:.4f}"
                )
            else:
                print(f"Epoch {epoch}: train loss {avg_train_loss:.4f}")

        return state.train_losses, state.val_losses
