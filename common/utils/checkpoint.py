import json
import random
import shutil
import time
import types
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, RandomSampler, Subset

STATEFUL_TYPES = (nn.Module, Optimizer, LRScheduler)


@dataclass
class CheckpointState:
    model: nn.Module
    optimizer: Optimizer | None = None
    epoch: int = 0
    step: int = 0
    batch_idx: int = 0
    accumulators: dict[str, float] = field(default_factory=dict)
    scheduler: LRScheduler | None = None
    best_metric: float | None = None
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    vocab: dict[str, Any] = field(default_factory=dict)

    def step_batch(self, **updates: float) -> None:
        self.step += 1
        self.batch_idx += 1
        for k, v in updates.items():
            self.accumulators[k] = self.accumulators.get(k, 0.0) + float(v)

    def finish_epoch(self, train_loss: float, val_loss: float | None = None) -> None:
        self.epoch += 1
        self.batch_idx = 0
        self.accumulators.clear()
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)


class Checkpointer:
    def __init__(
        self,
        experiment_name: str,
        mode: Literal["min", "max"] = "min",
        save_rng: bool = True,
        save_every_steps: int | None = None,
        base_dir: str | Path | None = None,
    ) -> None:
        assert experiment_name.strip() != ""
        assert mode in ["min", "max"]
        assert save_every_steps is None or save_every_steps > 0
        self.experiment_name = experiment_name
        self.mode = mode
        self.save_rng = save_rng
        self.save_every_steps = save_every_steps
        self.base_dir = self._resolve_run_dir(
            experiment_name=experiment_name, base_dir=base_dir
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_run_dir(
        self, experiment_name: str, base_dir: str | Path | None = None
    ) -> Path:
        if base_dir:
            root = Path(base_dir)
            if (
                str(root).startswith("/content/drive")
                and not Path("/content/drive/MyDrive").is_dir()
            ):
                raise RuntimeError("Mount Google Drive First")
            return root / experiment_name
        elif Path("/content").is_dir():
            drive_root = Path("/content/drive/MyDrive")
            if not drive_root.is_dir():
                raise RuntimeError("Mount Google Drive First")
            return drive_root / "genesis-lab" / experiment_name
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / "outputs" / experiment_name

    def _is_stateful_type(self, tp: Any) -> bool:
        if isinstance(tp, type) and issubclass(tp, STATEFUL_TYPES):
            return True
        if get_origin(tp) in (Union, types.UnionType):
            return any(self._is_stateful_type(arg) for arg in get_args(tp))
        return False

    def _is_better(self, metric: float, best_metric: float | None) -> bool:
        if best_metric is None:
            return True
        if self.mode == "min":
            return metric < best_metric
        return metric > best_metric

    def log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_path = self.base_dir / "run.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")

    def _write_status(self, state: CheckpointState, is_best: bool) -> None:
        status = {
            "experiment_name": self.experiment_name,
            "epoch": state.epoch,
            "step": state.step,
            "batch_idx": state.batch_idx,
            "best_metric": state.best_metric,
            "is_best": is_best,
            "last_train_loss": state.train_losses[-1] if state.train_losses else None,
            "last_val_loss": state.val_losses[-1] if state.val_losses else None,
            "accumulators": dict(state.accumulators),
            "config": state.config,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        status_path = self.base_dir / "status.json"
        status_tmp = status_path.with_suffix(".tmp")
        status_tmp.write_text(
            json.dumps(status, indent=2, default=str) + "\n", encoding="utf-8"
        )
        status_tmp.replace(status_path)

    def save(self, state: CheckpointState, metric: float | None = None) -> Path:
        is_best = False
        if metric is not None and self._is_better(metric, state.best_metric):
            state.best_metric = metric
            is_best = True

        type_hints = get_type_hints(type(state))
        payload: dict[str, Any] = {}
        for f in fields(state):
            val = getattr(state, f.name)
            tp = type_hints.get(f.name, f.type)
            if self._is_stateful_type(tp):
                payload[f.name] = val.state_dict() if val is not None else None
            else:
                payload[f.name] = val

        if self.save_rng:
            rng_state: dict[str, Any] = {
                "torch": torch.get_rng_state(),
                "python": random.getstate(),
                "numpy": np.random.get_state(),
            }
            if torch.cuda.is_available():
                rng_state["cuda"] = torch.cuda.get_rng_state_all()
            payload["_rng_state"] = rng_state

        latest_path = self.base_dir / "latest.pt"
        tmp_path = latest_path.with_suffix(".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(latest_path)

        if is_best:
            best_path = self.base_dir / "best.pt"
            best_tmp = best_path.with_suffix(".tmp")
            shutil.copy2(latest_path, best_tmp)
            best_tmp.replace(best_path)

        self._write_status(state, is_best=is_best)
        last_tr = f"{state.train_losses[-1]:.6f}" if state.train_losses else "n/a"
        last_val = f"{state.val_losses[-1]:.6f}" if state.val_losses else "n/a"
        best_str = (
            f"{state.best_metric:.6f}" if state.best_metric is not None else "n/a"
        )
        self.log(
            f"saved latest.pt (is_best={is_best}) | epoch={state.epoch} "
            f"batch={state.batch_idx} step={state.step} | "
            f"train_loss={last_tr} val_loss={last_val} best={best_str}"
        )

        return latest_path

    def step(
        self,
        state: CheckpointState,
        **updates: float,
    ) -> Path | None:
        if updates:
            state.step_batch(**updates)
        if (
            self.save_every_steps is not None
            and state.step > 0
            and state.step % self.save_every_steps == 0
        ):
            return self.save(state)
        return None

    def end_epoch(
        self,
        state: CheckpointState,
        train_loss: float,
        val_loss: float | None = None,
        metric: float | None = None,
    ) -> Path:
        state.finish_epoch(train_loss=train_loss, val_loss=val_loss)
        return self.save(state, metric=metric)

    @staticmethod
    def iter_epoch(
        dataloader: DataLoader[Any],
        state: CheckpointState,
        seed: int = 42,
    ) -> DataLoader[Any]:
        dataset: Any = dataloader.dataset
        batch_size = dataloader.batch_size or 1
        n = len(dataset)
        is_shuffled = isinstance(dataloader.sampler, RandomSampler)
        if is_shuffled:
            g = torch.Generator()
            g.manual_seed(seed + state.epoch)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))

        start_idx = state.batch_idx * batch_size
        remaining_indices = indices[start_idx:]
        subset = Subset(dataset, remaining_indices)
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataloader.collate_fn,
            drop_last=dataloader.drop_last,
        )

    def load(
        self,
        state: CheckpointState,
        tag: Literal["latest", "best"] = "latest",
        device: torch.device | str = "cpu",
    ) -> bool:
        assert tag in ("latest", "best"), f"Invalid tag: {tag}"
        ckpt_path = self.base_dir / f"{tag}.pt"
        if not ckpt_path.is_file():
            return False

        payload: dict[str, Any] = torch.load(
            ckpt_path, map_location=device, weights_only=False
        )
        type_hints = get_type_hints(type(state))
        for f in fields(state):
            if f.name not in payload:
                continue
            saved_val = payload[f.name]
            tp = type_hints.get(f.name, f.type)
            if self._is_stateful_type(tp):
                current_obj = getattr(state, f.name)
                if current_obj is not None and saved_val is not None:
                    current_obj.load_state_dict(saved_val)
            elif f.name in ("config", "vocab"):
                current_dict = getattr(state, f.name)
                if current_dict and current_dict != saved_val:
                    raise ValueError(
                        f"Checkpoint {f.name} mismatch on load: "
                        f"current={current_dict} vs saved={saved_val}"
                    )
                setattr(state, f.name, saved_val)
            else:
                setattr(state, f.name, saved_val)

        if self.save_rng and "_rng_state" in payload:
            rng_state = payload["_rng_state"]
            if "torch" in rng_state:
                torch.set_rng_state(rng_state["torch"].cpu().to(torch.uint8))
            if "python" in rng_state:
                random.setstate(rng_state["python"])
            if "numpy" in rng_state:
                np.random.set_state(rng_state["numpy"])
            if "cuda" in rng_state and torch.cuda.is_available():
                cuda_states = [s.cpu().to(torch.uint8) for s in rng_state["cuda"]]
                torch.cuda.set_rng_state_all(cuda_states)

        best_str = (
            f"{state.best_metric:.6f}" if state.best_metric is not None else "n/a"
        )
        self.log(
            f"resumed from {tag}.pt | epoch={state.epoch} "
            f"batch={state.batch_idx} step={state.step} best={best_str}"
        )
        return True
