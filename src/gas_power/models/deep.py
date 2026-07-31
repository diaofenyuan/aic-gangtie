"""从零训练的 TCN/PatchTST 残差模型，依赖按需加载。"""

from __future__ import annotations

import random
from typing import Any, Sequence

import numpy as np
import pandas as pd

from gas_power.models.base import (
    FitProgressCallback,
    ForecastModel,
    OptionalDependencyError,
    prediction_column,
    validate_prediction_request,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


class NeuralResidualMultiHorizonModel(ForecastModel):
    """以最后值为基线，使用时间上下文学习多目标残差。"""

    def __init__(
        self,
        architecture: str,
        feature_builder: Any,
        context_steps: int = 192,
        epochs: int = 200,
        patience: int = 20,
        batch_size: int = 128,
        learning_rate: float = 1.0e-3,
        hidden_size: int = 64,
        patch_size: int = 16,
        dropout: float = 0.1,
        seeds: Sequence[int] | None = None,
        device: str = "auto",
        interval_minutes: int = 15,
    ):
        if architecture not in {"tcn", "patchtst"}:
            raise ValueError("深度模型架构必须是 tcn 或 patchtst")
        if context_steps <= 0 or epochs <= 0 or patience <= 0:
            raise ValueError("深度模型上下文和训练轮数必须大于 0")
        self.architecture = architecture
        self.feature_builder = feature_builder
        self.context_steps = int(context_steps)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.hidden_size = int(hidden_size)
        self.patch_size = int(patch_size)
        self.dropout = float(dropout)
        self.seeds = [int(seed) for seed in (seeds or [2026, 2027, 2028])]
        self.device_name = str(device)
        self.interval_minutes = int(interval_minutes)
        self.models_: list[Any] = []
        self.scaler_: Any = None
        self.feature_columns_: list[str] = []
        self.target_columns_: list[str] = []
        self.horizons_: list[int] = []
        self.train_metadata_: dict[str, object] = {}
        self._fit_progress_callback: FitProgressCallback | None = None

    def fit_progress_steps(
        self,
        target_columns: Sequence[str],
        horizons: Sequence[int],
    ) -> int:
        return len(self.seeds) * self.epochs

    def set_fit_progress_callback(
        self,
        callback: FitProgressCallback | None,
    ) -> None:
        self._fit_progress_callback = callback

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_fit_progress_callback"] = None
        state["model_states_"] = [
            {key: value.detach().cpu() for key, value in model.state_dict().items()}
            for model in self.models_
        ]
        state["models_"] = []
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        model_states = state.pop("model_states_", [])
        self.__dict__.update(state)
        self.models_ = []
        if model_states:
            output_size = len(self.target_columns_) * len(self.horizons_)
            for model_state in model_states:
                model = self._network(len(self.feature_columns_), output_size)
                model.load_state_dict(model_state)
                model.eval()
                self.models_.append(model)

    @staticmethod
    def _torch():
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise OptionalDependencyError(
                "已选择 PyTorch 深度模型，但当前环境未安装；请安装 high-accuracy 可选依赖。"
            ) from exc
        return torch, nn

    def _network(self, input_features: int, output_size: int):
        _, nn = self._torch()
        hidden_size = self.hidden_size
        dropout = self.dropout
        patch_size = self.patch_size
        if self.architecture == "tcn":
            class TCN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Conv1d(input_features, hidden_size, 3, padding=1, dilation=1),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Conv1d(hidden_size, hidden_size, 3, padding=2, dilation=2),
                        nn.GELU(),
                        nn.Conv1d(hidden_size, hidden_size, 3, padding=4, dilation=4),
                        nn.GELU(),
                    )
                    self.head = nn.Linear(hidden_size, output_size)

                def forward(self, x):
                    return self.head(self.net(x.transpose(1, 2))[:, :, -1])

            return TCN()

        class PatchTST(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = nn.Linear(patch_size * input_features, hidden_size)
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=4 if hidden_size % 4 == 0 else 2,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=False,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=2)
                self.head = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                length = x.shape[1]
                usable = (length // patch_size) * patch_size
                x = x[:, length - usable :]
                x = x.reshape(x.shape[0], usable // patch_size, patch_size * x.shape[2])
                return self.head(self.encoder(self.patch_embed(x)).mean(dim=1))

        return PatchTST()

    def _device(self, torch):
        if self.device_name == "cuda" or (
            self.device_name == "auto" and torch.cuda.is_available()
        ):
            return torch.device("cuda")
        return torch.device("cpu")

    def fit(
        self,
        frame: pd.DataFrame,
        target_columns: Sequence[str],
        horizons: Sequence[int],
        train_end: pd.Timestamp | None = None,
        *,
        raw_targets: pd.DataFrame | None = None,
        feature_matrix: pd.DataFrame | None = None,
        data_source: str = "training",
    ) -> "NeuralResidualMultiHorizonModel":
        if data_source == "scoring":
            raise ValueError("评分期 scoring 数据禁止用于深度模型拟合")
        torch, nn = self._torch()
        end = pd.Timestamp(train_end if train_end is not None else frame.index.max())
        features = (
            self.feature_builder.transform(frame)
            if feature_matrix is None
            else feature_matrix.reindex(frame.index)
        )
        self.feature_columns_ = [
            str(column) for column in features.columns if features[column].notna().any()
        ]
        self.feature_columns_ = [
            column for column in self.feature_columns_
            if np.isfinite(pd.to_numeric(features[column], errors="coerce").fillna(0.0)).all()
        ]
        if not self.feature_columns_:
            raise ValueError("深度模型没有有效特征")
        self.target_columns_ = [str(value) for value in target_columns]
        self.horizons_ = [int(value) for value in horizons]
        labels = frame[self.target_columns_] if raw_targets is None else raw_targets[self.target_columns_]
        end_position = int(frame.index.get_indexer([end])[0])
        context = self.context_steps
        max_horizon = max(self.horizons_)
        x_values = features[self.feature_columns_].fillna(0.0).to_numpy(dtype=np.float32)
        from sklearn.preprocessing import RobustScaler

        self.scaler_ = RobustScaler().fit(x_values[: end_position + 1])
        x_values = self.scaler_.transform(x_values).astype(np.float32)
        sample_positions: list[int] = []
        responses: list[np.ndarray] = []
        actuals: list[np.ndarray] = []
        last_position = end_position - max_horizon
        for position in range(context - 1, last_position + 1):
            future_positions = [position + horizon for horizon in self.horizons_]
            values = labels.iloc[future_positions][self.target_columns_].to_numpy(dtype=float)
            current = labels.iloc[position][self.target_columns_].to_numpy(dtype=float)
            if not np.isfinite(values).all() or not np.isfinite(current).all():
                continue
            sample_positions.append(position)
            responses.append((values - current).reshape(-1))
            actuals.append(np.maximum(np.abs(values.reshape(-1)), 1.0e-6))
        if len(sample_positions) < 32:
            raise ValueError("深度模型有效时间窗口不足 32 个")
        response_values = np.stack(responses).astype(np.float32)
        actual_values = np.stack(actuals).astype(np.float32)
        positive_actuals = actual_values[actual_values > 0.0]
        denominator_floor = (
            float(np.quantile(positive_actuals, 0.01))
            if positive_actuals.size
            else 1.0
        )
        weight_values = 1.0 / np.maximum(actual_values, denominator_floor)
        weight_values /= max(float(weight_values.mean()), 1.0e-6)
        position_values = np.asarray(sample_positions, dtype=np.int64)
        split = max(1, min(len(position_values) - 1, int(len(position_values) * 0.8)))

        class WindowDataset(torch.utils.data.Dataset):
            """按需切取上下文，避免把全部重叠窗口复制到内存。"""

            def __init__(self, indexes: np.ndarray):
                self.indexes = indexes

            def __len__(self) -> int:
                return len(self.indexes)

            def __getitem__(self, item: int):
                sample_index = int(self.indexes[item])
                position = int(position_values[sample_index])
                sample = np.ascontiguousarray(
                    x_values[position - context + 1 : position + 1]
                )
                return (
                    torch.from_numpy(sample),
                    torch.from_numpy(response_values[sample_index]),
                    torch.from_numpy(weight_values[sample_index]),
                )

        device = self._device(torch)
        pin_memory = device.type == "cuda"
        train_dataset = WindowDataset(np.arange(split, dtype=np.int64))
        validation_dataset = WindowDataset(
            np.arange(split, len(position_values), dtype=np.int64)
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
        self.models_ = []
        for seed in self.seeds:
            _set_seed(seed)
            net = self._network(
                len(self.feature_columns_), response_values.shape[1]
            ).to(device)
            optimizer = torch.optim.AdamW(net.parameters(), lr=self.learning_rate, weight_decay=1.0e-4)
            best_state = None
            best_loss = float("inf")
            stale = 0
            for _ in range(self.epochs):
                net.train()
                generator = torch.Generator().manual_seed(seed + _)
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=pin_memory,
                    generator=generator,
                )
                for xb, yb, wb in train_loader:
                    xb = xb.to(device, non_blocking=pin_memory)
                    yb = yb.to(device, non_blocking=pin_memory)
                    wb = wb.to(device, non_blocking=pin_memory)
                    prediction = net(xb)
                    smooth = nn.functional.smooth_l1_loss(prediction, yb, reduction="none")
                    loss = (smooth * wb).mean() + 0.1 * (
                        torch.abs(prediction - yb) * wb
                    ).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optimizer.step()
                net.eval()
                with torch.no_grad():
                    validation_loss = 0.0
                    validation_count = 0
                    for xb, yb, wb in validation_loader:
                        xb = xb.to(device, non_blocking=pin_memory)
                        yb = yb.to(device, non_blocking=pin_memory)
                        wb = wb.to(device, non_blocking=pin_memory)
                        prediction = net(xb)
                        batch_loss = (
                            nn.functional.smooth_l1_loss(
                                prediction, yb, reduction="none"
                            )
                            * wb
                        ).mean()
                        validation_loss += float(batch_loss.cpu()) * len(xb)
                        validation_count += len(xb)
                    val_loss = validation_loss / max(1, validation_count)
                if self._fit_progress_callback is not None:
                    self._fit_progress_callback(
                        f"随机种子 {seed}，轮次 {_ + 1}/{self.epochs}，损失 {val_loss:.4g}"
                    )
                if val_loss < best_loss - 1.0e-6:
                    best_loss = val_loss
                    best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                    if stale >= self.patience:
                        break
            if best_state is not None:
                net.load_state_dict(best_state)
            net.eval()
            self.models_.append(net.cpu())
        self.train_metadata_ = {
            "data_source": data_source,
            "raw_labels": True,
            "architecture": self.architecture,
            "context_steps": self.context_steps,
            "seeds": self.seeds,
            "best_model_count": len(self.models_),
        }
        return self

    def predict(
        self,
        frame: pd.DataFrame,
        origins: pd.DatetimeIndex,
        target_columns: Sequence[str],
        horizons: Sequence[int],
    ) -> pd.DataFrame:
        torch, _ = self._torch()
        validate_prediction_request(frame, origins, target_columns, horizons)
        if not self.models_ or self.scaler_ is None:
            raise RuntimeError("深度模型尚未拟合")
        features = self.feature_builder.transform(frame)
        feature_values = features[self.feature_columns_].fillna(0.0).to_numpy(dtype=np.float32)
        values = self.scaler_.transform(feature_values).astype(np.float32)
        device = self._device(torch)
        rows: list[np.ndarray] = []
        for origin in origins:
            position = int(frame.index.get_indexer([origin])[0])
            if position < self.context_steps - 1:
                raise ValueError("预测起点之前的历史不足深度模型上下文长度")
            sample = torch.tensor(
                values[position - self.context_steps + 1 : position + 1][None, ...],
                dtype=torch.float32,
            ).to(device)
            with torch.no_grad():
                prediction = torch.stack([net.to(device)(sample).cpu()[0] for net in self.models_]).mean(dim=0)
            current = pd.to_numeric(frame.loc[origin, self.target_columns_], errors="coerce").to_numpy(dtype=float)
            rows.append((prediction.numpy().reshape(len(self.horizons_), len(self.target_columns_)) + current).T.reshape(-1))
        output = pd.DataFrame(index=origins)
        output.index.name = "datetime"
        array = np.stack(rows)
        for target_index, target in enumerate(self.target_columns_):
            for horizon_index, horizon in enumerate(self.horizons_):
                output[prediction_column(target, horizon, self.interval_minutes)] = array[:, target_index * len(self.horizons_) + horizon_index]
        return output
