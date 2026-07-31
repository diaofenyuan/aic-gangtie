"""预测模型及工厂。"""

from gas_power.models.base import ForecastModel, OptionalDependencyError
from gas_power.models.factory import build_model

__all__ = ["ForecastModel", "OptionalDependencyError", "build_model"]

