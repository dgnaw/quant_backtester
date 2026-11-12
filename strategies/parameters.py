from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Iterable

SUPPORTED_PARAM_TYPES = (int, float, bool, str)


class ParameterError(ValueError):
    """Ngoại lệ gom toàn bộ các lỗi validation của ParameterSchema."""

    def __init__(self, errors: list[str] | str):
        if isinstance(errors, str):
            self.errors = [errors]
        else:
            self.errors = list(errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class Parameter:
    name: str
    param_type: type
    default: Any = None
    min_value: int | float | None = None
    max_value: int | float | None = None
    choices: tuple[Any, ...] | list[Any] | None = None
    required: bool = False
    description: str = ""
    step: int | float | None = None
    optimizable: bool = False

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Tên tham số (name) phải là chuỗi không rỗng.")
        if self.param_type not in SUPPORTED_PARAM_TYPES:
            raise ValueError(f"Kiểu dữ liệu {self.param_type} không được hỗ trợ. Chỉ hỗ trợ: int, float, bool, str.")
        # Rule 7: required=True không được đồng thời có default khác None
        if self.required and self.default is not None:
            raise ValueError(f"Tham số '{self.name}': required=True không được đồng thời khai báo default.")
        if self.min_value is not None and self.max_value is not None and self.min_value > self.max_value:
            raise ValueError(f"Tham số '{self.name}': min_value không được lớn hơn max_value.")


class ParameterSchema:
    def __init__(self, parameters: Iterable[Parameter]):
        self._params: dict[str, Parameter] = {}
        for p in parameters:
            if p.name in self._params:
                raise ValueError(f"Trùng tên tham số '{p.name}' trong ParameterSchema.")
            self._params[p.name] = p

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        return tuple(self._params.values())

    def _coerce_value(self, param: Parameter, val: Any, coerce: bool) -> tuple[Any, str | None]:
        t = param.param_type

        # Rule 1: bool không được chấp nhận cho int/float
        if isinstance(val, bool) and t in (int, float):
            return None, f"Tham số '{param.name}' yêu cầu kiểu {t.__name__}, không chấp nhận bool."

        # Rule 3: str chỉ được ép kiểu khi coerce=True
        if isinstance(val, str) and t is not str:
            if not coerce:
                return None, f"Tham số '{param.name}' yêu cầu kiểu {t.__name__}, nhận str khi coerce=False."
            s_val = val.strip()
            try:
                if t is int:
                    val = int(s_val)
                elif t is float:
                    val = float(s_val)
                elif t is bool:
                    low = s_val.lower()
                    if low in ("true", "1", "yes"):
                        val = True
                    elif low in ("false", "0", "no"):
                        val = False
                    else:
                        return None, f"Tham số '{param.name}' không thể ép kiểu '{val}' sang bool."
            except ValueError:
                return None, f"Tham số '{param.name}' không thể ép kiểu '{val}' sang {t.__name__}."

        # Rule 2: int có thể chuyển thành float khi param_type là float
        if t is float and isinstance(val, int) and not isinstance(val, bool):
            val = float(val)

        if t is int and (not isinstance(val, int) or isinstance(val, bool)):
            return None, f"Tham số '{param.name}' phải có kiểu int, nhận {type(val).__name__}."
        if t is float and (not isinstance(val, float) or isinstance(val, bool)):
            return None, f"Tham số '{param.name}' phải có kiểu float, nhận {type(val).__name__}."
        if t is bool and not isinstance(val, bool):
            return None, f"Tham số '{param.name}' phải có kiểu bool, nhận {type(val).__name__}."
        if t is str and not isinstance(val, str):
            return None, f"Tham số '{param.name}' phải có kiểu str, nhận {type(val).__name__}."

        # Rule 4: NaN và Inf bị từ chối
        if t in (int, float) and not math.isfinite(float(val)):
            return None, f"Tham số '{param.name}' không chấp nhận NaN hoặc Inf."

        # Rule 5: min_value và max_value là inclusive
        if param.min_value is not None and val < param.min_value:
            return None, f"Tham số '{param.name}' ({val}) nhỏ hơn min_value ({param.min_value})."
        if param.max_value is not None and val > param.max_value:
            return None, f"Tham số '{param.name}' ({val}) lớn hơn max_value ({param.max_value})."

        # Rule 6: choices được kiểm tra sau khi chuẩn hóa kiểu dữ liệu
        if param.choices is not None and val not in param.choices:
            return None, f"Tham số '{param.name}' ({val!r}) nằm ngoài choices hợp lệ {tuple(param.choices)}."

        return val, None

    def validate(self, raw_params: dict[str, Any] | None = None, coerce: bool = False) -> dict[str, Any]:
        given = dict(raw_params) if raw_params else {}
        errors: list[str] = []
        validated: dict[str, Any] = {}

        # Rule 9: Tham số không khai báo phải báo lỗi
        for key in given:
            if key not in self._params:
                errors.append(f"Tham số '{key}' không được khai báo trong schema.")

        for name, param in self._params.items():
            if name in given and given[name] is not None:
                val, err = self._coerce_value(param, given[name], coerce=coerce)
                if err:
                    errors.append(err)
                else:
                    validated[name] = val
            else:
                # Rule 8: Thiếu tham số bắt buộc phải báo lỗi
                if param.required:
                    errors.append(f"Thiếu tham số bắt buộc '{name}'.")
                elif param.default is not None:
                    val, err = self._coerce_value(param, param.default, coerce=coerce)
                    if err:
                        errors.append(err)
                    else:
                        validated[name] = val
                else:
                    validated[name] = None

        # Rule 10: Mọi vi phạm được gom vào ParameterError
        if errors:
            raise ParameterError(errors)

        return validated

