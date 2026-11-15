from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any
import pandas as pd

from quant_backtester.config import InstrumentConfig
from quant_backtester.models.order import ExecutionPhase, validate_and_normalize_lot
from quant_backtester.models.trade import EntryContribution, TradeAllocation


class PositionSide(Enum):
    FLAT = 0
    LONG = 1
    SHORT = -1


@dataclass(frozen=True)
class PositionView:
    symbol: str
    side: PositionSide = PositionSide.FLAT
    quantity: float = 0.0
    units: float = 0.0
    entry_price: float = 0.0
    entry_time: pd.Timestamp | None = None
    unrealized_pnl: float | None = None

    @property
    def is_flat(self) -> bool:
        return self.side == PositionSide.FLAT or abs(self.units) <= 1e-12

    @property
    def is_long(self) -> bool:
        return self.side == PositionSide.LONG and self.units > 1e-12

    @property
    def is_short(self) -> bool:
        return self.side == PositionSide.SHORT and self.units > 1e-12


_POSITION_COUNTER = 0


def _next_position_id() -> str:
    global _POSITION_COUNTER
    _POSITION_COUNTER += 1
    return f"POS-{_POSITION_COUNTER:05d}"


@dataclass
class Position:
    symbol: str
    side: PositionSide = PositionSide.FLAT
    quantity: float = 0.0
    average_entry_price: float = 0.0
    entry_time: pd.Timestamp | None = None
    entry_fee: float = 0.0
    entry_slippage_cost: float = 0.0
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    opened_bar_index: int | None = None
    opened_phase: ExecutionPhase | None = None
    instrument_config: InstrumentConfig | None = None
    position_id: str = ""
    contributions: list[EntryContribution] = field(default_factory=list)
    is_closed: bool = False

    def __post_init__(self):
        if self.instrument_config is None:
            self.instrument_config = InstrumentConfig(symbol=self.symbol)
        if not self.position_id:
            self.position_id = _next_position_id()

    @property
    def contract_size(self) -> float:
        assert self.instrument_config is not None
        return float(self.instrument_config.contract_size)

    @property
    def units(self) -> float:
        return float(self.quantity) * self.contract_size

    @units.setter
    def units(self, val: float) -> None:
        self.quantity = float(val) / self.contract_size if self.contract_size > 0 else 0.0

    @property
    def entry_price(self) -> float:
        return self.average_entry_price

    @entry_price.setter
    def entry_price(self, val: float) -> None:
        self.average_entry_price = float(val)

    @property
    def avg_entry_price(self) -> float:
        return self.average_entry_price

    @property
    def stop_loss(self) -> float | None:
        return self.stop_loss_price

    @stop_loss.setter
    def stop_loss(self, val: float | None) -> None:
        self.stop_loss_price = float(val) if val is not None else None

    @property
    def take_profit(self) -> float | None:
        return self.take_profit_price

    @take_profit.setter
    def take_profit(self, val: float | None) -> None:
        self.take_profit_price = float(val) if val is not None else None

    @property
    def is_flat(self) -> bool:
        return self.is_closed or self.side == PositionSide.FLAT or abs(self.quantity) < 1e-12

    @property
    def is_long(self) -> bool:
        return not self.is_closed and self.side == PositionSide.LONG and self.quantity > 1e-12

    @property
    def is_short(self) -> bool:
        return not self.is_closed and self.side == PositionSide.SHORT and self.quantity > 1e-12

    def _resolve_and_validate_qty_units(
        self,
        quantity: float | None,
        units: float | None,
    ) -> tuple[float, float]:
        assert self.instrument_config is not None
        inst = self.instrument_config
        cs = self.contract_size

        if quantity is not None and units is not None:
            if isinstance(quantity, bool) or isinstance(units, bool):
                raise ValueError("quantity và units phải là số thực.")
            q_f = float(quantity)
            u_f = float(units)
            if not math.isfinite(q_f) or not math.isfinite(u_f):
                raise ValueError("quantity và units phải hữu hạn.")
            if not math.isclose(u_f, q_f * cs, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"units ({u_f}) không nhất quán với quantity ({q_f}) × contract_size ({cs}).")
        elif quantity is not None:
            if isinstance(quantity, bool) or not isinstance(quantity, (int, float)):
                raise ValueError("quantity phải là số thực.")
            q_f = float(quantity)
            if not math.isfinite(q_f):
                raise ValueError("quantity phải hữu hạn.")
            u_f = q_f * cs
        elif units is not None:
            if isinstance(units, bool) or not isinstance(units, (int, float)):
                raise ValueError("units phải là số thực.")
            u_f = float(units)
            if not math.isfinite(u_f):
                raise ValueError("units phải hữu hạn.")
            q_f = u_f / cs
        else:
            raise ValueError("Phải cung cấp quantity hoặc units.")

        norm_q = validate_and_normalize_lot(
            q_f,
            min_lot=inst.min_lot,
            max_lot=inst.max_lot,
            lot_step=inst.lot_step,
        )
        return norm_q, norm_q * cs

    @staticmethod
    def _validate_price_fee_slippage(price: float, fee: float, slippage_cost: float) -> tuple[float, float, float]:
        for name, v in (("price", price), ("fee", fee), ("slippage_cost", slippage_cost)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ValueError(f"{name} phải là số thực hữu hạn.")
        p = float(price)
        f = float(fee)
        s = float(slippage_cost)
        if p <= 0.0:
            raise ValueError("price phải > 0.")
        if f < 0.0:
            raise ValueError("fee không được âm.")
        if s < 0.0:
            raise ValueError("slippage_cost không được âm.")
        return p, f, s

    def open(
        self,
        side: PositionSide,
        price: float,
        units: float | None = None,
        timestamp: pd.Timestamp | None = None,
        fee: float = 0.0,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
        opened_bar_index: int | None = None,
        opened_phase: ExecutionPhase | None = None,
        *,
        quantity: float | None = None,
        slippage_cost: float = 0.0,
        order_id: str | None = None,
        fill_id: str | None = None,
    ) -> None:
        if self.is_closed:
            raise ValueError(
                f"Position '{self.position_id}' ({self.symbol}) đã kết thúc vòng đời (CLOSED). Hãy khởi tạo Position mới."
            )
        if not self.is_flat:
            raise ValueError(f"Position '{self.position_id}' ({self.symbol}) đang mở, không thể gọi open() lại.")
        if side not in (PositionSide.LONG, PositionSide.SHORT):
            raise ValueError("side khi mở Position phải là LONG hoặc SHORT.")

        p, f, s = self._validate_price_fee_slippage(price, fee, slippage_cost)
        qty, u = self._resolve_and_validate_qty_units(quantity, units)
        ts = pd.Timestamp(timestamp) if timestamp is not None else pd.Timestamp("1970-01-01")
        b_idx = int(opened_bar_index) if opened_bar_index is not None else None

        self.side = side
        self.quantity = qty
        self.average_entry_price = p
        self.entry_time = ts
        self.entry_fee = f
        self.entry_slippage_cost = s
        self.stop_loss_price = float(stop_loss_price) if stop_loss_price is not None else None
        self.take_profit_price = float(take_profit_price) if take_profit_price is not None else None
        self.opened_bar_index = b_idx
        self.opened_phase = opened_phase
        self.contributions = [
            EntryContribution(
                order_id=order_id,
                fill_id=fill_id,
                time=ts,
                bar_index=b_idx,
                price=p,
                initial_units=u,
                remaining_units=u,
                initial_entry_fee=f,
                remaining_entry_fee=f,
                initial_slippage_cost=s,
                remaining_slippage_cost=s,
                contract_size=self.contract_size,
            )
        ]

    def scale_in(
        self,
        side: PositionSide,
        price: float,
        units: float | None = None,
        timestamp: pd.Timestamp | None = None,
        fee: float = 0.0,
        *,
        quantity: float | None = None,
        slippage_cost: float = 0.0,
        order_id: str | None = None,
        fill_id: str | None = None,
        bar_index: int | None = None,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> None:
        if self.is_flat or self.is_closed:
            raise ValueError(f"Không thể scale_in khi Position ({self.symbol}) đang FLAT hoặc CLOSED.")
        if side != self.side:
            raise ValueError(f"Scale-in phải cùng chiều với Position hiện tại ({self.side.name}).")

        p, f, s = self._validate_price_fee_slippage(price, fee, slippage_cost)
        add_qty, add_u = self._resolve_and_validate_qty_units(quantity, units)

        assert self.instrument_config is not None
        inst = self.instrument_config
        new_total_qty = validate_and_normalize_lot(
            self.quantity + add_qty,
            min_lot=inst.min_lot,
            max_lot=inst.max_lot,
            lot_step=inst.lot_step,
        )

        old_u = self.units
        new_u = old_u + add_u
        self.average_entry_price = (self.average_entry_price * old_u + p * add_u) / new_u
        self.quantity = new_total_qty
        self.entry_fee += f
        self.entry_slippage_cost += s

        if stop_loss_price is not None:
            self.stop_loss_price = float(stop_loss_price)
        if take_profit_price is not None:
            self.take_profit_price = float(take_profit_price)

        ts = pd.Timestamp(timestamp) if timestamp is not None else (self.entry_time or pd.Timestamp("1970-01-01"))
        b_idx = int(bar_index) if bar_index is not None else None
        self.contributions.append(
            EntryContribution(
                order_id=order_id,
                fill_id=fill_id,
                time=ts,
                bar_index=b_idx,
                price=p,
                initial_units=add_u,
                remaining_units=add_u,
                initial_entry_fee=f,
                remaining_entry_fee=f,
                initial_slippage_cost=s,
                remaining_slippage_cost=s,
                contract_size=self.contract_size,
            )
        )

    def unrealized_pnl(self, current_price: float) -> float:
        if self.is_flat:
            return 0.0
        if isinstance(current_price, bool) or not isinstance(current_price, (int, float)) or not math.isfinite(float(current_price)):
            raise ValueError("current_price phải là số thực hữu hạn.")
        multiplier = 1.0 if self.is_long else -1.0
        return (float(current_price) - self.average_entry_price) * self.units * multiplier

    def close(
        self,
        exit_price: float,
        timestamp: pd.Timestamp,
        fee: float = 0.0,
        reason: str = "EXIT",
        extra_metadata: dict[str, Any] | None = None,
        *,
        close_units: float | None = None,
        close_quantity: float | None = None,
        exit_order_id: str | None = None,
        exit_fill_id: str | None = None,
        exit_bar_index: int | None = None,
        exit_slippage_cost: float = 0.0,
        trade_id: str = "TRD-00001",
    ) -> dict[str, Any]:
        if self.is_flat or self.is_closed:
            raise ValueError(f"Không thể đóng vị thế đang Flat hoặc đã Closed ({self.symbol}).")

        exit_p, exit_fee_val, exit_slip_val = self._validate_price_fee_slippage(
            exit_price, fee, exit_slippage_cost
        )

        curr_units = self.units
        if close_units is None and close_quantity is None:
            c_qty = self.quantity
            c_units = curr_units
        else:
            c_qty, c_units = self._resolve_and_validate_qty_units(close_quantity, close_units)

        if c_units <= 0.0 or c_units > curr_units + 1e-9:
            raise ValueError(f"close_units ({c_units}) phải nằm trong khoảng (0, {curr_units}].")

        is_full_close = abs(c_units - curr_units) <= 1e-9
        ratio = 1.0 if is_full_close else (c_units / curr_units)

        if is_full_close:
            c_units = curr_units
            c_qty = self.quantity
            allocated_entry_fee = self.entry_fee
            allocated_entry_slippage = self.entry_slippage_cost
        else:
            assert self.instrument_config is not None
            inst = self.instrument_config
            # Kiểm tra phần còn lại cũng hợp lệ theo lot_step
            validate_and_normalize_lot(
                self.quantity - c_qty,
                min_lot=inst.min_lot,
                max_lot=inst.max_lot,
                lot_step=inst.lot_step,
            )
            allocated_entry_fee = self.entry_fee * ratio
            allocated_entry_slippage = self.entry_slippage_cost * ratio

        allocations: list[TradeAllocation] = []
        entry_order_ids: list[str] = []
        entry_fill_ids: list[str] = []

        # Nếu Position được sửa trực tiếp trong test cũ mà không có contributions, tạo 1 contribution tổng hợp
        if not self.contributions:
            self.contributions = [
                EntryContribution(
                    order_id=None,
                    fill_id=None,
                    time=self.entry_time or pd.Timestamp(timestamp),
                    bar_index=self.opened_bar_index,
                    price=self.average_entry_price,
                    initial_units=curr_units,
                    remaining_units=curr_units,
                    initial_entry_fee=self.entry_fee,
                    remaining_entry_fee=self.entry_fee,
                    initial_slippage_cost=self.entry_slippage_cost,
                    remaining_slippage_cost=self.entry_slippage_cost,
                    contract_size=self.contract_size,
                )
            ]

        for contrib in self.contributions:
            if contrib.remaining_units <= 1e-12:
                continue
            if is_full_close:
                alloc_u = contrib.remaining_units
                alloc_f = contrib.remaining_entry_fee
                alloc_s = contrib.remaining_slippage_cost
                contrib.remaining_units = 0.0
                contrib.remaining_entry_fee = 0.0
                contrib.remaining_slippage_cost = 0.0
            else:
                alloc_u = contrib.remaining_units * ratio
                alloc_f = contrib.remaining_entry_fee * ratio
                alloc_s = contrib.remaining_slippage_cost * ratio
                contrib.remaining_units -= alloc_u
                contrib.remaining_entry_fee -= alloc_f
                contrib.remaining_slippage_cost -= alloc_s

            alloc_q = alloc_u / self.contract_size if self.contract_size > 0 else 0.0
            allocations.append(
                TradeAllocation(
                    trade_id=trade_id,
                    source_entry_order_id=contrib.order_id,
                    source_entry_fill_id=contrib.fill_id,
                    source_entry_time=contrib.time,
                    source_entry_bar_index=contrib.bar_index,
                    allocated_units=alloc_u,
                    allocated_quantity=alloc_q,
                    allocated_entry_fee=alloc_f,
                    allocated_entry_slippage_cost=alloc_s,
                )
            )
            if contrib.order_id is not None and contrib.order_id not in entry_order_ids:
                entry_order_ids.append(contrib.order_id)
            if contrib.fill_id is not None and contrib.fill_id not in entry_fill_ids:
                entry_fill_ids.append(contrib.fill_id)

        single_entry_order_id = entry_order_ids[0] if len(entry_order_ids) == 1 and len(allocations) == 1 else None
        single_entry_fill_id = entry_fill_ids[0] if len(entry_fill_ids) == 1 and len(allocations) == 1 else None

        weighted_hb: float | None = None
        holding_bars: int | float | None = None
        if exit_bar_index is not None and allocations and all(a.source_entry_bar_index is not None for a in allocations):
            total_alloc_u = sum(a.allocated_units for a in allocations)
            if total_alloc_u > 0:
                weighted_hb = sum(
                    a.allocated_units * (int(exit_bar_index) - int(a.source_entry_bar_index))  # type: ignore[arg-type]
                    for a in allocations
                ) / total_alloc_u
                if len(allocations) == 1:
                    holding_bars = int(exit_bar_index) - int(allocations[0].source_entry_bar_index)  # type: ignore[arg-type]
                else:
                    holding_bars = weighted_hb
        elif exit_bar_index is not None and self.opened_bar_index is not None:
            hb_int = int(exit_bar_index) - int(self.opened_bar_index)
            weighted_hb = float(hb_int)
            holding_bars = hb_int

        multiplier = 1.0 if self.is_long else -1.0
        avg_ep = self.average_entry_price
        gross_pnl = (exit_p - avg_ep) * c_units * multiplier
        total_fee = allocated_entry_fee + exit_fee_val
        net_pnl = gross_pnl - total_fee
        entry_notional = avg_ep * c_units
        return_pct = (net_pnl / entry_notional) * 100.0 if entry_notional > 0 else 0.0

        trade_record: dict[str, Any] = {
            "trade_id": trade_id,
            "position_id": self.position_id,
            "symbol": self.symbol,
            "direction": "LONG" if self.is_long else "SHORT",
            "entry_time": self.entry_time,
            "exit_time": pd.Timestamp(timestamp),
            "entry_bar_index": self.opened_bar_index if len(allocations) <= 1 else None,
            "exit_bar_index": int(exit_bar_index) if exit_bar_index is not None else None,
            "entry_price": avg_ep,
            "exit_price": exit_p,
            "entry_order_ids": entry_order_ids,
            "entry_fill_ids": entry_fill_ids,
            "entry_order_id": single_entry_order_id,
            "entry_fill_id": single_entry_fill_id,
            "exit_order_id": exit_order_id,
            "exit_fill_id": exit_fill_id,
            "quantity": c_qty,
            "units": c_units,
            "holding_bars": holding_bars,
            "weighted_holding_bars": weighted_hb,
            "entry_fee": allocated_entry_fee,
            "exit_fee": exit_fee_val,
            "total_fee": total_fee,
            "entry_slippage_cost": allocated_entry_slippage,
            "exit_slippage_cost": exit_slip_val,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "return_pct": return_pct,
            "reason": reason,
            "_allocations": allocations,
        }
        if extra_metadata:
            trade_record.update(extra_metadata)

        if is_full_close:
            self.side = PositionSide.FLAT
            self.quantity = 0.0
            self.average_entry_price = 0.0
            self.entry_time = None
            self.entry_fee = 0.0
            self.entry_slippage_cost = 0.0
            self.stop_loss_price = None
            self.take_profit_price = None
            self.opened_bar_index = None
            self.opened_phase = None
            self.contributions.clear()
            self.is_closed = True
        else:
            self.quantity = validate_and_normalize_lot(
                self.quantity - c_qty,
                min_lot=self.instrument_config.min_lot,  # type: ignore[union-attr]
                max_lot=self.instrument_config.max_lot,  # type: ignore[union-attr]
                lot_step=self.instrument_config.lot_step,  # type: ignore[union-attr]
            )
            self.entry_fee = max(0.0, self.entry_fee - allocated_entry_fee)
            self.entry_slippage_cost = max(0.0, self.entry_slippage_cost - allocated_entry_slippage)

        return trade_record

    def to_view(self, mark_price: float | None = None) -> PositionView:
        if self.is_flat:
            return PositionView(
                symbol=self.symbol,
                side=PositionSide.FLAT,
                quantity=0.0,
                units=0.0,
                entry_price=0.0,
                entry_time=None,
                unrealized_pnl=0.0 if mark_price is not None else None,
            )
        upnl = self.unrealized_pnl(mark_price) if mark_price is not None else None
        return PositionView(
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            units=self.units,
            entry_price=self.average_entry_price,
            entry_time=self.entry_time,
            unrealized_pnl=upnl,
        )
