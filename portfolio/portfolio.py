from __future__ import annotations
from typing import Any
import pandas as pd
from .account import Account
from .position import Position, PositionSide
from quant_backtester.models.fill import Fill
from quant_backtester.models.order import OrderSide


class Portfolio:
    def __init__(self, account: Account):
        self.account = account
        self.positions: dict[str, Position] = {}
        self.closed_trades: list[dict[str, Any]] = []

    def get_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def on_fill(self, fill: Fill):
        pos = self.get_position(fill.symbol)
        # Trừ phí khớp lệnh trên tài khoản
        self.account.deduct_fee(fill.fee, fill.time, f"COMMISSION_{fill.side.name}")

        target_side = PositionSide.LONG if fill.side == OrderSide.BUY else PositionSide.SHORT

        if pos.is_flat:
            # Mở vị thế mới
            pos.open(side=target_side, price=fill.price, units=fill.units, timestamp=fill.time, fee=fill.fee)
        elif pos.side == target_side:
            # Tăng vị thế (hiện tại tính weighted average price)
            total_units = pos.units + fill.units
            pos.entry_price = (pos.entry_price * pos.units + fill.price * fill.units) / total_units
            pos.units = total_units
            pos.entry_fee += fill.fee
        else:
            # Đảo chiều hoặc đóng vị thế
            if fill.units == pos.units:
                trade = pos.close(exit_price=fill.price, timestamp=fill.time, fee=fill.fee, reason=fill.reason)
                self.account.realize_trade_pnl(trade["gross_pnl"], fill.fee, fill.time, f"CLOSE_{pos.symbol}")
                self.closed_trades.append(trade)
            elif fill.units > pos.units:
                # Đóng vị thế cũ rồi mở vị thế ngược lại với lượng dôi dư
                remaining_units = fill.units - pos.units
                ratio = pos.units / fill.units
                close_fee = fill.fee * ratio
                open_fee = fill.fee * (1 - ratio)

                trade = pos.close(exit_price=fill.price, timestamp=fill.time, fee=close_fee, reason="REVERSE")
                self.account.realize_trade_pnl(trade["gross_pnl"], close_fee, fill.time, f"REVERSE_{pos.symbol}")
                self.closed_trades.append(trade)

                pos.open(side=target_side, price=fill.price, units=remaining_units, timestamp=fill.time, fee=open_fee)

    def get_equity(self, current_prices: dict[str, float]) -> float:
        total_unrealized = 0.0
        for symbol, pos in self.positions.items():
            if not pos.is_flat and symbol in current_prices:
                total_unrealized += pos.unrealized_pnl(current_prices[symbol])
        self.account.unrealized_pnl = total_unrealized
        return self.account.equity

