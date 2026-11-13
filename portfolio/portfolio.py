from __future__ import annotations
from typing import Any
import pandas as pd

from .account import Account
from .position import Position, PositionSide
from quant_backtester.models.fill import Fill
from quant_backtester.models.order import ExecutionPhase, OrderSide
from quant_backtester.models.quote import QuoteBar


class Portfolio:
    def __init__(self, account: Account):
        self.account = account
        self.positions: dict[str, Position] = {}
        self.closed_trades: list[dict[str, Any]] = []

    def get_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def on_fill(
        self,
        fill: Fill,
        *,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
        bar_index: int | None = None,
        phase: ExecutionPhase | None = None,
    ):
        pos = self.get_position(fill.symbol)
        target_side = PositionSide.LONG if fill.side == OrderSide.BUY else PositionSide.SHORT

        if pos.is_flat:
            # Trừ phí mở vị thế trên tài khoản
            self.account.deduct_fee(fill.fee, fill.time, f"COMMISSION_{fill.side.name}")
            # Mở vị thế mới
            pos.open(
                side=target_side,
                price=fill.price,
                units=fill.units,
                timestamp=fill.time,
                fee=fill.fee,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                opened_bar_index=bar_index,
                opened_phase=phase,
            )
        elif pos.side == target_side:
            # Trừ phí tăng vị thế trên tài khoản
            self.account.deduct_fee(fill.fee, fill.time, f"COMMISSION_{fill.side.name}")
            # Tăng vị thế (hiện tại tính weighted average price)
            total_units = pos.units + fill.units
            pos.entry_price = (pos.entry_price * pos.units + fill.price * fill.units) / total_units
            pos.units = total_units
            pos.entry_fee += fill.fee
        else:
            # Đảo chiều hoặc đóng vị thế (realize_trade_pnl đã trừ exit_fee)
            if abs(fill.units - pos.units) <= 1e-9:
                trade = pos.close(exit_price=fill.price, timestamp=fill.time, fee=fill.fee, reason=fill.reason)
                self.account.realize_trade_pnl(trade["gross_pnl"], fill.fee, fill.time, f"CLOSE_{pos.symbol}")
                self.closed_trades.append(trade)
            elif fill.units > pos.units:
                # Đóng vị thế cũ rồi mở vị thế ngược lại với lượng dôi dư
                remaining_units = fill.units - pos.units
                ratio = pos.units / fill.units
                close_fee = fill.fee * ratio
                open_fee = fill.fee * (1.0 - ratio)

                trade = pos.close(exit_price=fill.price, timestamp=fill.time, fee=close_fee, reason="REVERSE")
                self.account.realize_trade_pnl(trade["gross_pnl"], close_fee, fill.time, f"REVERSE_{pos.symbol}")
                self.closed_trades.append(trade)

                self.account.deduct_fee(open_fee, fill.time, f"COMMISSION_{fill.side.name}")
                pos.open(
                    side=target_side,
                    price=fill.price,
                    units=remaining_units,
                    timestamp=fill.time,
                    fee=open_fee,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                    opened_bar_index=bar_index,
                    opened_phase=phase,
                )

    def get_equity(self, current_prices: dict[str, float | QuoteBar]) -> float:
        """
        Mark-to-Market Equity:
        Nếu truyền QuoteBar: vị thế LONG định giá theo Bid Close, vị thế SHORT định giá theo Ask Close.
        Nếu truyền float: định giá trực tiếp theo float (tương thích ngược).
        """
        total_unrealized = 0.0
        for symbol, pos in self.positions.items():
            if not pos.is_flat and symbol in current_prices:
                price_or_quote = current_prices[symbol]
                if isinstance(price_or_quote, QuoteBar):
                    mark_p = price_or_quote.bid_close if pos.is_long else price_or_quote.ask_close
                else:
                    mark_p = float(price_or_quote)
                total_unrealized += pos.unrealized_pnl(mark_p)
        self.account.unrealized_pnl = total_unrealized
        return self.account.equity
