from __future__ import annotations
import copy
import math
from typing import Any, Mapping
import pandas as pd

from quant_backtester.config import InstrumentConfig, PositionMode
from quant_backtester.models.fill import Fill
from quant_backtester.models.order import ExecutionPhase, OrderSide
from quant_backtester.models.quote import QuoteBar
from quant_backtester.models.trade import (
    EntryContribution,
    ReconciliationReport,
    TradeAllocation,
    TradeRecord,
)
from .account import Account
from .ledger import TransactionType
from .position import Position, PositionSide, PositionView


TRADE_COLUMNS = [
    "trade_id",
    "position_id",
    "symbol",
    "direction",
    "entry_time",
    "exit_time",
    "entry_bar_index",
    "exit_bar_index",
    "entry_price",
    "exit_price",
    "entry_order_ids",
    "entry_fill_ids",
    "entry_order_id",
    "entry_fill_id",
    "exit_order_id",
    "exit_fill_id",
    "quantity",
    "units",
    "holding_bars",
    "weighted_holding_bars",
    "entry_fee",
    "exit_fee",
    "total_fee",
    "entry_slippage_cost",
    "exit_slippage_cost",
    "gross_pnl",
    "net_pnl",
    "return_pct",
    "reason",
]

OPEN_POSITION_COLUMNS = [
    "position_id",
    "symbol",
    "side",
    "quantity",
    "units",
    "entry_time",
    "entry_price",
    "entry_fee",
    "entry_slippage_cost",
    "mark_price",
    "unrealized_pnl",
    "entry_order_ids",
    "entry_fill_ids",
]

TRADE_ALLOCATION_COLUMNS = [
    "trade_id",
    "source_entry_order_id",
    "source_entry_fill_id",
    "source_entry_time",
    "source_entry_bar_index",
    "allocated_units",
    "allocated_quantity",
    "allocated_entry_fee",
    "allocated_entry_slippage_cost",
]


class Portfolio:
    def __init__(
        self,
        account: Account,
        instrument_config: InstrumentConfig | None = None,
        default_position_mode: PositionMode = PositionMode.NETTING_SINGLE,
        position_mode_by_symbol: dict[str, PositionMode] | None = None,
        instrument_by_symbol: dict[str, InstrumentConfig] | None = None,
        default_leverage: float = 100.0,
    ):
        self.account = account
        self.instrument_config = instrument_config
        self.default_leverage = float(default_leverage) if default_leverage > 0 else 100.0
        self.instrument_by_symbol: dict[str, InstrumentConfig] = dict(instrument_by_symbol) if instrument_by_symbol else {}
        if instrument_config is not None:
            self.account.validate_currency_compatibility(instrument_config.quote_currency)
            self.instrument_by_symbol.setdefault(instrument_config.symbol, instrument_config)
        for inst in self.instrument_by_symbol.values():
            self.account.validate_currency_compatibility(inst.quote_currency)

        self.default_position_mode = (
            default_position_mode
            if isinstance(default_position_mode, PositionMode)
            else PositionMode(default_position_mode)
        )
        self.position_mode_by_symbol: dict[str, PositionMode] = {}
        for sym, mode in (position_mode_by_symbol or {}).items():
            self.position_mode_by_symbol[str(sym)] = mode if isinstance(mode, PositionMode) else PositionMode(mode)

        self.active_positions: dict[str, Position] = {}
        self._flat_positions: dict[str, Position] = {}
        self.closed_trades: list[dict[str, Any]] = []
        self.trade_records: list[TradeRecord] = []
        self.trade_allocations: list[TradeAllocation] = []
        self._all_entry_contributions: list[EntryContribution] = []
        self._processed_fill_ids: set[str] = set()
        self._fill_process_count: int = 0
        self._trade_counter: int = 0
        self._position_counter: int = 0

    @property
    def positions(self) -> dict[str, Position]:
        return self.active_positions

    def _next_trade_id(self) -> str:
        self._trade_counter += 1
        return f"TRD-{self._trade_counter:05d}"

    def _next_position_id(self) -> str:
        self._position_counter += 1
        return f"POS-{self._position_counter:05d}"

    def _get_instrument(self, symbol: str) -> InstrumentConfig:
        if symbol in self.instrument_by_symbol:
            return self.instrument_by_symbol[symbol]
        if self.instrument_config is not None and self.instrument_config.symbol == symbol:
            return self.instrument_config
        if self.instrument_config is not None:
            inst = copy.deepcopy(self.instrument_config)
            inst.symbol = symbol
            return inst
        inst = InstrumentConfig(symbol=symbol)
        self.account.validate_currency_compatibility(inst.quote_currency)
        return inst

    def get_position_mode(self, symbol: str) -> PositionMode:
        return self.position_mode_by_symbol.get(symbol, self.default_position_mode)

    def get_position(self, symbol: str) -> Position:
        pos = self.active_positions.get(symbol)
        if pos is not None and not pos.is_flat:
            return pos
        flat_pos = self._flat_positions.get(symbol)
        if flat_pos is None or not flat_pos.is_flat or flat_pos.is_closed:
            flat_pos = Position(symbol=symbol, instrument_config=self._get_instrument(symbol), position_id="POS-FLAT")
            self._flat_positions[symbol] = flat_pos
        return flat_pos

    def get_position_view(self, symbol: str, mark_price: QuoteBar | float | None = None) -> PositionView:
        pos = self.active_positions.get(symbol)
        if pos is None or pos.is_flat:
            return PositionView(
                symbol=symbol,
                side=PositionSide.FLAT,
                quantity=0.0,
                units=0.0,
                entry_price=0.0,
                entry_time=None,
                unrealized_pnl=0.0 if mark_price is not None else None,
            )
        if isinstance(mark_price, QuoteBar):
            mp: float | None = mark_price.bid_close if pos.is_long else mark_price.ask_close
        elif mark_price is not None:
            mp = float(mark_price)
        else:
            mp = None
        return pos.to_view(mark_price=mp)

    def _snapshot_state(self) -> dict[str, Any]:
        active_contrib_snapshots = [
            (idx, copy.copy(c))
            for idx, c in enumerate(self._all_entry_contributions)
            if c.remaining_units > 0.0
        ]
        return {
            "active_positions": copy.deepcopy(self.active_positions),
            "len_closed_trades": len(self.closed_trades),
            "len_trade_records": len(self.trade_records),
            "len_trade_allocations": len(self.trade_allocations),
            "len_all_entry_contributions": len(self._all_entry_contributions),
            "active_contrib_snapshots": active_contrib_snapshots,
            "trade_counter": self._trade_counter,
            "position_counter": self._position_counter,
            "account_balance": self.account.balance,
            "account_net_deposits": self.account.net_deposits,
            "account_realized_gross_pnl": self.account.realized_gross_pnl,
            "account_total_fees_paid": self.account.total_fees_paid,
            "account_realized_fees": self.account.realized_fees,
            "account_open_position_entry_fees": self.account.open_position_entry_fees,
            "account_unrealized_pnl": self.account.unrealized_pnl,
            "account_used_margin": self.account.used_margin,
            "len_ledger_entries": len(self.account.ledger.entries),
            "ledger_counter": self.account.ledger._entry_counter,
        }

    def _restore_state(self, snap: dict[str, Any]) -> None:
        self.active_positions = snap["active_positions"]
        del self.closed_trades[snap["len_closed_trades"]:]
        del self.trade_records[snap["len_trade_records"]:]
        del self.trade_allocations[snap["len_trade_allocations"]:]
        del self._all_entry_contributions[snap["len_all_entry_contributions"]:]
        for idx, c_copy in snap["active_contrib_snapshots"]:
            self._all_entry_contributions[idx] = c_copy
        # Re-link active positions' contributions to _all_entry_contributions
        for pos in self.active_positions.values():
            for i, pc in enumerate(pos.contributions):
                for idx in range(len(self._all_entry_contributions) - 1, -1, -1):
                    ac = self._all_entry_contributions[idx]
                    if ac.fill_id == pc.fill_id and ac.order_id == pc.order_id and ac.time == pc.time:
                        self._all_entry_contributions[idx] = pc
                        break
        self._trade_counter = snap["trade_counter"]
        self._position_counter = snap["position_counter"]
        self.account.balance = snap["account_balance"]
        self.account.net_deposits = snap["account_net_deposits"]
        self.account.realized_gross_pnl = snap["account_realized_gross_pnl"]
        self.account.total_fees_paid = snap["account_total_fees_paid"]
        self.account.realized_fees = snap["account_realized_fees"]
        self.account.open_position_entry_fees = snap["account_open_position_entry_fees"]
        self.account.unrealized_pnl = snap["account_unrealized_pnl"]
        self.account.used_margin = snap["account_used_margin"]
        del self.account.ledger.entries[snap["len_ledger_entries"]:]
        self.account.ledger._entry_counter = snap["ledger_counter"]

    def _update_used_margin(
        self,
        current_prices: Mapping[str, float | QuoteBar] | None = None,
    ) -> float:
        total_margin = 0.0
        for symbol, pos in self.active_positions.items():
            if pos.is_flat or pos.units <= 0.0:
                continue
            inst = self._get_instrument(symbol)
            lev = float(inst.leverage) if inst.leverage is not None else self.default_leverage
            if current_prices is not None and symbol in current_prices:
                pq = current_prices[symbol]
                if isinstance(pq, QuoteBar):
                    mark_p = pq.bid_close if pos.is_long else pq.ask_close
                else:
                    mark_p = float(pq)
            else:
                mark_p = float(pos.avg_entry_price)
            if lev > 0.0 and mark_p > 0.0:
                total_margin += (pos.units * mark_p) / lev
        self.account.used_margin = total_margin
        return total_margin

    def on_fill(
        self,
        fill: Fill,
        *,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
        bar_index: int | None = None,
        phase: ExecutionPhase | None = None,
    ) -> None:
        if not isinstance(fill, Fill):
            raise TypeError("fill phải là một thực thể của Fill.")
        if not fill.fill_id:
            raise ValueError("fill.fill_id không được rỗng.")
        if fill.fill_id in self._processed_fill_ids:
            raise ValueError(f"DUPLICATE_FILL: Fill '{fill.fill_id}' đã được xử lý.")

        inst = self._get_instrument(fill.symbol)
        self.account.validate_currency_compatibility(inst.quote_currency)

        eff_bar_idx = bar_index
        if eff_bar_idx is None and fill.metadata and "bar_index" in fill.metadata:
            eff_bar_idx = int(fill.metadata["bar_index"])

        eff_phase = phase
        if eff_phase is None and fill.metadata and "phase" in fill.metadata:
            raw_ph = fill.metadata["phase"]
            eff_phase = raw_ph if isinstance(raw_ph, ExecutionPhase) else ExecutionPhase(raw_ph)

        slip_cost = (
            float(fill.metadata["slippage_cost"])
            if (fill.metadata and "slippage_cost" in fill.metadata)
            else float(fill.slippage) * float(fill.units)
        )

        target_side = PositionSide.LONG if fill.side == OrderSide.BUY else PositionSide.SHORT
        pos = self.active_positions.get(fill.symbol)
        is_currently_flat = pos is None or pos.is_flat

        # Kiểm tra sớm PositionMode trước khi tạo thay đổi
        if not is_currently_flat and pos is not None and pos.side == target_side:
            mode = self.get_position_mode(fill.symbol)
            if mode == PositionMode.NETTING_SINGLE:
                raise ValueError(
                    f"SCALE_IN_NOT_ALLOWED: Symbol '{fill.symbol}' có chế độ NETTING_SINGLE, không được phép nhồi lệnh cùng chiều."
                )

        snap = self._snapshot_state()
        try:
            if is_currently_flat:
                # Nhánh 1 & 2: Mở vị thế mới từ FLAT
                new_pos = Position(
                    symbol=fill.symbol,
                    instrument_config=inst,
                    position_id=self._next_position_id(),
                )
                new_pos.open(
                    side=target_side,
                    price=fill.price,
                    units=fill.units,
                    quantity=fill.quantity,
                    timestamp=fill.time,
                    fee=fill.fee,
                    slippage_cost=slip_cost,
                    order_id=fill.order_id,
                    fill_id=fill.fill_id,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                    opened_bar_index=eff_bar_idx,
                    opened_phase=eff_phase,
                )
                self.account.deduct_fee(
                    fill.fee,
                    fill.time,
                    reason=f"COMMISSION_{fill.side.name}",
                    is_opening=True,
                    symbol=fill.symbol,
                    order_id=fill.order_id,
                    fill_id=fill.fill_id,
                )
                self.active_positions[fill.symbol] = new_pos
                self._all_entry_contributions.append(new_pos.contributions[-1])

            elif pos is not None and pos.side == target_side:
                # Nhánh 3 & 4: Scale-in cùng chiều (khi NETTING_SCALE_IN)
                pos.scale_in(
                    side=target_side,
                    price=fill.price,
                    units=fill.units,
                    quantity=fill.quantity,
                    timestamp=fill.time,
                    fee=fill.fee,
                    slippage_cost=slip_cost,
                    order_id=fill.order_id,
                    fill_id=fill.fill_id,
                    bar_index=eff_bar_idx,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                )
                self.account.deduct_fee(
                    fill.fee,
                    fill.time,
                    reason=f"COMMISSION_{fill.side.name}",
                    is_opening=True,
                    symbol=fill.symbol,
                    order_id=fill.order_id,
                    fill_id=fill.fill_id,
                )
                self._all_entry_contributions.append(pos.contributions[-1])

            else:
                assert pos is not None
                # Ngược chiều: Partial Close, Full Close, hoặc Single-Fill Over-Close Reverse
                if fill.units <= pos.units + 1e-9:
                    # Nhánh 5, 6, 8, 9: Partial Close hoặc Full Close
                    trade_id = self._next_trade_id()
                    c_units = pos.units if abs(fill.units - pos.units) <= 1e-9 else fill.units
                    c_qty = pos.quantity if abs(fill.units - pos.units) <= 1e-9 else fill.quantity
                    trade_dict = pos.close(
                        exit_price=fill.price,
                        timestamp=fill.time,
                        fee=fill.fee,
                        reason=fill.reason,
                        close_units=c_units,
                        close_quantity=c_qty,
                        exit_order_id=fill.order_id,
                        exit_fill_id=fill.fill_id,
                        exit_bar_index=eff_bar_idx,
                        exit_slippage_cost=slip_cost,
                        trade_id=trade_id,
                    )
                    allocs: list[TradeAllocation] = trade_dict.pop("_allocations", [])
                    self.account.realize_trade_pnl(
                        trade_dict["gross_pnl"],
                        fill.fee,
                        fill.time,
                        reason=f"CLOSE_{pos.symbol}",
                        symbol=pos.symbol,
                        order_id=fill.order_id,
                        fill_id=fill.fill_id,
                        trade_id=trade_id,
                        allocated_entry_fee=trade_dict["entry_fee"],
                    )
                    self.closed_trades.append(trade_dict)
                    self.trade_records.append(TradeRecord(**trade_dict))
                    self.trade_allocations.extend(allocs)

                    if pos.is_closed or pos.is_flat:
                        self.active_positions.pop(fill.symbol, None)
                        if not self.active_positions:
                            self.account.unrealized_pnl = 0.0

                else:
                    # Nhánh 7 & 10: Single-Fill Over-Close Reverse (fill.units > pos.units)
                    close_units = pos.units
                    close_qty = pos.quantity
                    remaining_units = fill.units - close_units
                    remaining_qty = remaining_units / inst.contract_size

                    close_ratio = close_units / fill.units
                    close_fee = fill.fee * close_ratio
                    open_fee = fill.fee - close_fee
                    close_slip = slip_cost * close_ratio
                    open_slip = slip_cost - close_slip

                    trade_id = self._next_trade_id()
                    trade_dict = pos.close(
                        exit_price=fill.price,
                        timestamp=fill.time,
                        fee=close_fee,
                        reason="REVERSE",
                        close_units=close_units,
                        close_quantity=close_qty,
                        exit_order_id=fill.order_id,
                        exit_fill_id=fill.fill_id,
                        exit_bar_index=eff_bar_idx,
                        exit_slippage_cost=close_slip,
                        trade_id=trade_id,
                    )
                    allocs = trade_dict.pop("_allocations", [])
                    self.account.realize_trade_pnl(
                        trade_dict["gross_pnl"],
                        close_fee,
                        fill.time,
                        reason=f"REVERSE_{pos.symbol}",
                        symbol=pos.symbol,
                        order_id=fill.order_id,
                        fill_id=fill.fill_id,
                        trade_id=trade_id,
                        allocated_entry_fee=trade_dict["entry_fee"],
                    )
                    self.closed_trades.append(trade_dict)
                    self.trade_records.append(TradeRecord(**trade_dict))
                    self.trade_allocations.extend(allocs)
                    self.active_positions.pop(fill.symbol, None)

                    # Mở Position mới hoàn toàn cho phần dư ngược chiều (position_id mới)
                    rev_pos = Position(
                        symbol=fill.symbol,
                        instrument_config=inst,
                        position_id=self._next_position_id(),
                    )
                    rev_pos.open(
                        side=target_side,
                        price=fill.price,
                        units=remaining_units,
                        quantity=remaining_qty,
                        timestamp=fill.time,
                        fee=open_fee,
                        slippage_cost=open_slip,
                        order_id=fill.order_id,
                        fill_id=fill.fill_id,
                        stop_loss_price=stop_loss_price,
                        take_profit_price=take_profit_price,
                        opened_bar_index=eff_bar_idx,
                        opened_phase=eff_phase,
                    )
                    self.account.deduct_fee(
                        open_fee,
                        fill.time,
                        reason=f"COMMISSION_{fill.side.name}",
                        is_opening=True,
                        symbol=fill.symbol,
                        order_id=fill.order_id,
                        fill_id=fill.fill_id,
                    )
                    self.active_positions[fill.symbol] = rev_pos
                    self._all_entry_contributions.append(rev_pos.contributions[-1])

        except Exception:
            self._restore_state(snap)
            raise

        # Commit thành công -> đánh dấu fill_id đã xử lý và cập nhật used_margin
        self._processed_fill_ids.add(fill.fill_id)
        self._fill_process_count += 1
        self._update_used_margin()

    def get_equity(self, current_prices: Mapping[str, float | QuoteBar]) -> float:
        """
        Mark-to-Market Equity:
        Nếu truyền QuoteBar: vị thế LONG định giá theo Bid Close, vị thế SHORT định giá theo Ask Close.
        Nếu truyền float: định giá trực tiếp theo float (tương thích ngược cho unit test đơn lẻ).
        """
        total_unrealized = 0.0
        for symbol, pos in list(self.active_positions.items()):
            if not pos.is_flat and symbol in current_prices:
                price_or_quote = current_prices[symbol]
                if isinstance(price_or_quote, QuoteBar):
                    mark_p = price_or_quote.bid_close if pos.is_long else price_or_quote.ask_close
                else:
                    mark_p = float(price_or_quote)
                total_unrealized += pos.unrealized_pnl(mark_p)
        self.account.unrealized_pnl = total_unrealized
        self._update_used_margin(current_prices)
        return self.account.equity

    # -------------------------------------------------------------------------
    # EXPORT DATAFRAMES (TASK E7)
    # -------------------------------------------------------------------------
    def trades_to_dataframe(self) -> pd.DataFrame:
        if not self.closed_trades:
            return pd.DataFrame(columns=TRADE_COLUMNS)
        df = pd.DataFrame(self.closed_trades)
        for col in TRADE_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[TRADE_COLUMNS]

    def open_positions_to_dataframe(self, mark_prices: Mapping[str, QuoteBar] | None = None) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for symbol, pos in self.active_positions.items():
            if pos.is_flat:
                continue
            mark_p: float | None = None
            upnl: float = 0.0
            if mark_prices is not None and symbol in mark_prices:
                qb = mark_prices[symbol]
                if not isinstance(qb, QuoteBar):
                    raise TypeError(
                        f"open_positions_to_dataframe yêu cầu QuoteBar có đầy đủ Bid/Ask cho symbol '{symbol}', nhận {type(qb).__name__}."
                    )
                mark_p = qb.bid_close if pos.is_long else qb.ask_close
                upnl = pos.unrealized_pnl(mark_p)
            else:
                mark_p = pos.average_entry_price
                upnl = 0.0

            entry_order_ids = [c.order_id for c in pos.contributions if c.order_id is not None and c.remaining_units > 1e-12]
            entry_fill_ids = [c.fill_id for c in pos.contributions if c.fill_id is not None and c.remaining_units > 1e-12]
            rows.append({
                "position_id": pos.position_id,
                "symbol": pos.symbol,
                "side": pos.side.name,
                "quantity": pos.quantity,
                "units": pos.units,
                "entry_time": pos.entry_time,
                "entry_price": pos.average_entry_price,
                "entry_fee": pos.entry_fee,
                "entry_slippage_cost": pos.entry_slippage_cost,
                "mark_price": mark_p,
                "unrealized_pnl": upnl,
                "entry_order_ids": entry_order_ids,
                "entry_fill_ids": entry_fill_ids,
            })

        if not rows:
            return pd.DataFrame(columns=OPEN_POSITION_COLUMNS)
        return pd.DataFrame(rows, columns=OPEN_POSITION_COLUMNS)

    def trade_allocations_to_dataframe(self) -> pd.DataFrame:
        if not self.trade_allocations:
            return pd.DataFrame(columns=TRADE_ALLOCATION_COLUMNS)
        return pd.DataFrame([a.to_dict() for a in self.trade_allocations], columns=TRADE_ALLOCATION_COLUMNS)

    # -------------------------------------------------------------------------
    # RECONCILIATION & 7 ACCOUNTING INVARIANTS (TASK E6)
    # -------------------------------------------------------------------------
    def reconcile(self, mark_prices: Mapping[str, QuoteBar | float] | None = None) -> ReconciliationReport:
        passed: list[str] = []
        failed: list[str] = []
        diffs: dict[str, float] = {}
        msgs: list[str] = []

        def _check(name: str, actual: float, expected: float, detail: str) -> None:
            diff = float(actual) - float(expected)
            diffs[name] = diff
            if math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9):
                passed.append(name)
            else:
                failed.append(name)
                msgs.append(f"[{name}] {detail}: actual={actual:.10f}, expected={expected:.10f}, diff={diff:.10e}")

        acc = self.account
        entries = acc.ledger.entries

        # Invariant 1: Balance == InitialBalance + NetDeposits + RealizedGrossPnL - TotalFeesPaid
        expected_balance = acc.initial_balance + acc.net_deposits + acc.realized_gross_pnl - acc.total_fees_paid
        _check("INVARIANT_1_BALANCE", acc.balance, expected_balance, "Account.balance lệch công thức nguồn")

        # Invariant 2: RealizedGrossPnL == sum(ClosedTrades.gross_pnl) == sum(Ledger[TRADE_PNL].amount)
        trades_gross = sum(float(t["gross_pnl"]) for t in self.closed_trades)
        ledger_gross = sum(float(e.amount) for e in entries if e.transaction_type == TransactionType.TRADE_PNL)
        _check("INVARIANT_2A_REALIZED_GROSS_TRADES", acc.realized_gross_pnl, trades_gross, "Account.realized_gross_pnl lệch tổng ClosedTrades.gross_pnl")
        _check("INVARIANT_2B_REALIZED_GROSS_LEDGER", ledger_gross, trades_gross, "Tổng Ledger TRADE_PNL lệch tổng ClosedTrades.gross_pnl")

        # Invariant 3: Fee Allocation (TotalFeesPaid == RealizedFees + OpenPositionEntryFees == -sum(Ledger[FEE].amount))
        trades_realized_fees = sum(float(t["entry_fee"]) + float(t["exit_fee"]) for t in self.closed_trades)
        open_pos_fees = sum(float(p.entry_fee) for p in self.active_positions.values() if not p.is_flat)
        ledger_fees = -sum(float(e.amount) for e in entries if e.transaction_type == TransactionType.FEE)
        _check("INVARIANT_3A_FEE_SPLIT", acc.total_fees_paid, acc.realized_fees + acc.open_position_entry_fees, "TotalFeesPaid lệch RealizedFees + OpenPositionEntryFees")
        _check("INVARIANT_3B_FEE_POSITIONS_TRADES", acc.total_fees_paid, trades_realized_fees + open_pos_fees, "TotalFeesPaid lệch phí trên ClosedTrades + OpenPositions")
        _check("INVARIANT_3C_FEE_LEDGER", acc.total_fees_paid, ledger_fees, "TotalFeesPaid lệch tổng bút toán FEE trên Ledger")

        # Invariant 4: Ledger Running Balance
        running_bal = acc.initial_balance
        ledger_chain_ok = True
        max_chain_diff = 0.0
        for idx, e in enumerate(entries):
            running_bal += float(e.amount)
            d = float(e.balance_after) - running_bal
            if abs(d) > abs(max_chain_diff):
                max_chain_diff = d
            if not math.isclose(float(e.balance_after), running_bal, rel_tol=1e-9, abs_tol=1e-9):
                ledger_chain_ok = False
                msgs.append(f"[INVARIANT_4_LEDGER_RUNNING_BALANCE] Dòng {idx} ({e.entry_id}): balance_after={e.balance_after}, expected={running_bal}")
        diffs["INVARIANT_4_LEDGER_RUNNING_BALANCE"] = max_chain_diff
        if ledger_chain_ok and math.isclose(running_bal, acc.balance, rel_tol=1e-9, abs_tol=1e-9):
            passed.append("INVARIANT_4_LEDGER_RUNNING_BALANCE")
        else:
            failed.append("INVARIANT_4_LEDGER_RUNNING_BALANCE")
            if ledger_chain_ok:
                msgs.append(f"[INVARIANT_4_LEDGER_RUNNING_BALANCE] Số dư cuối Ledger ({running_bal}) khác Account.balance ({acc.balance})")

        # Invariant 5: Equity == Balance + sum(OpenPositions.unrealized_pnl)
        if mark_prices is not None:
            self.get_equity(mark_prices)
        expected_equity = acc.balance + acc.unrealized_pnl
        _check("INVARIANT_5_EQUITY", acc.equity, expected_equity, "Account.equity lệch Balance + UnrealizedPnL")

        # Invariant 6: EntryContribution Conservation (allocated + remaining == initial)
        contrib_ok = True
        max_contrib_diff = 0.0
        for c in self._all_entry_contributions:
            allocs_for_c = [
                a for a in self.trade_allocations
                if a.source_entry_fill_id == c.fill_id and a.source_entry_order_id == c.order_id and a.source_entry_time == c.time
            ]
            sum_u = sum(a.allocated_units for a in allocs_for_c) + c.remaining_units
            sum_f = sum(a.allocated_entry_fee for a in allocs_for_c) + c.remaining_entry_fee
            sum_s = sum(a.allocated_entry_slippage_cost for a in allocs_for_c) + c.remaining_slippage_cost
            for label, actual_v, init_v in (
                ("units", sum_u, c.initial_units),
                ("entry_fee", sum_f, c.initial_entry_fee),
                ("slippage_cost", sum_s, c.initial_slippage_cost),
            ):
                d = actual_v - init_v
                if abs(d) > abs(max_contrib_diff):
                    max_contrib_diff = d
                if not math.isclose(actual_v, init_v, rel_tol=1e-9, abs_tol=1e-9):
                    contrib_ok = False
                    msgs.append(
                        f"[INVARIANT_6_ENTRY_CONTRIBUTION] Fill {c.fill_id} ({label}): allocated+remaining={actual_v}, initial={init_v}"
                    )
        diffs["INVARIANT_6_ENTRY_CONTRIBUTION"] = max_contrib_diff
        if contrib_ok:
            passed.append("INVARIANT_6_ENTRY_CONTRIBUTION")
        else:
            failed.append("INVARIANT_6_ENTRY_CONTRIBUTION")

        # Invariant 7: Fill Idempotency & Ledger Fill Traceability
        idemp_ok = len(self._processed_fill_ids) == self._fill_process_count
        for e in entries:
            if e.transaction_type in (TransactionType.FEE, TransactionType.TRADE_PNL) and e.fill_id is not None:
                if e.fill_id not in self._processed_fill_ids:
                    idemp_ok = False
                    msgs.append(f"[INVARIANT_7_FILL_IDEMPOTENCY] LedgerEntry {e.entry_id} tham chiếu fill_id lạ: {e.fill_id}")
        diffs["INVARIANT_7_FILL_IDEMPOTENCY"] = float(self._fill_process_count - len(self._processed_fill_ids))
        if idemp_ok:
            passed.append("INVARIANT_7_FILL_IDEMPOTENCY")
        else:
            failed.append("INVARIANT_7_FILL_IDEMPOTENCY")
            if len(self._processed_fill_ids) != self._fill_process_count:
                msgs.append(
                    f"[INVARIANT_7_FILL_IDEMPOTENCY] Số lần xử lý Fill ({self._fill_process_count}) khác số fill_id duy nhất ({len(self._processed_fill_ids)})"
                )

        return ReconciliationReport(
            is_valid=(len(failed) == 0),
            checks_passed=passed,
            checks_failed=failed,
            differences=diffs,
            messages=msgs,
        )
