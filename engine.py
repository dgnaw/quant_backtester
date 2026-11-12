"""Strategy-agnostic event-driven OHLC backtest engine."""
from __future__ import annotations
from typing import Any
import pandas as pd

from .config import EngineConfig, BacktestConfig
from .strategies.base import Strategy, StrategyMode
from .strategies.context import FillView, StrategyContext
from .strategies.adapters import frame_to_signals
from .strategies.translator import SignalTranslator
from .portfolio.account import Account
from .portfolio.portfolio import Portfolio
from .portfolio.position import PositionSide
from .execution.base import ExecutionModel
from .execution.immediate import ImmediateExecutionModel
from .execution.fee import PercentageFeeModel
from .execution.slippage import FixedSlippageModel
from .execution.order_manager import OrderManager
from .risk.base import BaseRiskManager
from .risk.simple import SimpleRiskManager
from .models.order import OrderIntent, OrderSide, OrderType
from .models.result import BacktestResult
from .models.signal import Signal, SignalType
from .analytics import calculate_metrics


class BacktestEngine:
    """
    Backtest Engine đóng vai trò là Orchestrator:
    - Điều phối luồng dữ liệu (Data Feed -> Strategy -> RiskManager -> ExecutionModel -> Portfolio).
    - Hoàn toàn Strategy-Agnostic: không chứa logic riêng của bất kỳ chiến lược nào.
    - Hỗ trợ Dependency Injection cho ExecutionModel và RiskManager.
    - Hỗ trợ cả hai chế độ Strategy: VECTORIZED và EVENT_DRIVEN (cùng BOTH).
    """
    def __init__(
        self,
        config: EngineConfig | None = None,
        execution_model: ExecutionModel | None = None,
        risk_manager: BaseRiskManager | None = None,
    ):
        self.config = config if config is not None else EngineConfig()

        # Tiêm phụ thuộc (Dependency Injection) các Model
        self.execution_model = execution_model if execution_model is not None else ImmediateExecutionModel(
            fee_model=PercentageFeeModel(self.config.fee.fee_rate_per_side),
            slippage_model=FixedSlippageModel(self.config.slippage.slippage_price),
        )
        self.risk_manager = risk_manager if risk_manager is not None else SimpleRiskManager(
            config=self.config.risk,
            instrument_config=self.config.instrument,
        )

    def _resolve_execution_mode(self, strategy: Strategy, run_mode: StrategyMode | None) -> StrategyMode:
        strategy.__class__.validate_mode_contract()
        strat_mode = getattr(strategy, "mode", StrategyMode.VECTORIZED)
        if run_mode is not None:
            if strat_mode != StrategyMode.BOTH and run_mode != strat_mode:
                raise ValueError(
                    f"Strategy '{strategy.name}' có mode={strat_mode.value}, không thể chạy ở run_mode={run_mode.value}."
                )
            return run_mode
        if strat_mode == StrategyMode.EVENT_DRIVEN:
            return StrategyMode.EVENT_DRIVEN
        return StrategyMode.VECTORIZED

    def _emit_bar_signal(
        self,
        exec_mode: StrategyMode,
        strategy: Strategy,
        signal_list: list[Signal] | None,
        bar_idx: int,
        row: pd.Series,
        df_bars: pd.DataFrame,
        symbol: str,
        pos_view,
    ) -> Signal:
        if exec_mode == StrategyMode.VECTORIZED:
            assert signal_list is not None
            return signal_list[bar_idx]

        ctx = StrategyContext(
            time=pd.Timestamp(row["time"]),
            bar_index=bar_idx,
            symbol=symbol,
            position=pos_view,
            data=df_bars,
        )
        emitted = strategy.on_bar(row.copy(), ctx)
        if emitted is None:
            return Signal(time=pd.Timestamp(row["time"]), symbol=symbol, signal_type=SignalType.FLAT)
        if isinstance(emitted, list):
            if not emitted:
                return Signal(time=pd.Timestamp(row["time"]), symbol=symbol, signal_type=SignalType.FLAT)
            sig = emitted[0]
        else:
            sig = emitted
        if not isinstance(sig, Signal):
            raise TypeError(f"on_bar() phải trả về Signal | list[Signal] | None, nhận {type(sig).__name__}.")
        if sig.symbol != symbol:
            raise ValueError(f"Signal.symbol ({sig.symbol}) không khớp với cfg.instrument.symbol ({symbol}).")
        return sig

    @staticmethod
    def _call_approve(
        risk_manager: BaseRiskManager,
        intent: OrderIntent,
        balance: float,
        equity: float,
        ref_price: float,
    ) -> OrderIntent | None:
        try:
            return risk_manager.approve_order_intent(intent, balance, equity, reference_price=ref_price)
        except TypeError:
            return risk_manager.approve_order_intent(intent, balance, equity)

    @staticmethod
    def _call_intrabar_exit(
        risk_manager: BaseRiskManager,
        symbol: str,
        side: OrderSide,
        entry_price: float,
        units: float,
        bar: pd.Series,
        active_directives: dict[str, Any],
    ) -> OrderIntent | None:
        try:
            return risk_manager.check_intrabar_exit(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                units=units,
                bar=bar,
                stop_loss_price=active_directives.get("stop_loss_price"),
                take_profit_price=active_directives.get("take_profit_price"),
                stop_loss_pct=active_directives.get("stop_loss_pct"),
                take_profit_pct=active_directives.get("take_profit_pct"),
            )
        except TypeError:
            return risk_manager.check_intrabar_exit(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                units=units,
                bar=bar,
            )

    def run(
        self,
        market_data: pd.DataFrame,
        strategy: Strategy,
        trade_start: int = 0,
        run_mode: StrategyMode | None = None,
    ) -> BacktestResult:
        if hasattr(strategy, "reset"):
            strategy.reset()

        cfg = self.config
        symbol = cfg.instrument.symbol
        exec_mode = self._resolve_execution_mode(strategy, run_mode)

        if exec_mode == StrategyMode.VECTORIZED:
            raw_output = strategy.generate_signals(market_data.copy())
            df_bars = (raw_output.data if hasattr(raw_output, "data") else raw_output).reset_index(drop=True)
            signal_list: list[Signal] | None = frame_to_signals(raw_output, symbol=symbol)
        else:
            required_ohlc = {"time", "open", "high", "low", "close"}
            missing = required_ohlc.difference(market_data.columns)
            if missing:
                raise ValueError(f"Dữ liệu đầu vào thiếu cột OHLC bắt buộc: {sorted(missing)}")
            df_bars = market_data.copy().reset_index(drop=True)
            signal_list = None

        warmup = getattr(strategy, "warmup_period", 0)
        if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
            raise ValueError("warmup_period phải là số nguyên không âm.")
        if len(df_bars) < warmup:
            raise ValueError(f"Không đủ dữ liệu ({len(df_bars)} nến) cho warmup_period ({warmup}).")
        if not 0 <= trade_start < len(df_bars):
            raise ValueError("trade_start nằm ngoài phạm vi dữ liệu.")
        effective_trade_start = max(trade_start, warmup)

        account = Account(initial_balance=cfg.account.initial_balance, base_currency=cfg.account.base_currency)
        portfolio = Portfolio(account=account)
        order_manager = OrderManager()
        translator = SignalTranslator(
            default_quantity=cfg.instrument.lot_size,
            reverse_policy=cfg.reverse_policy,
        )

        pos = portfolio.get_position(symbol)
        active_directives: dict[str, Any] = {}

        pending_intent: OrderIntent | None = None
        equity_rows: list[dict] = []

        for i, row in df_bars.iterrows():
            bar_time = pd.Timestamp(row["time"])
            open_price = float(row["open"])

            if i < effective_trade_start:
                equity_rows.append({"time": row.time, "balance": account.balance, "equity": account.balance, "position": 0})
                if i == effective_trade_start - 1 and i >= warmup - 1 and i + 1 < len(df_bars):
                    sig = self._emit_bar_signal(exec_mode, strategy, signal_list, i, row, df_bars, symbol, pos.to_view())
                    pending_intent = translator.translate(sig, pos.to_view())
                continue

            current_side_order = OrderSide.BUY if pos.is_long else (OrderSide.SELL if pos.is_short else None)

            # 1. Kiểm tra GAP tại Open qua RiskManager (nếu đang giữ vị thế)
            if not pos.is_flat and current_side_order is not None:
                gap_intent = self._call_intrabar_exit(
                    self.risk_manager, symbol, current_side_order, pos.entry_price, pos.units, row, active_directives
                )
                if gap_intent and "GAP" in gap_intent.reason:
                    order = order_manager.create_from_intent(gap_intent, bar_time, pos.units)
                    fill = self.execution_model.execute_order(order, row)
                    if fill:
                        portfolio.on_fill(fill)
                        active_directives.clear()
                        if hasattr(strategy, "on_fill"):
                            strategy.on_fill(FillView.from_fill(fill))

            # 2. Xử lý pending_intent (Exit hoặc Entry đã phát sinh từ nến trước) tại Open nến này
            if pending_intent is not None:
                if not pos.is_flat and pending_intent.reason == "STRATEGY_EXIT":
                    order = order_manager.create_from_intent(pending_intent, bar_time, pos.units)
                    fill = self.execution_model.execute_order(order, row)
                    if fill:
                        portfolio.on_fill(fill)
                        active_directives.clear()
                        if hasattr(strategy, "on_fill"):
                            strategy.on_fill(FillView.from_fill(fill))
                    # Nếu là REVERSE_CLOSE và có reverse_signal đi kèm, xử lý yêu cầu mở vị thế ngược chiều
                    rev_sig = pending_intent.metadata.get("reverse_signal") if pending_intent.metadata else None
                    if pos.is_flat and rev_sig is not None:
                        rev_intent = translator.translate(rev_sig, pos.to_view())
                        if rev_intent is not None:
                            approved_rev = self._call_approve(
                                self.risk_manager, rev_intent, account.balance, account.equity, open_price
                            )
                            if approved_rev:
                                qty_lots = approved_rev.quantity if approved_rev.quantity is not None else cfg.instrument.lot_size
                                order_units = cfg.instrument.lots_to_units(qty_lots)
                                order = order_manager.create_from_intent(approved_rev, bar_time, order_units)
                                fill = self.execution_model.execute_order(order, row)
                                if fill:
                                    portfolio.on_fill(fill)
                                    active_directives = {
                                        "stop_loss_price": approved_rev.stop_loss_price,
                                        "take_profit_price": approved_rev.take_profit_price,
                                        "stop_loss_pct": approved_rev.stop_loss_pct,
                                        "take_profit_pct": approved_rev.take_profit_pct,
                                    }
                                    if hasattr(strategy, "on_fill"):
                                        strategy.on_fill(FillView.from_fill(fill))
                            elif hasattr(strategy, "on_intent_rejected"):
                                strategy.on_intent_rejected(rev_intent, getattr(self.risk_manager, "last_reject_reason", "RISK_REJECTED"))
                elif pos.is_flat and pending_intent.reason != "STRATEGY_EXIT":
                    approved_intent = self._call_approve(
                        self.risk_manager, pending_intent, account.balance, account.equity, open_price
                    )
                    if approved_intent:
                        qty_lots = approved_intent.quantity if approved_intent.quantity is not None else cfg.instrument.lot_size
                        order_units = cfg.instrument.lots_to_units(qty_lots)
                        order = order_manager.create_from_intent(approved_intent, bar_time, order_units)
                        fill = self.execution_model.execute_order(order, row)
                        if fill:
                            portfolio.on_fill(fill)
                            active_directives = {
                                "stop_loss_price": approved_intent.stop_loss_price,
                                "take_profit_price": approved_intent.take_profit_price,
                                "stop_loss_pct": approved_intent.stop_loss_pct,
                                "take_profit_pct": approved_intent.take_profit_pct,
                            }
                            if hasattr(strategy, "on_fill"):
                                strategy.on_fill(FillView.from_fill(fill))
                    elif hasattr(strategy, "on_intent_rejected"):
                        strategy.on_intent_rejected(pending_intent, getattr(self.risk_manager, "last_reject_reason", "RISK_REJECTED"))
            pending_intent = None

            # 4. Kiểm tra Intrabar SL/TP (giữa nến Low/High)
            if not pos.is_flat:
                current_side_order = OrderSide.BUY if pos.is_long else OrderSide.SELL
                intrabar_intent = self._call_intrabar_exit(
                    self.risk_manager, symbol, current_side_order, pos.entry_price, pos.units, row, active_directives
                )
                if intrabar_intent:
                    order = order_manager.create_from_intent(intrabar_intent, bar_time, pos.units)
                    fill = self.execution_model.execute_order(order, row)
                    if fill:
                        portfolio.on_fill(fill)
                        active_directives.clear()
                        if hasattr(strategy, "on_fill"):
                            strategy.on_fill(FillView.from_fill(fill))

            # 5. Ghi nhận trạng thái Equity cuối nến
            current_equity = portfolio.get_equity({symbol: float(row["close"])})
            position_code = 1 if pos.is_long else (-1 if pos.is_short else 0)
            equity_rows.append({
                "time": row.time,
                "balance": account.balance,
                "equity": current_equity,
                "position": position_code,
            })

            # 6. Tạo quyết định tại giá Close của cây nến hiện tại -> Khớp tại Open nến sau
            if i + 1 < len(df_bars) and i >= warmup - 1:
                sig = self._emit_bar_signal(exec_mode, strategy, signal_list, i, row, df_bars, symbol, pos.to_view())
                pending_intent = translator.translate(sig, pos.to_view())

        # Đóng vị thế bắt buộc ở cây nến cuối cùng nếu có cấu hình
        if not pos.is_flat and cfg.force_close_at_end:
            last = df_bars.iloc[-1]
            last_side = OrderSide.SELL if pos.is_long else OrderSide.BUY
            force_intent = OrderIntent(
                symbol=symbol,
                side=last_side,
                order_type=OrderType.MARKET,
                limit_price=float(last["close"]),
                quantity=cfg.instrument.lot_size,
                reason="END_OF_TEST",
            )
            order = order_manager.create_from_intent(force_intent, pd.Timestamp(last["time"]), pos.units)
            fill = self.execution_model.execute_order(order, last)
            if fill:
                portfolio.on_fill(fill)
                active_directives.clear()
                if hasattr(strategy, "on_fill"):
                    strategy.on_fill(FillView.from_fill(fill))
            equity_rows[-1].update(balance=account.balance, equity=account.balance, position=0)

        equity_df = pd.DataFrame(equity_rows)
        trades_df = pd.DataFrame(portfolio.closed_trades)
        orders_df = order_manager.to_dataframe()

        # Tính toán bộ metrics
        metrics = calculate_metrics(
            {"equity": equity_df, "trades": trades_df, "final_balance": account.balance},
            cfg,
        )

        return BacktestResult(
            equity=equity_df,
            trades=trades_df,
            orders=orders_df,
            metrics=metrics,
            config=cfg,
            strategy_name=strategy.name,
            strategy_parameters=strategy.get_parameters(),
            signal_log=translator.signal_log,
            effective_trade_start=effective_trade_start,
        )
