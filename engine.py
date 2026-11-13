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
from .execution.base import ExecutionModel
from .execution.immediate import ImmediateExecutionModel
from .execution.standard import StandardExecutionModel
from .execution.spread import FixedSpreadModel, SpreadModel, ZeroSpreadModel
from .execution.fee import BaseFeeModel, CompositeFeeModel, FixedCommissionModel, PercentageFeeModel, ZeroFeeModel
from .execution.slippage import BaseSlippageModel, FixedSlippageModel, PercentageSlippageModel, ZeroSlippageModel
from .execution.order_manager import OrderManager
from .risk.base import BaseRiskManager
from .risk.simple import SimpleRiskManager
from .models.order import Order, OrderIntent, OrderOrigin, OrderSide, OrderStatus, OrderType
from .models.quote import QuoteBar
from .models.result import BacktestResult
from .models.signal import Signal, SignalType
from .analytics import calculate_metrics


class BacktestEngine:
    """
    Backtest Engine đóng vai trò là Orchestrator:
    - Điều phối luồng dữ liệu (Data Feed -> Strategy -> RiskManager -> ExecutionModel -> Portfolio).
    - Hoàn toàn Strategy-Agnostic: không chứa logic riêng của bất kỳ chiến lược nào.
    - Hỗ trợ Dependency Injection cho ExecutionModel, SpreadModel và RiskManager.
    - Hỗ trợ cả hai chế độ Strategy: VECTORIZED và EVENT_DRIVEN (cùng BOTH).
    """
    def __init__(
        self,
        config: EngineConfig | None = None,
        execution_model: ExecutionModel | None = None,
        risk_manager: BaseRiskManager | None = None,
        spread_model: SpreadModel | None = None,
    ):
        self.config = config if config is not None else EngineConfig()
        self.spread_model = spread_model if spread_model is not None else self._build_spread_model(self.config)

        # Tiêm phụ thuộc (Dependency Injection) các Model
        if execution_model is not None:
            self.execution_model = execution_model
        else:
            fee_model = self._build_fee_model(self.config)
            slip_model = self._build_slippage_model(self.config)
            self.execution_model = StandardExecutionModel(
                spread_model=self.spread_model,
                fee_model=fee_model,
                slippage_model=slip_model,
                conflict_policy=self.config.intrabar_conflict_policy,
                limit_penetration=self.config.limit_penetration,
                contract_size=self.config.instrument.contract_size,
            )
        self.risk_manager = risk_manager if risk_manager is not None else SimpleRiskManager(
            config=self.config.risk,
            instrument_config=self.config.instrument,
        )

    @staticmethod
    def _build_spread_model(cfg: EngineConfig) -> SpreadModel:
        if cfg.spread.spread_points is not None and cfg.spread.spread_points > 0.0:
            return FixedSpreadModel.from_points(cfg.spread.spread_points, cfg.instrument.point_size)
        if cfg.spread.spread_price > 0.0:
            return FixedSpreadModel(spread_price=cfg.spread.spread_price)
        return ZeroSpreadModel()

    @staticmethod
    def _build_fee_model(cfg: EngineConfig) -> BaseFeeModel:
        models: list[BaseFeeModel] = []
        if cfg.fee.fee_rate_per_side > 0.0:
            models.append(PercentageFeeModel(cfg.fee.fee_rate_per_side, currency=cfg.account.base_currency))
        if cfg.fee.commission_per_lot > 0.0:
            models.append(
                FixedCommissionModel(
                    amount_per_lot=cfg.fee.commission_per_lot,
                    mode=cfg.fee.commission_mode,
                    contract_size=cfg.instrument.contract_size,
                    currency=cfg.account.base_currency,
                )
            )
        if not models:
            return ZeroFeeModel(currency=cfg.account.base_currency)
        if len(models) == 1:
            return models[0]
        return CompositeFeeModel(models, currency=cfg.account.base_currency)

    @staticmethod
    def _build_slippage_model(cfg: EngineConfig) -> BaseSlippageModel:
        if cfg.slippage.slippage_rate > 0.0:
            return PercentageSlippageModel(cfg.slippage.slippage_rate)
        if cfg.slippage.slippage_price > 0.0:
            return FixedSlippageModel(cfg.slippage.slippage_price)
        return ZeroSlippageModel()

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
        ref_price: float | None,
    ) -> OrderIntent | None:
        try:
            return risk_manager.approve_order_intent(intent, balance, equity, reference_price=ref_price)
        except TypeError:
            return risk_manager.approve_order_intent(intent, balance, equity)

    @staticmethod
    def _finalize_sl_tp(
        risk_manager: BaseRiskManager,
        order: Order,
        proposed_fill_price: float,
    ) -> tuple[bool, str, float | None, float | None]:
        if hasattr(risk_manager, "finalize_entry_sl_tp"):
            return risk_manager.finalize_entry_sl_tp(order, proposed_fill_price)
        sl_p = order.stop_loss_price
        tp_p = order.take_profit_price
        if sl_p is None and order.stop_loss_pct:
            sl_p = proposed_fill_price * (1.0 - order.stop_loss_pct) if order.side == OrderSide.BUY else proposed_fill_price * (1.0 + order.stop_loss_pct)
        if tp_p is None and order.take_profit_pct:
            tp_p = proposed_fill_price * (1.0 + order.take_profit_pct) if order.side == OrderSide.BUY else proposed_fill_price * (1.0 - order.take_profit_pct)
        return True, "OK", sl_p, tp_p

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

    def _submit_intent(
        self,
        intent: OrderIntent,
        bar_time: pd.Timestamp,
        bar_index: int,
        eligible_from_bar: int,
        default_ref_price: float,
        pos_units: float,
        account_balance: float,
        current_equity: float,
        order_manager: OrderManager,
        strategy: Strategy,
    ) -> Order | None:
        cfg = self.config
        if intent.origin == OrderOrigin.STRATEGY_EXIT or intent.reason == "STRATEGY_EXIT":
            qty_lots = pos_units / cfg.instrument.contract_size if cfg.instrument.contract_size > 0 else cfg.instrument.lot_size
            exit_intent = OrderIntent(
                symbol=intent.symbol,
                side=intent.side,
                order_type=OrderType.MARKET,
                quantity=qty_lots,
                origin=OrderOrigin.STRATEGY_EXIT,
                reason="STRATEGY_EXIT",
                metadata=dict(intent.metadata) if intent.metadata else {},
            )
            return order_manager.create_from_intent(
                exit_intent,
                time=bar_time,
                units=pos_units,
                bar_index=bar_index,
                eligible_from_bar=eligible_from_bar,
            )

        if intent.order_type == OrderType.LIMIT and intent.limit_price is not None:
            ref_price = intent.limit_price
        elif intent.order_type == OrderType.STOP_MARKET and intent.stop_price is not None:
            ref_price = intent.stop_price
        else:
            ref_price = default_ref_price

        approved_intent = self._call_approve(
            self.risk_manager, intent, account_balance, current_equity, ref_price
        )
        if approved_intent is not None:
            qty_lots = approved_intent.quantity if approved_intent.quantity is not None else cfg.instrument.lot_size
            order_units = cfg.instrument.lots_to_units(qty_lots)
            return order_manager.create_from_intent(
                approved_intent,
                time=bar_time,
                units=order_units,
                bar_index=bar_index,
                eligible_from_bar=eligible_from_bar,
            )

        reject_reason = getattr(self.risk_manager, "last_reject_reason", None) or "RISK_REJECTED"
        raw_qty = intent.quantity if isinstance(intent.quantity, (int, float)) and intent.quantity > 0 else cfg.instrument.lot_size
        raw_units = cfg.instrument.lots_to_units(raw_qty)
        order_manager.create_rejected_order(
            time=bar_time,
            symbol=intent.symbol,
            side=intent.side,
            order_type=intent.order_type,
            quantity=float(raw_qty),
            units=float(raw_units),
            limit_price=intent.limit_price,
            stop_price=intent.stop_price,
            bar_index=bar_index,
            reject_reason=reject_reason,
            origin=OrderOrigin.STRATEGY_ENTRY,
        )
        if hasattr(strategy, "on_intent_rejected"):
            strategy.on_intent_rejected(intent, reject_reason)
        return None

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

        if not isinstance(self.execution_model, StandardExecutionModel):
            return self._run_custom_execution_model(
                df_bars=df_bars,
                strategy=strategy,
                exec_mode=exec_mode,
                signal_list=signal_list,
                warmup=warmup,
                effective_trade_start=effective_trade_start,
            )

        # ---------------------------------------------------------------------
        # STANDARD 3-PHASE BAR EXECUTION PIPELINE (GROUP D V1.0)
        # ---------------------------------------------------------------------
        exec_model: StandardExecutionModel = self.execution_model
        spread_model = exec_model.spread_model if hasattr(exec_model, "spread_model") else self.spread_model
        quote_df = spread_model.to_quote_frame(df_bars, price_type=cfg.spread.price_type, symbol=symbol)
        quote_bars = [QuoteBar.from_series(quote_df.iloc[idx], symbol=symbol) for idx in range(len(quote_df))]

        account = Account(initial_balance=cfg.account.initial_balance, base_currency=cfg.account.base_currency)
        portfolio = Portfolio(account=account)
        order_manager = OrderManager()
        translator = SignalTranslator(
            default_quantity=cfg.instrument.lot_size,
            reverse_policy=cfg.reverse_policy,
        )

        pos = portfolio.get_position(symbol)
        pending_exit_order: Order | None = None
        equity_rows: list[dict[str, Any]] = []

        for i, row in df_bars.iterrows():
            quote_bar = quote_bars[i]
            bar_time = quote_bar.time

            if i < effective_trade_start:
                equity_rows.append({
                    "time": row.time,
                    "balance": account.balance,
                    "equity": account.balance,
                    "position": 0,
                })
                if i == effective_trade_start - 1 and i >= warmup - 1 and i + 1 < len(df_bars):
                    sig = self._emit_bar_signal(exec_mode, strategy, signal_list, i, row, df_bars, symbol, pos.to_view())
                    intent = translator.translate(sig, pos.to_view())
                    if intent is not None:
                        ref_close = quote_bar.ask_close if intent.side == OrderSide.BUY else quote_bar.bid_close
                        created = self._submit_intent(
                            intent=intent,
                            bar_time=bar_time,
                            bar_index=i,
                            eligible_from_bar=i + 1,
                            default_ref_price=ref_close,
                            pos_units=pos.units,
                            account_balance=account.balance,
                            current_equity=account.balance,
                            order_manager=order_manager,
                            strategy=strategy,
                        )
                        if created is not None and created.origin == OrderOrigin.STRATEGY_EXIT:
                            pending_exit_order = created
                continue

            # =================================================================
            # PHASE 1: OPEN PROCESSING
            # =================================================================
            # 1.1 Protective SL/TP Gap at Open (for positions opened on prior bars)
            if not pos.is_flat:
                gap_prop = exec_model.propose_protective_open(pos, quote_bar, bar_index=i)
                if gap_prop is not None:
                    fill = exec_model.commit_proposal(gap_prop, order_manager, portfolio)
                    if hasattr(strategy, "on_fill"):
                        strategy.on_fill(FillView.from_fill(fill))
                    # Cancel stale pending strategy exit order if position was just closed by Gap SL/TP
                    if pending_exit_order is not None and pending_exit_order.status == OrderStatus.PENDING:
                        order_manager.cancel_order(
                            pending_exit_order,
                            timestamp=bar_time,
                            bar_index=i,
                            reason="POSITION_ALREADY_CLOSED",
                        )
                        pending_exit_order = None

            # 1.2 Pending STRATEGY_EXIT Order at Open
            if pending_exit_order is not None:
                if pending_exit_order.status == OrderStatus.PENDING:
                    if pos.is_flat:
                        order_manager.cancel_order(
                            pending_exit_order,
                            timestamp=bar_time,
                            bar_index=i,
                            reason="POSITION_ALREADY_CLOSED",
                        )
                    else:
                        exit_prop = exec_model.propose_entry_open(pending_exit_order, quote_bar, bar_index=i)
                        if exit_prop is not None:
                            fill = exec_model.commit_proposal(exit_prop, order_manager, portfolio)
                            if hasattr(strategy, "on_fill"):
                                strategy.on_fill(FillView.from_fill(fill))
                            # If REVERSE policy attached reverse_signal, create the opposite entry order now
                            rev_sig = pending_exit_order.metadata.get("reverse_signal") if pending_exit_order.metadata else None
                            if pos.is_flat and rev_sig is not None:
                                rev_intent = translator.translate(rev_sig, pos.to_view())
                                if rev_intent is not None:
                                    ref_open = quote_bar.ask_open if rev_intent.side == OrderSide.BUY else quote_bar.bid_open
                                    self._submit_intent(
                                        intent=rev_intent,
                                        bar_time=bar_time,
                                        bar_index=i,
                                        eligible_from_bar=i,
                                        default_ref_price=ref_open,
                                        pos_units=pos.units,
                                        account_balance=account.balance,
                                        current_equity=account.balance,
                                        order_manager=order_manager,
                                        strategy=strategy,
                                    )
                pending_exit_order = None

            # 1.3 Pending STRATEGY_ENTRY Order at Open (Market, Marketable Limit, or Gapped Stop)
            entry_order = order_manager.get_pending_entry_order(symbol)
            if pos.is_flat and entry_order is not None:
                entry_open_prop = exec_model.propose_entry_open(entry_order, quote_bar, bar_index=i)
                if entry_open_prop is not None:
                    ok, reason, sl_p, tp_p = self._finalize_sl_tp(
                        self.risk_manager, entry_order, entry_open_prop.fill_price
                    )
                    if ok:
                        fill = exec_model.commit_proposal(
                            entry_open_prop,
                            order_manager,
                            portfolio,
                            stop_loss_price=sl_p,
                            take_profit_price=tp_p,
                        )
                        if hasattr(strategy, "on_fill"):
                            strategy.on_fill(FillView.from_fill(fill))
                    else:
                        order_manager.reject_pending_order(
                            entry_order, timestamp=bar_time, bar_index=i, reason=reason
                        )
                        if hasattr(strategy, "on_intent_rejected"):
                            strategy.on_intent_rejected(entry_order, reason)

            # =================================================================
            # PHASE 2: RANGE PROCESSING
            # =================================================================
            # 2.1 Protective SL/TP in Range (for positions open prior to Range)
            if not pos.is_flat:
                range_exit_prop = exec_model.propose_protective_range(pos, quote_bar, bar_index=i)
                if range_exit_prop is not None:
                    fill = exec_model.commit_proposal(range_exit_prop, order_manager, portfolio)
                    if hasattr(strategy, "on_fill"):
                        strategy.on_fill(FillView.from_fill(fill))

            # 2.2 Pending LIMIT / STOP_MARKET Entry in Range (when flat)
            # Positions opened here get opened_phase=RANGE and are NOT evaluated for SL/TP in this same bar!
            entry_order = order_manager.get_pending_entry_order(symbol)
            if pos.is_flat and entry_order is not None:
                range_entry_prop = exec_model.propose_entry_range(entry_order, quote_bar, bar_index=i)
                if range_entry_prop is not None:
                    ok, reason, sl_p, tp_p = self._finalize_sl_tp(
                        self.risk_manager, entry_order, range_entry_prop.fill_price
                    )
                    if ok:
                        fill = exec_model.commit_proposal(
                            range_entry_prop,
                            order_manager,
                            portfolio,
                            stop_loss_price=sl_p,
                            take_profit_price=tp_p,
                        )
                        if hasattr(strategy, "on_fill"):
                            strategy.on_fill(FillView.from_fill(fill))
                    else:
                        order_manager.reject_pending_order(
                            entry_order, timestamp=bar_time, bar_index=i, reason=reason
                        )
                        if hasattr(strategy, "on_intent_rejected"):
                            strategy.on_intent_rejected(entry_order, reason)

            # =================================================================
            # PHASE 3: CLOSE PROCESSING & MARK-TO-MARKET
            # =================================================================
            # 3.1 Check Order Expirations after Bar i
            next_bar_time = quote_bars[i + 1].time if i + 1 < len(quote_bars) else None
            order_manager.check_expirations_after_bar(
                bar_index=i, timestamp=bar_time, next_timestamp=next_bar_time
            )

            # 3.2 Mark-to-Market Equity at Bar Close (Long @ Bid Close, Short @ Ask Close)
            current_equity = portfolio.get_equity({symbol: quote_bar})
            position_code = 1 if pos.is_long else (-1 if pos.is_short else 0)
            equity_rows.append({
                "time": row.time,
                "balance": account.balance,
                "equity": current_equity,
                "position": position_code,
            })

            # 3.3 Emit Strategy Signal at Bar Close -> Create Pending Order eligible from Bar i + 1
            if i + 1 < len(df_bars) and i >= warmup - 1:
                sig = self._emit_bar_signal(exec_mode, strategy, signal_list, i, row, df_bars, symbol, pos.to_view())
                intent = translator.translate(sig, pos.to_view())
                if intent is not None:
                    ref_close = quote_bar.ask_close if intent.side == OrderSide.BUY else quote_bar.bid_close
                    created = self._submit_intent(
                        intent=intent,
                        bar_time=bar_time,
                        bar_index=i,
                        eligible_from_bar=i + 1,
                        default_ref_price=ref_close,
                        pos_units=pos.units,
                        account_balance=account.balance,
                        current_equity=current_equity,
                        order_manager=order_manager,
                        strategy=strategy,
                    )
                    if created is not None and created.origin == OrderOrigin.STRATEGY_EXIT:
                        pending_exit_order = created

        # =====================================================================
        # END OF TEST: EXPIRE PENDING ORDERS & FORCE CLOSE OPEN POSITION
        # =====================================================================
        if len(quote_bars) > 0:
            last_idx = len(quote_bars) - 1
            last_quote = quote_bars[last_idx]
            order_manager.expire_all_pending(
                timestamp=last_quote.time, bar_index=last_idx, reason="END_OF_TEST"
            )
            if not pos.is_flat and cfg.force_close_at_end:
                eot_prop = exec_model.propose_end_of_test_close(pos, last_quote, bar_index=last_idx)
                if eot_prop is not None:
                    fill = exec_model.commit_proposal(eot_prop, order_manager, portfolio)
                    if hasattr(strategy, "on_fill"):
                        strategy.on_fill(FillView.from_fill(fill))
                equity_rows[-1].update(balance=account.balance, equity=account.balance, position=0)

        equity_df = pd.DataFrame(equity_rows)
        trades_df = pd.DataFrame(portfolio.closed_trades)
        orders_df = order_manager.to_dataframe()
        events_df = order_manager.events_to_dataframe()

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
            order_events=events_df,
        )

    def _run_custom_execution_model(
        self,
        df_bars: pd.DataFrame,
        strategy: Strategy,
        exec_mode: StrategyMode,
        signal_list: list[Signal] | None,
        warmup: int,
        effective_trade_start: int,
    ) -> BacktestResult:
        """
        Nhánh tương thích cho các custom ExecutionModel được inject trực tiếp
        (ví dụ: ImmediateExecutionModel hoặc MockExecutionModel trong các bài test DI).
        """
        cfg = self.config
        symbol = cfg.instrument.symbol
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
        equity_rows: list[dict[str, Any]] = []

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

            if not pos.is_flat and current_side_order is not None:
                gap_intent = self._call_intrabar_exit(
                    self.risk_manager, symbol, current_side_order, pos.entry_price, pos.units, row, active_directives
                )
                if gap_intent and "GAP" in gap_intent.reason:
                    order = order_manager.create_from_intent(gap_intent, bar_time, pos.units, bar_index=i)
                    fill = self.execution_model.execute_order(order, row)
                    if fill:
                        portfolio.on_fill(fill)
                        active_directives.clear()
                        if hasattr(strategy, "on_fill"):
                            strategy.on_fill(FillView.from_fill(fill))

            if pending_intent is not None:
                if not pos.is_flat and pending_intent.reason == "STRATEGY_EXIT":
                    order = order_manager.create_from_intent(pending_intent, bar_time, pos.units, bar_index=i)
                    fill = self.execution_model.execute_order(order, row)
                    if fill:
                        portfolio.on_fill(fill)
                        active_directives.clear()
                        if hasattr(strategy, "on_fill"):
                            strategy.on_fill(FillView.from_fill(fill))
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
                                order = order_manager.create_from_intent(approved_rev, bar_time, order_units, bar_index=i)
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
                        order = order_manager.create_from_intent(approved_intent, bar_time, order_units, bar_index=i)
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

            if not pos.is_flat:
                current_side_order = OrderSide.BUY if pos.is_long else OrderSide.SELL
                intrabar_intent = self._call_intrabar_exit(
                    self.risk_manager, symbol, current_side_order, pos.entry_price, pos.units, row, active_directives
                )
                if intrabar_intent:
                    order = order_manager.create_from_intent(intrabar_intent, bar_time, pos.units, bar_index=i)
                    fill = self.execution_model.execute_order(order, row)
                    if fill:
                        portfolio.on_fill(fill)
                        active_directives.clear()
                        if hasattr(strategy, "on_fill"):
                            strategy.on_fill(FillView.from_fill(fill))

            current_equity = portfolio.get_equity({symbol: float(row["close"])})
            position_code = 1 if pos.is_long else (-1 if pos.is_short else 0)
            equity_rows.append({
                "time": row.time,
                "balance": account.balance,
                "equity": current_equity,
                "position": position_code,
            })

            if i + 1 < len(df_bars) and i >= warmup - 1:
                sig = self._emit_bar_signal(exec_mode, strategy, signal_list, i, row, df_bars, symbol, pos.to_view())
                pending_intent = translator.translate(sig, pos.to_view())

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
            order = order_manager.create_from_intent(force_intent, pd.Timestamp(last["time"]), pos.units, bar_index=len(df_bars) - 1)
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
        events_df = order_manager.events_to_dataframe()

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
            order_events=events_df,
        )
