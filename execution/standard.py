from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd

from quant_backtester.config import IntrabarConflictPolicy
from quant_backtester.execution.base import ExecutionModel
from quant_backtester.execution.fee import BaseFeeModel, PercentageFeeModel
from quant_backtester.execution.order_manager import OrderManager
from quant_backtester.execution.slippage import BaseSlippageModel, ZeroSlippageModel
from quant_backtester.execution.spread import SpreadModel, ZeroSpreadModel
from quant_backtester.models.fill import Fill
from quant_backtester.models.order import (
    ExecutionPhase,
    Order,
    OrderOrigin,
    OrderSide,
    OrderStatus,
    OrderType,
)
from quant_backtester.models.quote import QuoteBar
from quant_backtester.portfolio.portfolio import Portfolio
from quant_backtester.portfolio.position import Position


@dataclass(frozen=True)
class ExecutionProposal:
    """
    Đề xuất khớp lệnh tại Giai đoạn 1 (Propose) trước khi Validate/Finalize và Commit.
    Hoàn toàn bất biến và chưa làm thay đổi Order status, Position, Account hay Ledger.
    """
    symbol: str
    time: pd.Timestamp
    bar_index: int
    phase: ExecutionPhase
    side: OrderSide
    quantity: float
    units: float
    base_price: float
    fill_price: float
    slippage: float
    origin: OrderOrigin
    reason: str
    order: Order | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StandardExecutionModel(ExecutionModel):
    """
    Mô hình khớp lệnh tiêu chuẩn V1.0 cho dữ liệu OHLC Bid/Ask:
    - Hỗ trợ MARKET, LIMIT (kèm limit_penetration), STOP_MARKET.
    - Hỗ trợ Bid/Ask convention: Mở BUY @ Ask, Đóng BUY @ Bid; Mở SELL @ Bid, Đóng SELL @ Ask.
    - Áp dụng Slippage cho MARKET, STOP_MARKET, PROTECTIVE_SL (kể cả SL Gap) và END_OF_TEST.
    - Không áp dụng Slippage cho LIMIT và PROTECTIVE_TP (kể cả TP Gap).
    - Hỗ trợ đánh giá Protective SL/TP tại Open (Gap) và trong Range (kèm IntrabarConflictPolicy).
    - Hỗ trợ quy trình Two-Phase Fill: Propose -> Validate/Finalize -> Commit.
    """

    def __init__(
        self,
        spread_model: SpreadModel | None = None,
        fee_model: BaseFeeModel | None = None,
        slippage_model: BaseSlippageModel | None = None,
        conflict_policy: IntrabarConflictPolicy = IntrabarConflictPolicy.WORST_CASE_SL_FIRST,
        limit_penetration: float = 0.0,
        contract_size: float = 100.0,
    ):
        self.spread_model = spread_model if spread_model is not None else ZeroSpreadModel()
        self.fee_model = fee_model if fee_model is not None else PercentageFeeModel(0.00015)
        self.slippage_model = slippage_model if slippage_model is not None else ZeroSlippageModel()
        self.conflict_policy = conflict_policy
        self.limit_penetration = float(limit_penetration)
        self.contract_size = float(contract_size) if contract_size > 0 else 100.0
        self._fill_counter = 0

    def _apply_slippage(self, base_price: float, side: OrderSide) -> tuple[float, float]:
        fill_p = self.slippage_model.calculate_fill_price(base_price, side)
        slip = abs(fill_p - base_price)
        return fill_p, slip

    # -------------------------------------------------------------------------
    # 1. PROPOSE ENTRY / STRATEGY EXIT AT OPEN PHASE
    # -------------------------------------------------------------------------
    def propose_entry_open(self, order: Order, quote_bar: QuoteBar, bar_index: int) -> ExecutionProposal | None:
        if order.status != OrderStatus.PENDING:
            return None
        if bar_index < order.eligible_from_bar:
            return None

        # MARKET order
        if order.order_type == OrderType.MARKET:
            base_p = quote_bar.ask_open if order.side == OrderSide.BUY else quote_bar.bid_open
            fill_p, slip = self._apply_slippage(base_p, order.side)
            return ExecutionProposal(
                symbol=order.symbol,
                time=quote_bar.time,
                bar_index=bar_index,
                phase=ExecutionPhase.OPEN,
                side=order.side,
                quantity=order.quantity,
                units=order.units,
                base_price=base_p,
                fill_price=fill_p,
                slippage=slip,
                origin=order.origin,
                reason=order.reason,
                order=order,
            )

        # Marketable LIMIT order at Open (favorable gap across limit price)
        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            lim = order.limit_price
            if order.side == OrderSide.BUY and quote_bar.ask_open <= lim + 1e-12:
                base_p = quote_bar.ask_open
                return ExecutionProposal(
                    symbol=order.symbol,
                    time=quote_bar.time,
                    bar_index=bar_index,
                    phase=ExecutionPhase.OPEN,
                    side=order.side,
                    quantity=order.quantity,
                    units=order.units,
                    base_price=base_p,
                    fill_price=base_p,  # No adverse slippage on Limit
                    slippage=0.0,
                    origin=order.origin,
                    reason=order.reason,
                    order=order,
                )
            if order.side == OrderSide.SELL and quote_bar.bid_open >= lim - 1e-12:
                base_p = quote_bar.bid_open
                return ExecutionProposal(
                    symbol=order.symbol,
                    time=quote_bar.time,
                    bar_index=bar_index,
                    phase=ExecutionPhase.OPEN,
                    side=order.side,
                    quantity=order.quantity,
                    units=order.units,
                    base_price=base_p,
                    fill_price=base_p,  # No adverse slippage on Limit
                    slippage=0.0,
                    origin=order.origin,
                    reason=order.reason,
                    order=order,
                )

        # STOP_MARKET order gapped across stop_price at Open
        if order.order_type == OrderType.STOP_MARKET and order.stop_price is not None:
            stp = order.stop_price
            if order.side == OrderSide.BUY and quote_bar.ask_open >= stp - 1e-12:
                base_p = quote_bar.ask_open
                fill_p, slip = self._apply_slippage(base_p, order.side)
                return ExecutionProposal(
                    symbol=order.symbol,
                    time=quote_bar.time,
                    bar_index=bar_index,
                    phase=ExecutionPhase.OPEN,
                    side=order.side,
                    quantity=order.quantity,
                    units=order.units,
                    base_price=base_p,
                    fill_price=fill_p,
                    slippage=slip,
                    origin=order.origin,
                    reason=order.reason,
                    order=order,
                )
            if order.side == OrderSide.SELL and quote_bar.bid_open <= stp + 1e-12:
                base_p = quote_bar.bid_open
                fill_p, slip = self._apply_slippage(base_p, order.side)
                return ExecutionProposal(
                    symbol=order.symbol,
                    time=quote_bar.time,
                    bar_index=bar_index,
                    phase=ExecutionPhase.OPEN,
                    side=order.side,
                    quantity=order.quantity,
                    units=order.units,
                    base_price=base_p,
                    fill_price=fill_p,
                    slippage=slip,
                    origin=order.origin,
                    reason=order.reason,
                    order=order,
                )

        return None

    # -------------------------------------------------------------------------
    # 2. PROPOSE ENTRY IN RANGE PHASE (LIMIT & STOP_MARKET)
    # -------------------------------------------------------------------------
    def propose_entry_range(self, order: Order, quote_bar: QuoteBar, bar_index: int) -> ExecutionProposal | None:
        if order.status != OrderStatus.PENDING:
            return None
        if bar_index < order.eligible_from_bar:
            return None

        pen = self.limit_penetration

        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            lim = order.limit_price
            if order.side == OrderSide.BUY:
                # Open was above limit; check if Ask Low reaches (lim - penetration)
                if quote_bar.ask_open <= lim + 1e-12:
                    base_p = quote_bar.ask_open
                elif quote_bar.ask_low <= (lim - pen) + 1e-12:
                    base_p = lim
                else:
                    return None
                return ExecutionProposal(
                    symbol=order.symbol,
                    time=quote_bar.time,
                    bar_index=bar_index,
                    phase=ExecutionPhase.RANGE,
                    side=order.side,
                    quantity=order.quantity,
                    units=order.units,
                    base_price=base_p,
                    fill_price=base_p,
                    slippage=0.0,
                    origin=order.origin,
                    reason=order.reason,
                    order=order,
                )

            if order.side == OrderSide.SELL:
                if quote_bar.bid_open >= lim - 1e-12:
                    base_p = quote_bar.bid_open
                elif quote_bar.bid_high >= (lim + pen) - 1e-12:
                    base_p = lim
                else:
                    return None
                return ExecutionProposal(
                    symbol=order.symbol,
                    time=quote_bar.time,
                    bar_index=bar_index,
                    phase=ExecutionPhase.RANGE,
                    side=order.side,
                    quantity=order.quantity,
                    units=order.units,
                    base_price=base_p,
                    fill_price=base_p,
                    slippage=0.0,
                    origin=order.origin,
                    reason=order.reason,
                    order=order,
                )

        if order.order_type == OrderType.STOP_MARKET and order.stop_price is not None:
            stp = order.stop_price
            if order.side == OrderSide.BUY:
                if quote_bar.ask_open >= stp - 1e-12:
                    base_p = quote_bar.ask_open
                elif quote_bar.ask_high >= stp - 1e-12:
                    base_p = stp
                else:
                    return None
                fill_p, slip = self._apply_slippage(base_p, order.side)
                return ExecutionProposal(
                    symbol=order.symbol,
                    time=quote_bar.time,
                    bar_index=bar_index,
                    phase=ExecutionPhase.RANGE,
                    side=order.side,
                    quantity=order.quantity,
                    units=order.units,
                    base_price=base_p,
                    fill_price=fill_p,
                    slippage=slip,
                    origin=order.origin,
                    reason=order.reason,
                    order=order,
                )

            if order.side == OrderSide.SELL:
                if quote_bar.bid_open <= stp + 1e-12:
                    base_p = quote_bar.bid_open
                elif quote_bar.bid_low <= stp + 1e-12:
                    base_p = stp
                else:
                    return None
                fill_p, slip = self._apply_slippage(base_p, order.side)
                return ExecutionProposal(
                    symbol=order.symbol,
                    time=quote_bar.time,
                    bar_index=bar_index,
                    phase=ExecutionPhase.RANGE,
                    side=order.side,
                    quantity=order.quantity,
                    units=order.units,
                    base_price=base_p,
                    fill_price=fill_p,
                    slippage=slip,
                    origin=order.origin,
                    reason=order.reason,
                    order=order,
                )

        return None

    # -------------------------------------------------------------------------
    # 3. PROPOSE PROTECTIVE SL/TP AT OPEN (GAP) AND IN RANGE
    # -------------------------------------------------------------------------
    def propose_protective_open(self, position: Position, quote_bar: QuoteBar, bar_index: int) -> ExecutionProposal | None:
        if position.is_flat:
            return None
        # Chỉ những vị thế đã mở từ các bar trước mới có thể bị Gap qua SL/TP tại Open của bar hiện tại
        if position.opened_bar_index is not None and position.opened_bar_index >= bar_index:
            return None

        sl = position.stop_loss_price
        tp = position.take_profit_price
        qty_lots = position.units / self.contract_size

        if position.is_long:
            # Long closes at BID
            if sl is not None and quote_bar.bid_open <= sl + 1e-12:
                base_p = quote_bar.bid_open
                fill_p, slip = self._apply_slippage(base_p, OrderSide.SELL)
                return ExecutionProposal(
                    symbol=position.symbol,
                    time=quote_bar.time,
                    bar_index=bar_index,
                    phase=ExecutionPhase.OPEN,
                    side=OrderSide.SELL,
                    quantity=qty_lots,
                    units=position.units,
                    base_price=base_p,
                    fill_price=fill_p,
                    slippage=slip,
                    origin=OrderOrigin.PROTECTIVE_SL,
                    reason="STOP_GAP",
                )
            if tp is not None and quote_bar.bid_open >= tp - 1e-12:
                base_p = quote_bar.bid_open
                return ExecutionProposal(
                    symbol=position.symbol,
                    time=quote_bar.time,
                    bar_index=bar_index,
                    phase=ExecutionPhase.OPEN,
                    side=OrderSide.SELL,
                    quantity=qty_lots,
                    units=position.units,
                    base_price=base_p,
                    fill_price=base_p,  # TP does not suffer slippage
                    slippage=0.0,
                    origin=OrderOrigin.PROTECTIVE_TP,
                    reason="TAKE_PROFIT_GAP",
                )

        elif position.is_short:
            # Short closes at ASK
            if sl is not None and quote_bar.ask_open >= sl - 1e-12:
                base_p = quote_bar.ask_open
                fill_p, slip = self._apply_slippage(base_p, OrderSide.BUY)
                return ExecutionProposal(
                    symbol=position.symbol,
                    time=quote_bar.time,
                    bar_index=bar_index,
                    phase=ExecutionPhase.OPEN,
                    side=OrderSide.BUY,
                    quantity=qty_lots,
                    units=position.units,
                    base_price=base_p,
                    fill_price=fill_p,
                    slippage=slip,
                    origin=OrderOrigin.PROTECTIVE_SL,
                    reason="STOP_GAP",
                )
            if tp is not None and quote_bar.ask_open <= tp + 1e-12:
                base_p = quote_bar.ask_open
                return ExecutionProposal(
                    symbol=position.symbol,
                    time=quote_bar.time,
                    bar_index=bar_index,
                    phase=ExecutionPhase.OPEN,
                    side=OrderSide.BUY,
                    quantity=qty_lots,
                    units=position.units,
                    base_price=base_p,
                    fill_price=base_p,  # TP does not suffer slippage
                    slippage=0.0,
                    origin=OrderOrigin.PROTECTIVE_TP,
                    reason="TAKE_PROFIT_GAP",
                )

        return None

    def propose_protective_range(self, position: Position, quote_bar: QuoteBar, bar_index: int) -> ExecutionProposal | None:
        if position.is_flat:
            return None

        # D8 Rule: Position mở giữa nến trong RANGE phase không được đánh giá SL/TP trong chính RANGE phase đó!
        if position.opened_bar_index == bar_index and position.opened_phase == ExecutionPhase.RANGE:
            return None

        sl = position.stop_loss_price
        tp = position.take_profit_price
        qty_lots = position.units / self.contract_size

        if position.is_long:
            sl_hit = sl is not None and quote_bar.bid_low <= sl + 1e-12
            tp_hit = tp is not None and quote_bar.bid_high >= tp - 1e-12
            if not sl_hit and not tp_hit:
                return None

            ambiguous = bool(sl_hit and tp_hit)
            meta = {"ambiguous_sl_tp": True} if ambiguous else {}

            choose_sl = sl_hit and (not tp_hit or self.conflict_policy == IntrabarConflictPolicy.WORST_CASE_SL_FIRST)
            if choose_sl:
                assert sl is not None
                base_p = min(quote_bar.bid_open, sl)
                fill_p, slip = self._apply_slippage(base_p, OrderSide.SELL)
                return ExecutionProposal(
                    symbol=position.symbol,
                    time=quote_bar.time,
                    bar_index=bar_index,
                    phase=ExecutionPhase.RANGE,
                    side=OrderSide.SELL,
                    quantity=qty_lots,
                    units=position.units,
                    base_price=base_p,
                    fill_price=fill_p,
                    slippage=slip,
                    origin=OrderOrigin.PROTECTIVE_SL,
                    reason="STOP_LOSS",
                    metadata=meta,
                )
            else:
                assert tp is not None
                base_p = max(quote_bar.bid_open, tp)
                return ExecutionProposal(
                    symbol=position.symbol,
                    time=quote_bar.time,
                    bar_index=bar_index,
                    phase=ExecutionPhase.RANGE,
                    side=OrderSide.SELL,
                    quantity=qty_lots,
                    units=position.units,
                    base_price=base_p,
                    fill_price=base_p,
                    slippage=0.0,
                    origin=OrderOrigin.PROTECTIVE_TP,
                    reason="TAKE_PROFIT",
                    metadata=meta,
                )

        elif position.is_short:
            sl_hit = sl is not None and quote_bar.ask_high >= sl - 1e-12
            tp_hit = tp is not None and quote_bar.ask_low <= tp + 1e-12
            if not sl_hit and not tp_hit:
                return None

            ambiguous = bool(sl_hit and tp_hit)
            meta = {"ambiguous_sl_tp": True} if ambiguous else {}

            choose_sl = sl_hit and (not tp_hit or self.conflict_policy == IntrabarConflictPolicy.WORST_CASE_SL_FIRST)
            if choose_sl:
                assert sl is not None
                base_p = max(quote_bar.ask_open, sl)
                fill_p, slip = self._apply_slippage(base_p, OrderSide.BUY)
                return ExecutionProposal(
                    symbol=position.symbol,
                    time=quote_bar.time,
                    bar_index=bar_index,
                    phase=ExecutionPhase.RANGE,
                    side=OrderSide.BUY,
                    quantity=qty_lots,
                    units=position.units,
                    base_price=base_p,
                    fill_price=fill_p,
                    slippage=slip,
                    origin=OrderOrigin.PROTECTIVE_SL,
                    reason="STOP_LOSS",
                    metadata=meta,
                )
            else:
                assert tp is not None
                base_p = min(quote_bar.ask_open, tp)
                return ExecutionProposal(
                    symbol=position.symbol,
                    time=quote_bar.time,
                    bar_index=bar_index,
                    phase=ExecutionPhase.RANGE,
                    side=OrderSide.BUY,
                    quantity=qty_lots,
                    units=position.units,
                    base_price=base_p,
                    fill_price=base_p,
                    slippage=0.0,
                    origin=OrderOrigin.PROTECTIVE_TP,
                    reason="TAKE_PROFIT",
                    metadata=meta,
                )

        return None

    # -------------------------------------------------------------------------
    # 4. PROPOSE END_OF_TEST CLOSE
    # -------------------------------------------------------------------------
    def propose_end_of_test_close(self, position: Position, quote_bar: QuoteBar, bar_index: int) -> ExecutionProposal | None:
        if position.is_flat:
            return None
        close_side = OrderSide.SELL if position.is_long else OrderSide.BUY
        base_p = quote_bar.bid_close if position.is_long else quote_bar.ask_close
        fill_p, slip = self._apply_slippage(base_p, close_side)
        qty_lots = position.units / self.contract_size
        return ExecutionProposal(
            symbol=position.symbol,
            time=quote_bar.time,
            bar_index=bar_index,
            phase=ExecutionPhase.CLOSE,
            side=close_side,
            quantity=qty_lots,
            units=position.units,
            base_price=base_p,
            fill_price=fill_p,
            slippage=slip,
            origin=OrderOrigin.END_OF_TEST,
            reason="END_OF_TEST",
        )

    # -------------------------------------------------------------------------
    # 5. COMMIT PROPOSAL (PHASE 3 OF TWO-PHASE FILL)
    # -------------------------------------------------------------------------
    def commit_proposal(
        self,
        proposal: ExecutionProposal,
        order_manager: OrderManager,
        portfolio: Portfolio,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> Fill:
        pos = portfolio.get_position(proposal.symbol)
        is_opening = (
            pos.is_flat
            or (pos.is_long and proposal.side == OrderSide.BUY)
            or (pos.is_short and proposal.side == OrderSide.SELL)
        )

        order = proposal.order
        if order is None:
            # System-generated protective exit or END_OF_TEST order for full audit trail
            sys_otype = OrderType.LIMIT if proposal.origin == OrderOrigin.PROTECTIVE_TP else OrderType.MARKET
            order = order_manager.create_order(
                time=proposal.time,
                symbol=proposal.symbol,
                side=proposal.side,
                order_type=sys_otype,
                quantity=proposal.quantity,
                units=proposal.units,
                limit_price=proposal.base_price if sys_otype == OrderType.LIMIT else None,
                reason=proposal.reason,
                bar_index=proposal.bar_index,
                eligible_from_bar=proposal.bar_index,
                origin=proposal.origin,
                metadata=proposal.metadata,
            )

        order_manager.mark_filled(
            order,
            fill_price=proposal.fill_price,
            filled_quantity=proposal.quantity,
            timestamp=proposal.time,
            bar_index=proposal.bar_index,
            reason=proposal.reason,
        )

        fee = self.fee_model.calculate_fee(
            proposal.fill_price,
            proposal.units,
            quantity=proposal.quantity,
            side=proposal.side,
            is_opening=is_opening,
        )

        meta = dict(proposal.metadata) if proposal.metadata else {}
        meta["slippage_cost"] = proposal.slippage * proposal.units
        meta["phase"] = proposal.phase.value
        meta["origin"] = proposal.origin.value
        meta["base_price"] = proposal.base_price
        meta["bar_index"] = proposal.bar_index

        self._fill_counter += 1
        fill = Fill(
            fill_id=f"FILL-{self._fill_counter:05d}",
            order_id=order.order_id,
            time=proposal.time,
            symbol=proposal.symbol,
            side=proposal.side,
            price=proposal.fill_price,
            quantity=proposal.quantity,
            units=proposal.units,
            fee=fee,
            slippage=proposal.slippage,
            reason=proposal.reason,
            metadata=meta,
        )

        portfolio.on_fill(
            fill,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            bar_index=proposal.bar_index,
            phase=proposal.phase,
        )
        return fill

    # Compatibility method with ExecutionModel ABC
    def execute_order(self, order: Order, bar: pd.Series) -> Fill | None:
        if order.status != OrderStatus.PENDING:
            return None
        raw_price = float(order.limit_price) if order.limit_price is not None else float(bar["open"])
        if order.order_type == OrderType.LIMIT or order.origin == OrderOrigin.PROTECTIVE_TP:
            fill_p, slip = raw_price, 0.0
        else:
            fill_p, slip = self._apply_slippage(raw_price, order.side)
        fee = self.fee_model.calculate_fee(fill_p, order.units, quantity=order.quantity, side=order.side)
        self._fill_counter += 1
        fill = Fill(
            fill_id=f"FILL-{self._fill_counter:05d}",
            order_id=order.order_id,
            time=pd.Timestamp(bar["time"]),
            symbol=order.symbol,
            side=order.side,
            price=fill_p,
            quantity=order.quantity,
            units=order.units,
            fee=fee,
            slippage=slip,
            reason=order.reason,
            metadata={"slippage_cost": slip * order.units},
        )
        order.status = OrderStatus.FILLED
        return fill

