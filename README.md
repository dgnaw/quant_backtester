# Quant Backtesting Framework (modular starter)

## 1. Mục tiêu

Đây là bộ khung backtest có thể thay nhiều chiến lược mà không viết lại phần mô phỏng giao dịch. Hiện tại framework có một chiến lược mẫu: Bollinger Bands Mean Reversion.

**Nguyên tắc cốt lõi:** Strategy tạo tín hiệu; Engine mô phỏng việc thực thi tín hiệu. Strategy không tự sửa balance, tự tính phí hoặc tự ghi PnL.

## 2. Cấu trúc thư mục

```text
quant_backtester/
├── __init__.py
├── config.py                 # Cấu hình vốn, lot, phí, SL/TP, timeframe
├── data.py                   # Đọc và kiểm tra OHLC
├── engine.py                 # Engine dùng chung cho mọi strategy
├── analytics.py              # Equity, drawdown, Sharpe, trade metrics
├── walk_forward.py           # Walk-forward mẫu cho Mean Reversion
├── run_backtest.py           # CLI entry point
├── strategies/
│   ├── __init__.py
│   ├── base.py                # Interface bắt buộc của strategy
│   ├── mean_reversion.py      # Strategy đầu tiên
│   └── (các strategy mới đặt ở đây)
└── examples/
```

## 3. Trách nhiệm từng lớp

### Strategy

Strategy nhận dữ liệu thị trường và trả về DataFrame có các cột:

- `entry_signal`: `1` = mở Long, `-1` = mở Short, `0` = không có tín hiệu vào lệnh.
- `exit_long_signal`: tín hiệu đóng vị thế Long.
- `exit_short_signal`: tín hiệu đóng vị thế Short.

Các tín hiệu được xác định tại **Close của nến hiện tại**. Strategy không quyết định giá khớp lệnh. Engine thực thi quyết định ở **Open của nến kế tiếp**.

### Engine

Engine không biết Bollinger Bands, RSI, MACD hay FVG là gì. Engine chỉ đọc các tín hiệu chuẩn hóa và quản lý:

- Entry/exit theo thời gian.
- Vị thế Long/Short.
- Phí vào và phí ra.
- Slippage bất lợi cố định theo đơn vị giá.
- Stop Loss / Take Profit theo tỷ lệ phần trăm.
- Equity mark-to-market.
- Đóng vị thế cuối kỳ nếu `force_close_at_end=True`.

Nếu một nến OHLC chạm cả SL và TP, engine giả định SL xảy ra trước. Đây là giả định thận trọng vì OHLC không cho biết thứ tự giá đi trong nến. Nếu mở cửa vượt qua SL/TP, engine mô phỏng khớp tại Open thay vì giá stop/target cũ.

### Analytics

Các chỉ số dùng Equity Curve và trade log của Engine. Win rate, Profit Factor và Expectancy sử dụng `net_pnl`, đã trừ cả phí vào và phí ra.

### Walk-forward

Bản hiện tại có ví dụ Walk-Forward riêng cho Mean Reversion: chọn tham số trên đoạn train, rồi đánh giá trên đoạn kế tiếp. Đây là bản khung nghiên cứu, chưa phải một hệ thống tối ưu tham số tổng quát cho mọi Strategy.

## 4. Luồng hoạt động

```text
OHLC CSV
   ↓
Data validation
   ↓
Strategy.generate_signals()
   ↓
entry_signal / exit_long_signal / exit_short_signal
   ↓
BacktestEngine
   ├── Execute pending order at next bar Open
   ├── Apply slippage and fees
   ├── Check gap + intrabar SL/TP
   ├── Update balance and unrealized PnL
   └── Build equity curve and trade log
   ↓
Analytics / Walk-forward
   ↓
CSV reports
```

## 5. Cài đặt và chạy

Python 3.10+ được khuyến nghị.

```bash
pip install pandas numpy
```

Đặt terminal ở thư mục cha chứa `quant_backtester/`, rồi chạy:

```bash
python -m quant_backtester.run_backtest path/to/XAUUSD_M15.csv
```

Chạy kèm Walk-Forward:

```bash
python -m quant_backtester.run_backtest path/to/XAUUSD_M15.csv --walk-forward
```

Thay đổi tham số:

```bash
python -m quant_backtester.run_backtest path/to/XAUUSD_M15.csv \
  --window 30 --std-multiplier 2.2 --initial-balance 1000 \
  --lot-size 0.01 --contract-size 100 --fee-rate 0.00015 --slippage 0.05
```

CSV yêu cầu tối thiểu:

```csv
time,open,high,low,close
2026-01-01 00:00:00,4300.0,4302.0,4298.0,4301.0
```

Thời gian phải sắp xếp được; timestamp trùng và OHLC không hợp lệ sẽ bị từ chối. Hãy kiểm tra timezone, missing candles, nguồn giá (Bid/Ask/Mid), và dữ liệu bất thường trước khi backtest.

Kết quả được ghi vào thư mục `output/`:

- `trades_train.csv`, `trades_test.csv`
- `equity_train.csv`, `equity_test.csv`
- `walk_forward_folds.csv`, `walk_forward_summary.csv` nếu bật Walk-Forward

## 6. Cách thêm Strategy mới

Ví dụ tạo `strategies/rsi_mean_reversion.py`:

```python
from dataclasses import dataclass
import pandas as pd
from .base import Strategy, StrategySignals

@dataclass
class RSIMeanReversion(Strategy):
    period: int = 14
    oversold: float = 30
    overbought: float = 70
    name = "rsi_mean_reversion"

    def generate_signals(self, market_data: pd.DataFrame) -> StrategySignals:
        df = market_data.copy()
        # TODO: tính RSI tại đây
        df["entry_signal"] = 0
        df["exit_long_signal"] = False
        df["exit_short_signal"] = False
        # TODO: gán 1/-1 khi có entry; exit theo hướng vị thế
        return StrategySignals(df)

    def get_parameters(self) -> dict:
        return {"period": self.period, "oversold": self.oversold,
                "overbought": self.overbought}
```

Sau đó dùng chung Engine:

```python
from quant_backtester.config import BacktestConfig
from quant_backtester.data import load_ohlc_csv
from quant_backtester.engine import BacktestEngine
from quant_backtester.strategies.rsi_mean_reversion import RSIMeanReversion

market = load_ohlc_csv("data.csv")
engine = BacktestEngine(BacktestConfig())
strategy = RSIMeanReversion()
result = engine.run(market, strategy)
```

Để thêm chiến lược MACD, Breakout, Trend Following hoặc FVG, tạo file mới trong `strategies/` và tuân thủ đúng interface. Không copy engine sang file strategy mới.

## 7. Các quyết định mô phỏng quan trọng

1. **Signal timing:** chỉ dùng thông tin đã biết tại Close nến hiện tại.
2. **Execution timing:** khớp ở Open nến kế tiếp, tránh dùng Close vừa tạo tín hiệu làm giá khớp.
3. **SL/TP:** kiểm tra gap tại Open và mức chạm trong biên độ High/Low.
4. **Ambiguous candle:** nếu cả SL và TP cùng bị chạm trong một nến, giả định SL trước.
5. **Fees:** `fee_rate_per_side` tính theo tỷ lệ notional mỗi chiều. Ví dụ `0.00015` là 0.015% mỗi chiều.
6. **Slippage:** `slippage_price` là số đơn vị giá bất lợi cho mỗi lần khớp; đây là mô hình cố định đơn giản.
7. **Position sizing:** hiện dùng lot cố định; chưa có risk-based sizing.
8. **Final liquidation:** nếu bật, vị thế còn mở được đóng tại Close cuối cùng và tính phí thoát.

## 8. Hiểu các chỉ số

- **Total Return:** phần trăm thay đổi của Equity so với vốn ban đầu.
- **Max Drawdown:** mức sụt giảm lớn nhất từ đỉnh Equity trước đó.
- **Sharpe annualized:** lợi nhuận trung bình chia độ lệch chuẩn lợi nhuận, nhân căn bậc hai số nến/năm giả định. `bars_per_year` phải phù hợp timeframe và lịch giao dịch.
- **Win Rate:** tỷ lệ giao dịch có `net_pnl > 0`.
- **Profit Factor:** tổng lãi ròng của các lệnh thắng chia tổng lỗ tuyệt đối của các lệnh thua.
- **Expectancy:** PnL ròng trung bình trên mỗi giao dịch.

Không có chỉ số đơn lẻ nào chứng minh một strategy tốt. Cần xem cả số lượng lệnh, phân phối PnL, drawdown, thời gian nằm trong thị trường, độ ổn định tham số và hiệu quả ở nhiều chế độ thị trường.

## 9. Giới hạn hiện tại — đọc trước khi dùng tiền thật

Đây là **research framework**, chưa phải production/live trading engine.

- Phí đang là tỷ lệ notional đơn giản, chưa mô hình spread Bid/Ask thực tế, commission theo broker, swap/financing, market impact và phí chuyển đổi tiền tệ.
- Slippage cố định chưa mô phỏng spread theo phiên, volatility, liquidity, latency, partial fills hoặc rejected orders.
- Chưa có margin, leverage, margin call, stop-out, giới hạn rủi ro ngày, giới hạn drawdown hay sizing theo % vốn.
- Chưa có portfolio/multi-symbol, netting/hedging, correlation và tổng exposure.
- OHLC không xác định được đường đi giá bên trong nến; dùng dữ liệu tick/lower timeframe nếu cần mô phỏng thứ tự SL/TP chính xác hơn.
- Walk-forward hiện là mẫu cho Mean Reversion và các fold là các phiên OOS độc lập bắt đầu từ vốn giả định; chưa mô phỏng một tài khoản liên tục chuyển vốn/vị thế qua mọi fold.
- Chưa có bootstrap/Monte Carlo, kiểm định multiple-testing/PBO, regime analysis hoặc forward paper trading.
- Chưa có kết nối broker, order reconciliation, monitoring, alert và kill switch.

Đặc biệt, `contract_size=100` chỉ là ví dụ cấu hình cho một số hợp đồng XAUUSD; phải xác minh contract specification của symbol và broker bạn dùng. Không mặc định áp dụng cho mọi sản phẩm Forex/CFD.

## 10. Thứ tự phát triển khuyến nghị

1. Hoàn thiện data validation và xác nhận quy ước giá.
2. Viết unit tests cho engine: long/short, stop, target, gap, phí, end-of-test.
3. Thêm strategy mới qua interface, không sửa Engine nếu không có yêu cầu mô phỏng chung.
4. Chuẩn hóa cost model theo broker và instrument.
5. Thêm risk sizing, margin và portfolio risk.
6. Hoàn thiện WFA/robustness/statistical validation.
7. Paper forward test, đối chiếu simulated fills với broker demo.
8. Chỉ cân nhắc live sau khi có monitoring, kill switch và giới hạn rủi ro.

---

**Ghi nhớ:** Strategy trả lời *khi nào muốn giao dịch*; Engine trả lời *nếu giao dịch thì lệnh được khớp và tài khoản thay đổi như thế nào*.

## 11. Danh mục đầy đủ file trong project

```text
quant_backtester/
├── __init__.py
├── config.py
├── data.py
├── engine.py
├── analytics.py
├── walk_forward.py
├── run_backtest.py
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── README.md
├── strategies/
│   ├── __init__.py
│   ├── base.py
│   ├── mean_reversion.py
│   └── strategy_template.py
├── examples/
│   └── generate_sample_data.py
└── tests/
    └── test_framework.py
```

## 12. Chạy thử bằng dữ liệu mẫu và kiểm thử

Từ thư mục cha của `quant_backtester/`:

```bash
python -m pip install -e ".[dev]"
python -m quant_backtester.examples.generate_sample_data
python -m quant_backtester.run_backtest quant_backtester/examples/sample_ohlc.csv
python -m pytest quant_backtester/tests -v
```

Dữ liệu mẫu được sinh ngẫu nhiên để kiểm tra luồng chương trình; không dùng để đánh giá lợi nhuận chiến lược.

### Cài dependencies chính xác

Mở terminal tại **thư mục cha** chứa folder `quant_backtester/`:

```bash
python -m pip install -r quant_backtester/requirements.txt
python -m pip install pytest
```

Sau đó sinh dữ liệu mẫu và chạy:

```bash
python -m quant_backtester.examples.generate_sample_data
python -m quant_backtester.run_backtest quant_backtester/examples/sample_ohlc.csv
python -m pytest quant_backtester/tests -v
```

Không cần cài package ở chế độ editable. Cách chạy `python -m quant_backtester...` yêu cầu terminal đang đứng ở thư mục cha của `quant_backtester/`.
