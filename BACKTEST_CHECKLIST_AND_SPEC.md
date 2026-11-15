# ĐẶC TẢ THIẾT KẾ & CHECKLIST NGIỆM THU QUANT BACKTESTER

Tài liệu này đóng vai trò là **Kim chỉ nam kiến trúc (Architecture Blueprint)** và **Checklist nghiệm thu kỹ thuật (Acceptance Checklist)** cho toàn bộ quá trình xây dựng hệ thống `quant_backtester`.

---

## 1. 3 NGUYÊN TẮC CỐT LÕI (CORE PRINCIPLES)

1. **Strategy-agnostic:** Không Strategy nào được phép khiến Engine phải biết riêng logic của nó. Engine chỉ là một máy trạng thái (State Machine / Orchestrator) tiếp nhận Intent/Order chuẩn hóa và xử lý.
2. **Model-driven:** Execution, Fee, Slippage, Margin và Risk phải có thể thay thế bằng các model khác nhau (Pluggable Components) thay vì nhúng cứng vào Engine Core.
3. **Test-driven:** Mọi thay đổi trong Engine phải có test chứng minh kết quả đúng, đặc biệt là các phép tính tiền (Cash, Balance, Equity, Realized/Unrealized PnL) và giá khớp (Fill price, Slippage, Gap).

---

## 2. CHECKLIST TỔNG THỂ (84 TIÊU CHÍ)

> **Quy tắc nghiệm thu:** Chỉ đánh dấu `[x]` khi đã triển khai **VÀ** có Unit / Integration Test chứng minh tính đúng đắn, không chỉ vì đã viết code.

### A. Kiến trúc và Phân tách Trách nhiệm (P0) — [9/9] Hoàn thành
- [x] Engine không chứa logic riêng của bất kỳ Strategy nào.
- [x] Strategy chỉ phân tích dữ liệu và tạo quyết định giao dịch.
- [x] Có Base Strategy Interface thống nhất.
- [x] Execution Model độc lập với Engine Core.
- [x] Risk Management độc lập với Strategy.
- [x] Có lớp quản lý Portfolio/Account.
- [x] Cấu hình Engine, Strategy và Execution tách biệt.
- [x] Có Strategy Registry để đăng ký và lựa chọn Strategy.
- [x] Có cấu trúc BacktestResult thống nhất.

### B. Market Data và Data Validation (P0) — [8/8] Hoàn thành
- [x] Chuẩn hóa OHLCV và timestamp.
- [x] Kiểm tra cột bắt buộc, kiểu dữ liệu và giá trị null.
- [x] Kiểm tra High ≥ Open/Close, Low ≤ Open/Close và High ≥ Low.
- [x] Phát hiện timestamp trùng và dữ liệu không sắp xếp.
- [x] Phát hiện missing candles và khoảng nghỉ thị trường.
- [x] Quản lý timezone và DST nhất quán.
- [x] Lưu nguồn dữ liệu, symbol, timeframe và loại giá Bid/Ask/Mid.
- [x] Có thiết kế mở rộng cho nhiều symbol và timeframe.

### C. Strategy Interface (P0) — [8/8] Hoàn thành
- [x] Strategy có phương thức `generate_signals()` hoặc `on_bar()`.
- [x] Tham số Strategy có schema, mặc định và validation.
- [x] Có cơ chế quản lý state của Strategy rõ ràng.
- [x] Strategy khai báo warm-up requirement.
- [x] Chuẩn hóa signal: Long, Short, Exit, Flat.
- [x] Phân biệt rõ: Signal (ý định), Order Intent (lệnh) và Execution (khớp thực tế).
- [x] Có cách truyền SL/TP, trailing stop hoặc sizing instruction tùy chọn.
- [x] Strategy không tự gọi broker hoặc tự ghi nhận PnL.

### D. Order và Execution Model (P0) — [10/10] Hoàn thành
- [x] Hỗ trợ Market, Limit và Stop Order theo phạm vi đã chọn.
- [x] Không khớp lệnh tại giá đã biết trước khi tín hiệu được tạo (Anti-Lookahead).
- [x] Có quy tắc fill price rõ ràng.
- [x] Mô phỏng Bid/Ask spread.
- [x] Tính commission theo đúng đơn vị hợp đồng.
- [x] Có slippage model.
- [x] Xử lý gap qua Stop Loss/Take Profit.
- [x] Có quy tắc khi SL và TP cùng bị chạm trong một nến.
- [x] Thiết kế mở rộng cho partial fills và rejected orders.
- [x] Ghi lại order lifecycle và lý do fill/reject/cancel.

### E. Position và Account Management (P0) — [8/8] Hoàn thành
- [x] Quản lý trạng thái Flat, Long, Short.
- [x] Quản lý entry, exit và reverse position.
- [x] Mỗi trade có ID và liên kết với các order.
- [x] Phân biệt realized và unrealized PnL.
- [x] Phân bổ phí entry và exit vào trade.
- [x] Equity = Balance + Unrealized PnL theo đúng quy ước.
- [x] Có quy tắc đóng vị thế cuối kỳ.
- [x] Xác định rõ hỗ trợ một hay nhiều vị thế trên mỗi symbol.

### F. Risk Management và Portfolio (P1) — [0/8]
- [ ] Position sizing theo risk amount và stop distance.
- [ ] Giới hạn rủi ro tối đa mỗi giao dịch.
- [ ] Daily loss limit.
- [ ] Maximum drawdown limit.
- [ ] Giới hạn exposure theo symbol và toàn tài khoản.
- [ ] Mô phỏng margin, leverage và margin call.
- [ ] Quản lý rủi ro tương quan giữa các vị thế.
- [ ] Có cơ chế ngừng tạo lệnh khi vượt giới hạn rủi ro.

### G. Accounting và Performance Analytics (P0) — [0/8]
- [ ] Có ledger ghi nhận cash flow và giao dịch.
- [ ] Tính return từ equity series nhất quán.
- [ ] Tính drawdown từ equity curve.
- [ ] Sharpe có annualization factor phù hợp timeframe.
- [ ] Có Sortino và Calmar.
- [ ] Có expectancy, average R và profit factor.
- [ ] Trade statistics tính trên PnL ròng.
- [ ] Có benchmark và phân tích theo giai đoạn.

### H. Validation và Backtest Integrity (P0) — [0/9]
- [ ] Không có look-ahead bias.
- [ ] Warm-up không tạo giao dịch ngoài phạm vi kiểm thử.
- [ ] Có In-Sample và Out-of-Sample tách biệt.
- [ ] Walk-Forward ghép OOS theo đúng thứ tự thời gian.
- [ ] Cùng dữ liệu, config và seed cho kết quả có thể tái lập.
- [ ] Có stress test spread, commission và slippage.
- [ ] Có kiểm tra độ ổn định tham số.
- [ ] Có phân tích nhiều điều kiện/regime thị trường.
- [ ] Có kiểm soát multiple testing và overfitting.

### I. Testing và Software Quality (P0) — [0/8]
- [ ] Unit tests cho từng module.
- [ ] Integration tests cho toàn bộ lifecycle giao dịch.
- [ ] Golden tests với dữ liệu nhỏ và kết quả tính tay.
- [ ] Test gap, SL/TP cùng chạm, thiếu dữ liệu, cuối kỳ.
- [ ] Property tests cho các invariant tài chính.
- [ ] Test tính deterministic.
- [ ] Có coverage report cho phần core.
- [ ] Có tài liệu API, ví dụ và hướng dẫn phát triển Strategy.

### J. Extensibility và Production Readiness (P2) — [0/8]
- [ ] Có thể đăng ký Strategy và Execution Model mới.
- [ ] Có event hooks cho order, fill, trade và equity.
- [ ] Logging có cấu trúc và mức độ log.
- [ ] Có thể serialize config, result và trade log.
- [ ] Có thể chạy nhiều backtest độc lập song song.
- [ ] Engine có thể được gọi qua API mà không phụ thuộc UI.
- [ ] Có version cho Engine, Strategy và dữ liệu.
- [ ] Không trộn backtest execution với live broker execution.

---

## 3. THỨ TỰ TRIỂN KHAI THỰC TẾ

| Giai đoạn | Phạm vi nhóm | Mục tiêu nghiệm thu |
| :--- | :--- | :--- |
| **V1.0 — Core Engine** | **A, B, C, D, E, G, H, I** | Backtest một chiến lược chính xác, nhất quán từng cent. |
| **V1.1 — Risk & Portfolio** | **F** và mở rộng **E, G** | Quản lý vốn, Margin/Leverage và nhiều vị thế. |
| **V2.0 — Advanced Framework**| **J** và tính năng nâng cao | Phục vụ nghiên cứu chuyên sâu, API và Web dashboard. |

> [!IMPORTANT]
> Tuyệt đối không nhảy cóc sang V1.1 hoặc V2.0 trước khi V1.0 vượt qua cả 5 Tiêu chuẩn nghiệm thu (Gates) dưới đây.

---

## 4. TIÊU CHUẨN NGHIỆM THU ENGINE CORE V1.0 (5 GATES)

### Gate 01: Strategy Independence
- Có thể thay Mean Reversion bằng RSI mà không sửa Engine.
- Engine không import trực tiếp class cụ thể của Strategy.
- Strategy không tự tính PnL, balance hoặc equity.

### Gate 02: Execution Correctness
- Signal được tạo khi dữ liệu của nến đã hoàn tất.
- Lệnh Market được khớp theo quy tắc đã cấu hình, không dùng dữ liệu tương lai.
- Spread, commission và slippage được tính đúng.
- Gap và SL/TP cùng chạm có quy tắc xử lý xác định.

### Gate 03: Accounting Correctness
- Balance, unrealized PnL và equity được tính nhất quán.
- Phí được ghi nhận chính xác một lần.
- Trade Log khớp với Account Ledger.
- Tổng PnL trong báo cáo khớp với thay đổi equity sau khi xử lý các dòng tiền.

### Gate 04: Backtest Integrity
- Không có look-ahead bias.
- Warm-up không tạo giao dịch ngoài phạm vi backtest.
- Train và Test không bị rò rỉ dữ liệu.
- Walk-Forward tổng hợp OOS theo thứ tự thời gian.
- Cùng đầu vào và cấu hình cho kết quả tái lập được (Deterministic).

### Gate 05: Automated Testing
- Unit test cho từng module.
- Integration test cho toàn bộ vòng đời lệnh.
- Có test dữ liệu nhỏ với kết quả tính tay (Golden Test).
- Có test cho gap, đảo chiều, cuối kỳ, phí và SL/TP.
- Tất cả test quan trọng đều chạy thành công trong CI.

---

## 5. KIẾN TRÚC MỤC TIÊU (TARGET ARCHITECTURE)

```text
quant_backtester/
├── core/
│   ├── engine.py           # Điều phối vòng lặp Event Loop chính
│   ├── event_loop.py       # Quản lý luồng Bar / Tick Events
│   ├── context.py          # Ngữ cảnh thực thi
│   └── clock.py            # Quản lý mốc thời gian mô phỏng
│
├── data/
│   ├── data_feed.py        # Cung cấp dữ liệu nến cho Engine
│   ├── loader.py           # Đọc CSV / Database SQLite
│   ├── validator.py        # Kiểm tra tính toàn vẹn của nến OHLCV
│   └── resampler.py        # Chuyển đổi khung thời gian
│
├── instrument/
│   ├── base.py             # Instrument model: lot_size, contract_size, pip_size
│   └── forex.py            # Mô hình tính pip value, quote currency
│
├── strategies/
│   ├── base.py             # Abstract Base Class Strategy
│   ├── registry.py         # Strategy Registry (đăng ký / lookup)
│   └── implementations/    # Các chiến lược cụ thể
│       ├── mean_reversion.py
│       ├── rsi.py
│       └── trend_following.py
│
├── execution/
│   ├── order.py            # Order dataclasses (Market, Limit, Stop)
│   ├── order_manager.py    # Quản lý hàng đợi và vòng đời Order
│   ├── fill_model.py       # Mô phỏng khớp lệnh (Open/Close, Intrabar)
│   ├── slippage_model.py   # Mô phỏng trượt giá
│   └── fee_model.py        # Mô phỏng Spread và Commission
│
├── portfolio/
│   ├── account.py          # Quản lý Cash Balance và Equity
│   ├── position.py         # Quản lý Position (Long/Short, PnL)
│   ├── portfolio.py        # Quản lý đa vị thế
│   └── ledger.py           # Sổ cái ghi nhận dòng tiền
│
├── risk/
│   ├── position_sizing.py  # Tính Lot Size theo Risk / Stop Distance
│   ├── risk_manager.py     # Kiểm soát Drawdown / Daily Loss
│   └── margin_model.py     # Đòn bẩy và Margin Call
│
├── analytics/
│   ├── performance.py      # Return, Sharpe, Sortino, Calmar
│   ├── trade_analysis.py   # Win rate, Profit Factor, Expectancy
│   └── drawdown.py         # Drawdown curve & underwater chart
│
├── validation/
│   ├── train_test.py       # Phân tách In-Sample / Out-of-Sample
│   ├── walk_forward.py     # Cửa sổ trượt Rolling Window
│   └── robustness.py       # Stress testing & Overfitting check
│
├── results/
│   ├── result.py           # BacktestResult dataclass chuẩn
│   └── exporter.py         # Xuất báo cáo CSV / HTML / JSON
│
└── tests/
    ├── unit/               # Test từng hàm/class độc lập
    ├── integration/        # Test vòng đời từ Signal -> Fill -> PnL
    └── fixtures/           # Bộ nến mẫu tính tay (Golden Data)
```

---

## 6. MÔ HÌNH HÓA INSTRUMENT CHO QUANT FOREX / CFD

Đối với Forex và Vàng CFD, **tuyệt đối không hardcode** các thông số sau:

| Thuộc tính | EUR/USD | USD/JPY | XAU/USD (Gold) |
| :--- | :--- | :--- | :--- |
| **Contract Size** | 100,000 units | 100,000 units | Thường là 100 oz (tùy broker) |
| **Tick Size / Pip** | 0.0001 (pip), 0.00001 (point) | 0.01 (pip), 0.001 (point) | 0.01$ (pip), 0.001$ (point) |
| **Pip Value** | $10 / lot | Phụ thuộc tỷ giá USD/JPY | $1 / lot (mỗi biến động 0.01$) |
| **Quote Currency** | USD | JPY (cần quy đổi về USD) | USD |
| **Swap / Financing** | Theo broker | Theo broker | Theo broker |

Mọi tính toán PnL, Margin và Pip Value phải thông qua đối tượng `Instrument` cấu hình theo sàn.

---

## 7. TÀI LIỆU THAM KHẢO (REALITY MODELING THEO QUANTCONNECT LEAN)

- [LEAN — Reality Modeling Overview](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/key-concepts)
- [LEAN — Trading and Orders](https://www.quantconnect.com/docs/v2/writing-algorithms/trading-and-orders/key-concepts)
- [LEAN — Trade Fills Simulation](https://www.quantconnect.com/docs/v2/writing-algorithms/reality-modeling/trade-fills/key-concepts)

