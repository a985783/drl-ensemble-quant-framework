# 执行运行手册
- 交易所/经纪商：OKX 永续合约
- 标的：ETH/USDT:USDT
- 接入方式：ccxt
- 运行模式：Demo / Live（由 `OKX_DEMO_MODE` 控制）
- 关键环境变量：`OKX_API_KEY`、`OKX_SECRET_KEY`、`OKX_PASSPHRASE`、`OKX_DEMO_MODE`、`CONFIRM_REAL_MONEY`

## 1. 订单与路由
- 订单类型：Limit-then-Market（先限价，超时 60s 转市价）
- Reduce-only：减仓/平仓时强制 reduce-only
- 幂等性：使用 `action_id`（`execution_safety.py`）避免重复下单
- 失败处理：限价失败回退市价；市价未成交则撤销；成交失败清除 pending 允许下次尝试

## 2. 运行保障
- 限频：ccxt `enableRateLimit=True`
- 健康检查：时钟漂移检测、API 失败计数、对账检查
- 状态文件：`trading_state.json`
- 运行状态：`runs/live_status.json`、`runs/live_heartbeat.txt`
- 交易日志：`trade_logs.csv`

## 3. 启动与模式
- 单次执行（自动下单）：`python3 crypto_trader/live_trading_okx.py --auto`
- Shadow 模式（不下单，仅记录信号）：`python3 crypto_trader/live_trading_okx.py --shadow`
- 后台运行示例：`nohup python3 crypto_trader/live_trading_okx.py --auto > simulation.log 2>&1 &`

## 4. 实盘安全门槛
- 必须设置 `CONFIRM_REAL_MONEY=True` 才允许实盘运行，未设置将直接退出

## 5. 紧急处理
- 立即停止：终止运行进程或移除定时任务
- SAFE_MODE：触发后系统只允许 reduce-only
- 强平（手动）：调用 `force_close_all()` 市价全平
