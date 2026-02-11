# 日频实盘修复记录

## 修复完成清单

### ✅ 1. 统一风控参数
**文件**: `crypto_trader/live_trading_okx.py` 第112行

**修改**: max_drawdown_limit 从 1.0 改为 0.15
```python
self.risk_manager = RiskManager(max_drawdown_limit=0.15, freeze_period_steps=1)
```

**原因**: 与回测保持一致，启用回撤保护

---

### ✅ 2. 改造为日频调度
**文件**: `crypto_trader/start_simulation.py`

**修改**: 从60秒循环改为每日8:05执行一次
- 执行时间: 中国时间 8:05 (UTC+8)
- 执行窗口: 30分钟 (8:05 - 8:35)
- 防重复: 当日成功执行后不再重复

**使用方法**:
```bash
python crypto_trader/start_simulation.py
```

---

### ✅ 3. 防重复执行逻辑
**文件**: `crypto_trader/live_trading_okx.py`

**新增方法**:
- `has_executed_today()`: 检查今日是否已执行
- `mark_execution()`: 标记执行状态

**状态文件**: `runs/daily_status_YYYYMMDD.json`

**重试逻辑**: 失败后最多重试3次

---

### ✅ 4. 数据就绪检查
**文件**: `crypto_trader/live_trading_okx.py`

**新增方法**: `check_daily_data_ready()`

**检查逻辑**:
- 检查OKX最新日K线日期是否等于当前UTC日期
- 重试3次，间隔30秒
- 数据未就绪时发出警告但继续执行

---

### ✅ 5. 8:30未执行告警
**文件**: `scripts/check_daily_execution.py` (新建)

**功能**:
- 检查今日执行状态
- 未执行时发送告警（钉钉/企业微信/飞书）

**配置cron**:
```bash
./scripts/setup_cron.sh
```

会自动配置:
- 8:05 执行策略
- 8:35 检查告警
- 9:00 备用检查

---

## 快速部署步骤

### 1. 配置服务器时区
```bash
sudo timedatectl set-timezone Asia/Shanghai
timedatectl  # 验证
```

### 2. 配置定时任务
```bash
cd /path/to/强化学习\ i
./scripts/setup_cron.sh
```

### 3. 验证配置
```bash
crontab -l
```

应显示:
```
5 8 * * * cd ... && python3 .../run_live.py --auto >> .../runs/daily_$(date+\%Y\%m\%d).log 2>&1
35 8 * * * cd ... && python3 .../check_daily_execution.py >> .../runs/monitor.log 2>&1
```

### 4. 测试运行（模拟盘）
```bash
# 手动执行一次测试
python crypto_trader/run_demo.py

# 或启动日频调度器（前台运行）
python crypto_trader/start_simulation.py
```

---

## 实盘前检查清单

- [ ] 服务器时区设置为 Asia/Shanghai
- [ ] crontab 配置正确
- [ ] .env.live 配置正确（实盘API密钥）
- [ ] CONFIRM_REAL_MONEY=True
- [ ] 告警系统已配置（钉钉/企业微信/飞书webhook）
- [ ] 模拟盘运行3-5天验证正常
- [ ] 日志文件正常生成
- [ ] 每日8:35告警测试通过

---

## 监控命令

```bash
# 查看今日执行状态
cat runs/daily_status_$(date +%Y-%m-%d).json

# 查看今日执行日志
tail -f runs/daily_$(date +%Y%m%d).log

# 查看监控日志
tail -f runs/monitor.log

# 查看当前持仓和状态
cat runs/live_status.json
```

---

## 修复验证

所有修复已完成，实盘与回测现在:
- ✅ 风控参数一致 (max_drawdown_limit=0.15)
- ✅ 执行频率一致 (每日一次)
- ✅ 触发时间一致 (日线数据更新后)
- ✅ 防重复机制 (避免过度交易)
- ✅ 监控告警 (8:30未执行通知)
