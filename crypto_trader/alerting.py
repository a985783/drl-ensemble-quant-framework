import json
import os
import requests
from datetime import datetime
from dotenv import load_dotenv


class AlertManager:
    def __init__(self):
        load_dotenv()  # Ensure .env is loaded
        self.provider = (os.getenv("ALERT_PROVIDER") or "").strip().lower()
        self.webhook = (os.getenv("ALERT_WEBHOOK_URL") or "").strip()
        self.enabled = bool(self.provider and self.webhook)
        if self.enabled:
            print(f"✅ [Alert] Enabled provider: {self.provider}")
        else:
            print("⚠️ [Alert] Disabled (missing config)")

    def send(self, level: str, text: str):
        if not self.enabled:
            return
        payload = self._build_payload(level, text)
        if payload is None:
            return
            
        try:
            resp = requests.post(self.webhook, json=payload, timeout=5)
            if resp.status_code != 200:
                print(f"❌ [Alert] Failed to send: {resp.text}")
            # else:
            #    print(f"✅ [Alert] Sent: {text[:20]}...")
        except Exception as e:
            print(f"❌ [Alert] Exception: {e}")

    def send_trade(self, action: str, symbol: str, price: float, amount: float, equity: float):
        # 汉化交易方向
        direction = "🟢 买入" if "buy" in action.lower() else "🔴 卖出"
        
        msg = (
            f"【实盘成交】{direction} {symbol}\n"
            f"💰 成交均价: ${price:,.2f}\n"
            f"📦 成交张数: {amount}\n"
            f"💎 当前权益: ${equity:,.2f}"
        )
        self.send("交易", msg)

    def _build_payload(self, level: str, text: str):
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 映射日志级别为中文
        level_map = {
            "INFO": "ℹ️ 信息",
            "WARN": "⚠️ 警报",
            "ERROR": "🚨 报错",
            "TRADE": "💸 交易"
        }
        cn_level = level_map.get(level, level)
        
        full_text = f"[{cn_level}] {stamp}\n{text}"
        
        if self.provider == "wecom":
            return {"msgtype": "text", "text": {"content": full_text}}
        if self.provider == "dingtalk":
            return {
                "msgtype": "text", 
                "text": {"content": full_text},
                "at": {"isAtAll": False}
            }
        if self.provider == "feishu":
            return {"msg_type": "text", "content": {"text": full_text}}
        return None


alerts = AlertManager()
