
import ccxt
import os
from dotenv import load_dotenv
from crypto_trader.ccxt_utils import apply_ccxt_proxy_config

load_dotenv()

exchange = ccxt.okx(apply_ccxt_proxy_config({
    'apiKey': os.getenv('OKX_API_KEY'),
    'secret': os.getenv('OKX_SECRET_KEY'),
    'password': os.getenv('OKX_PASSPHRASE'),
}))
if os.getenv('OKX_DEMO_MODE') == 'True':
    exchange.set_sandbox_mode(True)

try:
    print("Fetching account configuration...")
    # config = exchange.fetch_account_configuration() # Not all exchanges support this
    # print(config)
    
    # Try private API for OKX
    res = exchange.private_get_account_config()
    print(f"Current Account Config (acctLv): {res['data'][0]['acctLv']}")
    print(f"Current Position Mode (posMode): {res['data'][0]['posMode']}")
    
    # Try to set margin mode with leverage
    print("Testing set_margin_mode with lever=1...")
    # exchange.set_margin_mode('cross', 'ETH/USDT:USDT', params={'lever': 1})
    # print("Success!")
except Exception as e:
    print(f"Error: {e}")
