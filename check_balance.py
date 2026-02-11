

import ccxt
import os
import sys
import argparse
from dotenv import load_dotenv

def check_balance():
    parser = argparse.ArgumentParser(description='Check OKX account balance.')
    parser.add_argument('--live', action='store_true', help='Check real account balance using .env.live')
    args = parser.parse_args()

    # Load environment variables
    if args.live:
        print("Loading configuration from .env.live for REAL trading...")
        load_dotenv('.env.live')
        # Ensure we are not in demo mode if using .env.live (override if set in file)
        # But we should respect the file content if it explicitly sets demo mode, though .env.live shouldn't.
        # Actually, let's just force sandbox off if --live is passed, assuming .env.live contains real keys.
        is_demo = False 
    else:
        load_dotenv()
        is_demo = os.getenv('OKX_DEMO_MODE') == 'True'
    
    api_key = os.getenv('OKX_API_KEY')
    secret_key = os.getenv('OKX_SECRET_KEY')
    passphrase = os.getenv('OKX_PASSPHRASE')
    
    if not api_key or not secret_key or not passphrase:
        print("Error: OKX API credentials not found.")
        return

    try:
        exchange = ccxt.okx({
            'apiKey': api_key,
            'secret': secret_key,
            'password': passphrase,
        })
        
        if is_demo:
            exchange.set_sandbox_mode(True)
            print("Running in DEMO mode")
        else:
            print("Running in REAL mode")
            
        print("Connecting to OKX...")
        
        # Fetch balance
        balance = exchange.fetch_balance()
        
        # Print USDT balance
        usdt_balance = balance.get('USDT', {})
        total_usdt = usdt_balance.get('total', 0)
        free_usdt = usdt_balance.get('free', 0)
        used_usdt = usdt_balance.get('used', 0)
        
        print(f"\nTime: {exchange.iso8601(exchange.milliseconds())}")
        print("-" * 30)
        print(f"USDT Total Balance: {total_usdt:.2f}")
        print(f"USDT Free Balance:  {free_usdt:.2f}")
        print(f"USDT Used Balance:  {used_usdt:.2f}")
        print("-" * 30)
        
        # Print other non-zero balances
        print("\nOther Non-Zero Balances:")
        has_other = False
        for currency, data in balance.items():
            if currency != 'USDT' and isinstance(data, dict) and data.get('total', 0) > 0:
                print(f"{currency}: {data['total']}")
                has_other = True
        
        if not has_other:
            print("None")
            
    except Exception as e:
        print(f"Error checking balance: {e}")

if __name__ == "__main__":
    check_balance()
