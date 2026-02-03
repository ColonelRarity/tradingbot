"""
Script to close positions manually.

Usage:
    python close_position.py --symbol QUICKUSDT --side LONG
    python close_position.py --symbol QUICKUSDT --all  # Close all positions
    python close_position.py --all  # Close ALL positions
"""

import sys
import argparse
from exchange.binance_client import get_binance_client, OrderSide
from config.settings import get_settings

def close_position_with_reduce_only(symbol: str):
    """Close position using MARKET order with reduceOnly=True."""
    client = get_binance_client()
    
    # Get current price
    current_price = client.get_ticker_price(symbol)
    print(f"Current price for {symbol}: {current_price:.8f}")
    
    # Get position
    positions = client.get_positions(symbol)
    position = positions[0] if positions else None
    
    if not position or abs(position.size) < 0.0001:
        print(f"No open position found for {symbol}")
        return False
    
    print(f"Current position: {position.size} {symbol} ({position.side})")
    print(f"Entry price: {position.entry_price:.8f}")
    print(f"Mark price: {position.mark_price:.8f}")
    print(f"Unrealized PnL: {position.unrealized_pnl:.2f} USDT")
    
    # Determine close side (opposite to position)
    if position.size > 0:  # LONG
        close_side = OrderSide.SELL
        close_quantity = abs(position.size)
    else:  # SHORT
        close_side = OrderSide.BUY
        close_quantity = abs(position.size)
    
    print(f"\nPlacing MARKET {close_side.value} order with reduceOnly=True:")
    print(f"  Symbol: {symbol}")
    print(f"  Quantity: {close_quantity:.8f}")
    
    try:
        # Format quantity using client's method
        formatted_qty = client._format_quantity(symbol, close_quantity)
        print(f"  Formatted Quantity: {formatted_qty}")
        
        # Place MARKET order with reduceOnly
        response = client._client.new_order(
            symbol=symbol,
            side=close_side.value,
            type="MARKET",
            quantity=formatted_qty,
            reduceOnly=True
        )
        
        print(f"SUCCESS! Order placed successfully!")
        print(f"  Order ID: {response.get('orderId')}")
        print(f"  Status: {response.get('status')}")
        
        return True
        
    except Exception as e:
        error_str = str(e)
        print(f"FAILED to place MARKET order: {error_str}")
        
        # Try with STOP_MARKET order very close to mark price
        if "PERCENT_PRICE" in error_str or "-4131" in error_str:
            print(f"\nPERCENT_PRICE error - price too far from mark price.")
            print(f"Trying with STOP_MARKET order very close to mark price...")
            
            mark_price = position.mark_price
            # Set trigger price very close (0.01% away)
            if close_side == OrderSide.SELL:  # Closing LONG
                trigger_price = mark_price * 0.9999  # Just below mark
            else:  # Closing SHORT
                trigger_price = mark_price * 1.0001  # Just above mark
            
            try:
                formatted_trigger = client._format_price(symbol, trigger_price)
                response = client._client.new_order(
                    symbol=symbol,
                    side=close_side.value,
                    type="STOP_MARKET",
                    stopPrice=formatted_trigger,
                    closePosition=True,
                    workingType="MARK_PRICE"
                )
                print(f"SUCCESS! STOP_MARKET order placed!")
                print(f"  Order ID: {response.get('orderId')}")
                print(f"  Trigger Price: {trigger_price:.8f}")
                print(f"  This will close the position when mark price reaches trigger")
                print(f"  You can cancel this order and try again if needed")
                return True
            except Exception as e2:
                print(f"FAILED: {e2}")
                print(f"\nUnable to close position automatically due to PERCENT_PRICE filter.")
                print(f"\nOptions:")
                print(f"  1. Wait for price to move closer to mark price (recommended)")
                print(f"  2. Use Binance Testnet web interface:")
                print(f"     https://testnet.binancefuture.com/en/futures/{symbol}")
                print(f"     - Go to Positions")
                print(f"     - Click 'Close' on the position")
                print(f"  3. Try again later when market conditions change")
                return False
        
        return False

def close_all_positions():
    """Close all open positions."""
    client = get_binance_client()
    
    # Get all positions
    all_positions = client.get_positions()
    open_positions = [p for p in all_positions if abs(p.size) > 0.0001]
    
    if not open_positions:
        print("No open positions found")
        return
    
    print(f"Found {len(open_positions)} open positions:")
    for pos in open_positions:
        print(f"  {pos.symbol}: {pos.size:.8f} ({pos.side}) | PnL: {pos.unrealized_pnl:.2f} USDT")
    
    print(f"\nClosing all positions...")
    
    for pos in open_positions:
        print(f"\n{'='*60}")
        close_position_with_reduce_only(pos.symbol)

def main():
    parser = argparse.ArgumentParser(description="Close trading positions")
    parser.add_argument("--symbol", type=str, help="Symbol to close (e.g., QUICKUSDT)")
    parser.add_argument("--side", type=str, choices=["LONG", "SHORT"], help="Position side")
    parser.add_argument("--all", action="store_true", help="Close all positions")
    parser.add_argument("--quantity", type=float, help="Quantity to close (optional, defaults to full position)")
    
    args = parser.parse_args()
    
    settings = get_settings()
    print("=" * 60)
    print("Close Position Script")
    print(f"Testnet: {settings.exchange.base_url}")
    print("=" * 60)
    
    if args.all:
        close_all_positions()
    elif args.symbol:
        close_position_with_reduce_only(args.symbol)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
