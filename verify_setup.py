#!/usr/bin/env python3
"""
Verification script to ensure all modules can be imported correctly.
"""

import sys

def verify_imports():
    """Verify all modules can be imported."""
    print("=" * 60)
    print("Self-Learning Trading Bot - Setup Verification")
    print("=" * 60)
    
    modules = [
        ("config.settings", "Configuration"),
        ("exchange.binance_client", "Exchange Client"),
        ("core.market_data", "Market Data"),
        ("core.feature_engineering", "Feature Engineering"),
        ("core.pattern_memory", "Pattern Memory"),
        ("core.signal_engine", "Signal Engine"),
        ("core.risk_engine", "Risk Engine"),
        ("core.order_manager", "Order Manager"),
        ("core.hedge_manager", "Hedge Manager"),
        ("core.position_tracker", "Position Tracker"),
        ("ml.model", "ML Models"),
        ("ml.trainer", "Model Trainer"),
        ("ml.inference", "Model Inference"),
    ]
    
    success = 0
    failed = 0
    
    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"[OK] {description:25} ({module_name})")
            success += 1
        except ImportError as e:
            print(f"[FAIL] {description:25} ({module_name}): {e}")
            failed += 1
        except Exception as e:
            print(f"[FAIL] {description:25} ({module_name}): {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {success} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        print("\nSome modules failed to import. Please check dependencies:")
        print("  pip install -r requirements.txt")
        return False
    
    # Additional checks
    print("\nRunning additional checks...")
    
    # Check settings
    try:
        from config.settings import get_settings
        settings = get_settings()
        print(f"[OK] Settings loaded (symbol: {settings.market_data.symbol})")
    except Exception as e:
        print(f"[FAIL] Settings error: {e}")
        failed += 1
    
    # Check ML model can be created
    try:
        from ml.model import create_model
        from core.feature_engineering import FeatureEngine
        
        fe = FeatureEngine()
        model = create_model(fe.feature_dim)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"[OK] ML Model created ({param_count:,} parameters)")
    except Exception as e:
        print(f"[FAIL] ML Model error: {e}")
        failed += 1
    
    # Check pattern memory DB
    try:
        from core.pattern_memory import PatternMemory
        pm = PatternMemory()
        count = pm.get_pattern_count()
        print(f"[OK] Pattern Memory initialized ({count} patterns)")
    except Exception as e:
        print(f"[FAIL] Pattern Memory error: {e}")
        failed += 1
    
    print()
    
    if failed == 0:
        print("[OK] All checks passed! Bot is ready to run.")
        print()
        print("To start the bot:")
        print("  python main.py --symbol BTCUSDT")
        print()
        print("Make sure to set environment variables:")
        print("  BINANCE_FUTURES_API_KEY=<your_testnet_key>")
        print("  BINANCE_FUTURES_API_SECRET=<your_testnet_secret>")
        return True
    else:
        print(f"[FAIL] {failed} checks failed. Please fix issues before running.")
        return False


if __name__ == "__main__":
    success = verify_imports()
    sys.exit(0 if success else 1)
