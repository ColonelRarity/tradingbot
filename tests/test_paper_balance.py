"""Paper balance resolution and position sizing caps."""

from bot.config import Config
from bot.risk.manager import RiskManager
from bot.strategy.scalper import Side


def test_paper_balance_overrides_real_seed():
    cfg = Config(
        dry_run=True,
        paper_balance=1000.0,
        paper_use_real_balance_seed=True,
    )
    from bot.paper.wallet_open import resolve_paper_target

    class FakeClient:
        def get_balance(self):
            raise AssertionError("should not call API when PAPER_BALANCE set")

    assert resolve_paper_target(cfg, FakeClient()) == 1000.0


def test_dry_run_caps_notional_per_slot():
    cfg = Config(dry_run=True, max_positions=3, leverage=5, risk_percent=1.0)
    rm = RiskManager(cfg)
    plan = rm.build_plan(
        Side.LONG,
        entry_price=100.0,
        atr_value=0.01,
        available_balance=100.0,
        leverage=5,
        min_qty=0.001,
        min_notional=5.0,
    )
    assert plan.valid
    assert plan.notional_usdt <= 100.0 * 5 / 3 * 0.92 + 0.01
