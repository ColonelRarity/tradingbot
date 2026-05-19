"""Risk rules for paper vs live."""

from bot.config import Config
from bot.risk.manager import RiskManager
from bot.strategy.scalper import Side


def test_paper_skips_daily_loss_limit():
    cfg = Config(dry_run=True, paper_skip_daily_loss_limit=True, daily_max_loss_pct=3.0)
    rm = RiskManager(cfg)
    rm._start_equity = 500.0
    rm._halted = True
    assert rm.daily_loss_exceeded(400.0) is False


def test_live_enforces_daily_loss_limit():
    cfg = Config(dry_run=False, daily_max_loss_pct=3.0)
    rm = RiskManager(cfg)
    rm.reset_day_if_needed(500.0)
    assert rm.daily_loss_exceeded(486.0) is False
    assert rm.daily_loss_exceeded(484.0) is True


def test_min_rr_expands_tp():
    cfg = Config(min_rr_ratio=1.5, sl_atr_mult=1.0, tp_atr_mult=1.0)
    rm = RiskManager(cfg)
    plan = rm.build_plan(Side.LONG, 100.0, 1.0, 1000.0, 5, 0.001, 5.0)
    assert plan.valid
    assert plan.take_profit_price - 100.0 >= 1.5 - 0.01
