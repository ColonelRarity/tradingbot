"""Paper exploration profile."""

import os

from bot.config import Config, apply_paper_exploration, reload_config


def test_exploration_auto_when_dry_run(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("PAPER_EXPLORATION", "auto")
    monkeypatch.setenv("STRATEGY_MODE", "strict")
    reload_config()
    from bot.config import load_config

    cfg = load_config()
    assert cfg.paper_exploration_active
    assert cfg.strategy_mode == "strict"
    assert cfg.ml_warmup_samples == 0
    assert cfg.paper_balance == 500.0
    assert cfg.paper_use_real_balance_seed is False
    assert cfg.paper_skip_daily_loss_limit is True
    assert cfg.rsi_long_max <= 65.0
    assert cfg.symbol_cooldown_sec >= 900.0


def test_exploration_off_keeps_strict(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("PAPER_EXPLORATION", "false")
    monkeypatch.setenv("STRATEGY_MODE", "strict")
    cfg = Config()
    assert not apply_paper_exploration(cfg)
    assert cfg.strategy_mode == "strict"
