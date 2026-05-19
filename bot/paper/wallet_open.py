"""Shared paper wallet open/reset for scalper and Champaign."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from bot.paper.account import PaperWallet

if TYPE_CHECKING:
    from bot.config import Config
    from bot.exchange.client import BinanceFuturesClient

logger = logging.getLogger(__name__)


def resolve_paper_target(cfg: "Config", client: Optional["BinanceFuturesClient"] = None) -> float:
    if cfg.paper_balance > 0:
        return cfg.paper_balance
    if cfg.paper_use_real_balance_seed and client is not None:
        try:
            return client.get_balance().available_balance
        except Exception:
            pass
    return cfg.paper_initial_balance


def should_force_paper_reset(cfg: "Config") -> bool:
    if cfg.paper_reset_on_start:
        return True
    if cfg.champaign_reset_state_on_start and cfg.paper_balance > 0:
        return True
    return False


def open_paper_wallet(
    cfg: "Config",
    client: Optional["BinanceFuturesClient"] = None,
    *,
    force_reset: bool = False,
) -> PaperWallet:
    target = resolve_paper_target(cfg, client)
    if force_reset or should_force_paper_reset(cfg):
        wallet = PaperWallet.create_fresh(target)
        wallet.save()
        return wallet

    loaded = PaperWallet.load()
    if loaded and abs(loaded.initial_balance - target) > 0.01:
        logger.warning(
            "Paper wallet was %.2f USDT; config wants %.2f — "
            "set PAPER_RESET_ON_START=true or CHAMPAIGN_RESET_STATE_ON_START=true to reseed",
            loaded.initial_balance,
            target,
        )
    return loaded or PaperWallet.create_fresh(target)


def log_paper_wallet(wallet: PaperWallet, configured: float) -> None:
    logger.info(
        "PAPER wallet: %.2f USDT (configured %.2f) | costs paid: %.4f "
        "(fee %.4f slip %.4f spread %.4f funding %.4f)",
        wallet.balance,
        configured,
        wallet.total_costs,
        wallet.total_commission,
        wallet.total_slippage,
        wallet.total_spread,
        wallet.total_funding,
    )
