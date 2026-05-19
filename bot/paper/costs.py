"""
Realistic cost model for paper trading (matches live futures micro-PnL).

Includes:
- Taker commission (entry + exit, MARKET orders)
- Slippage (worse fill vs mark)
- Bid/ask spread proxy
- Funding rate (pro-rated; live rate from Binance when available)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot.config import Config


FUNDING_INTERVAL_SEC = 8 * 3600  # Binance USD-M: every 8 hours


@dataclass
class FillResult:
    """Simulated execution price and immediate costs (USDT)."""
    fill_price: float
    commission: float
    slippage_cost: float
    spread_cost: float

    @property
    def total(self) -> float:
        return self.commission + self.slippage_cost + self.spread_cost


@dataclass
class CloseCosts:
    commission: float
    slippage_cost: float
    spread_cost: float
    funding: float

    @property
    def total(self) -> float:
        return self.commission + self.slippage_cost + self.spread_cost + self.funding


class PaperCostModel:
    def __init__(self, config: "Config"):
        self.taker_fee = config.paper_taker_fee or config.paper_fee_rate
        self.slippage_bps = config.paper_slippage_bps
        self.spread_bps = config.paper_spread_bps
        self.funding_fallback = config.paper_funding_fallback_rate
        self.use_live_funding = config.paper_funding_use_live

    def _bps_to_frac(self, bps: float) -> float:
        return bps / 10_000.0

    def simulate_entry_fill(
        self,
        mark: float,
        quantity: float,
        side: str,
    ) -> FillResult:
        slip = self._bps_to_frac(self.slippage_bps)
        spr = self._bps_to_frac(self.spread_bps)

        if side == "LONG":
            fill = mark * (1 + slip + spr)
        else:
            fill = mark * (1 - slip - spr)

        notional = fill * quantity
        commission = notional * self.taker_fee
        slippage_cost = abs(fill - mark) * quantity * (slip / max(slip + spr, 1e-12))
        spread_cost = mark * quantity * spr

        return FillResult(
            fill_price=fill,
            commission=commission,
            slippage_cost=slippage_cost,
            spread_cost=spread_cost,
        )

    def simulate_exit_fill(
        self,
        mark: float,
        quantity: float,
        side: str,
    ) -> FillResult:
        slip = self._bps_to_frac(self.slippage_bps)
        spr = self._bps_to_frac(self.spread_bps)

        if side == "LONG":
            fill = mark * (1 - slip - spr)
        else:
            fill = mark * (1 + slip + spr)

        notional = fill * quantity
        commission = notional * self.taker_fee
        slippage_cost = abs(fill - mark) * quantity
        spread_cost = mark * quantity * spr

        return FillResult(
            fill_price=fill,
            commission=commission,
            slippage_cost=slippage_cost,
            spread_cost=spread_cost,
        )

    def funding_payment(
        self,
        side: str,
        notional_usdt: float,
        funding_rate: float,
        seconds_held: float,
    ) -> float:
        """
        Pro-rated funding over hold time.
        Positive payment = debit from wallet (you pay).
        Binance: rate > 0 → longs pay, shorts receive.
        """
        if seconds_held <= 0 or notional_usdt <= 0:
            return 0.0
        fraction = min(seconds_held / FUNDING_INTERVAL_SEC, 1.0)
        raw = notional_usdt * funding_rate * fraction
        if side == "LONG":
            return raw
        return -raw
