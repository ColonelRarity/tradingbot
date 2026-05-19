"""Paper cost model — fees must eat into small scalps."""

from bot.config import Config
from bot.paper.costs import PaperCostModel


def _model() -> PaperCostModel:
    cfg = Config()
    cfg.paper_taker_fee = 0.0004
    cfg.paper_slippage_bps = 3.0
    cfg.paper_spread_bps = 2.0
    cfg.paper_fee_rate = 0.0004
    return PaperCostModel(cfg)


def test_round_trip_costs_positive_on_flat_price():
    m = _model()
    qty = 0.01
    mark = 100_000.0
    entry = m.simulate_entry_fill(mark, qty, "LONG")
    exit_ = m.simulate_exit_fill(mark, qty, "LONG")
    total = entry.total + exit_.total
    assert total > 0
    assert entry.fill_price > mark
    assert exit_.fill_price < mark


def test_funding_long_pays_when_rate_positive():
    m = _model()
    pay = m.funding_payment("LONG", 1000.0, 0.0005, 8 * 3600)
    assert pay > 0
