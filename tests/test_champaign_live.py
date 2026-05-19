"""Live Champaign: exchange client params and order manager (mocked)."""

from unittest.mock import MagicMock, patch

from bot.champaign.live_sync import _sl_tp_from_algos, reconcile_live_stacks
from bot.champaign.order_manager import ProtectiveOrderManager
from bot.champaign.state import ChampaignStack, LegState, ProtectiveSlot
from bot.config import Config
from bot.exchange.client import BinanceFuturesClient, Order, OrderSide


def test_algo_sl_includes_reduce_only_oneway():
    client = BinanceFuturesClient(Config(api_key="k", api_secret="s"))
    client._dual_side = False
    with patch.object(client, "_algo_order") as mock_algo:
        client.stop_loss("BTCUSDT", OrderSide.SELL, 99000.0, 0.01)
        params = mock_algo.call_args[0][0]
        assert params["reduceOnly"] == "true"
        assert params["type"] == "STOP_MARKET"


def test_update_sl_cancels_only_sl_algo():
    cfg = Config(champaign_max_protective_orders=2)
    client = MagicMock(spec=BinanceFuturesClient)
    client.is_dual_side.return_value = False
    client.stop_loss.return_value = Order(
        1, "X", "SELL", "STOP_MARKET", "NEW", 1.0, 0.99, is_algo=True
    )
    om = ProtectiveOrderManager(client, cfg, dry_run=False)
    stack = ChampaignStack(
        symbol="XUSDT",
        main=LegState("LONG", 1.0, 1.0, 10.0),
        active_sl=0.95,
        active_tp=1.5,
        protective=[
            ProtectiveSlot("SL", 0.95, 88),
            ProtectiveSlot("TP", 1.5, 99),
        ],
    )
    om.update_sl(stack, 1.02, 1.0, OrderSide.SELL, None)
    client.cancel_algo_order.assert_called_with(88, "XUSDT")
    client.stop_loss.assert_called_once()
    client.take_profit.assert_not_called()


def test_sl_tp_from_algos():
    algos = [
        Order(1, "A", "SELL", "STOP_MARKET", "NEW", 1.0, 0.98, is_algo=True),
        Order(2, "A", "SELL", "TAKE_PROFIT_MARKET", "NEW", 1.0, 1.2, is_algo=True),
    ]
    sl, tp, slots = _sl_tp_from_algos(algos, "LONG")
    assert sl == 0.98
    assert tp == 1.2
    assert len(slots) == 2


def test_reconcile_drops_closed_stack():
    cfg = Config(leverage=5)
    client = MagicMock()
    client.get_positions.return_value = []
    client.get_open_algo_orders.return_value = []
    stacks = {
        "OLD": ChampaignStack(symbol="OLD", main=LegState("LONG", 1, 1, 1)),
    }
    out = reconcile_live_stacks(client, cfg, stacks)
    assert "OLD" not in out
    client.cancel_all_orders.assert_called_with("OLD")
