"""The Shariah hard constraints.

If any test in this file starts failing, stop and understand why before
changing anything: these encode rules that are not supposed to be reachable.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from investment_box.core.errors import ComplianceError
from investment_box.core.types import ComplianceStatus, Side, Symbol
from investment_box.shariah.constraints import (
    CONSTRAINTS,
    assert_order_permissible,
    is_forbidden_instrument,
)

AMPLE_CASH = Decimal("10000")


def permissible(**overrides: object) -> dict[str, object]:
    """A baseline valid buy, so each test varies exactly one thing."""
    base: dict[str, object] = {
        "symbol": Symbol("SPUS"),
        "side": Side.BUY,
        "quantity": Decimal("2"),
        "position_quantity": Decimal("0"),
        "compliance_status": ComplianceStatus.COMPLIANT,
        "cash_available": AMPLE_CASH,
        "order_notional": Decimal("90"),
    }
    base.update(overrides)
    return base


class TestConstraintsAreFixed:
    def test_every_forbidden_mechanism_is_off(self) -> None:
        assert CONSTRAINTS.margin_allowed is False
        assert CONSTRAINTS.short_selling_allowed is False
        assert CONSTRAINTS.derivatives_allowed is False
        assert CONSTRAINTS.leveraged_or_inverse_allowed is False
        assert CONSTRAINTS.crypto_allowed is False

    def test_constraints_are_immutable(self) -> None:
        with pytest.raises((AttributeError, TypeError)):
            CONSTRAINTS.margin_allowed = True  # type: ignore[misc]


class TestForbiddenInstruments:
    @pytest.mark.parametrize("ticker", ["TQQQ", "SQQQ", "SPXL", "UPRO", "SOXL", "TZA"])
    def test_leveraged_tickers_blocked(self, ticker: str) -> None:
        assert is_forbidden_instrument(ticker) is not None

    @pytest.mark.parametrize("ticker", ["UVXY", "VXX", "SVXY"])
    def test_volatility_derivatives_blocked(self, ticker: str) -> None:
        assert is_forbidden_instrument(ticker) is not None

    @pytest.mark.parametrize("ticker", ["BITO", "BTC", "ETH"])
    def test_crypto_blocked(self, ticker: str) -> None:
        assert is_forbidden_instrument(ticker) is not None

    def test_option_symbol_blocked(self) -> None:
        assert is_forbidden_instrument("SPY240119C00450000") is not None

    def test_futures_root_blocked(self) -> None:
        assert is_forbidden_instrument("/ES") is not None

    def test_leveraged_name_blocked_even_with_a_neutral_ticker(self) -> None:
        """The ticker gives nothing away; the name does."""
        reason = is_forbidden_instrument("ABCD", name="ProShares Ultra S&P500 2x Daily")
        assert reason is not None
        assert "leveraged or inverse" in reason

    def test_inverse_name_blocked(self) -> None:
        assert is_forbidden_instrument("WXYZ", name="Direxion Daily Bear 3X Shares") is not None

    @pytest.mark.parametrize("asset_class", ["crypto", "option", "future", "forex"])
    def test_forbidden_asset_classes_blocked(self, asset_class: str) -> None:
        assert is_forbidden_instrument("ABCD", asset_class=asset_class) is not None

    @pytest.mark.parametrize("ticker", ["SPUS", "HLAL", "SPSK", "SPRE", "SPTE", "SPWO", "UMMA"])
    def test_seed_universe_passes(self, ticker: str) -> None:
        assert is_forbidden_instrument(ticker) is None


class TestOrderGate:
    def test_a_normal_compliant_buy_passes(self) -> None:
        assert_order_permissible(**permissible())  # type: ignore[arg-type]

    def test_buy_exceeding_settled_cash_is_margin(self) -> None:
        with pytest.raises(ComplianceError, match="would use margin"):
            assert_order_permissible(
                **permissible(cash_available=Decimal("50"), order_notional=Decimal("90"))  # type: ignore[arg-type]
            )

    def test_selling_more_than_held_is_a_short(self) -> None:
        with pytest.raises(ComplianceError, match="open a short position"):
            assert_order_permissible(
                **permissible(  # type: ignore[arg-type]
                    side=Side.SELL,
                    quantity=Decimal("5"),
                    position_quantity=Decimal("2"),
                )
            )

    def test_selling_exactly_what_is_held_is_fine(self) -> None:
        assert_order_permissible(
            **permissible(  # type: ignore[arg-type]
                side=Side.SELL, quantity=Decimal("2"), position_quantity=Decimal("2")
            )
        )

    def test_buying_a_forbidden_instrument_blocked(self) -> None:
        with pytest.raises(ComplianceError, match="forbidden instrument"):
            assert_order_permissible(**permissible(symbol=Symbol("TQQQ")))  # type: ignore[arg-type]

    def test_buying_non_compliant_blocked(self) -> None:
        with pytest.raises(ComplianceError, match="cannot buy a NON_COMPLIANT"):
            assert_order_permissible(
                **permissible(compliance_status=ComplianceStatus.NON_COMPLIANT)  # type: ignore[arg-type]
            )

    def test_selling_non_compliant_is_allowed(self) -> None:
        """The exit policy requires selling these -- blocking the sell would trap us."""
        assert_order_permissible(
            **permissible(  # type: ignore[arg-type]
                side=Side.SELL,
                quantity=Decimal("2"),
                position_quantity=Decimal("2"),
                compliance_status=ComplianceStatus.NON_COMPLIANT,
            )
        )

    @pytest.mark.parametrize(
        "status", [ComplianceStatus.DOUBTFUL, ComplianceStatus.UNKNOWN]
    )
    def test_doubtful_and_unknown_never_auto_traded(self, status: ComplianceStatus) -> None:
        with pytest.raises(ComplianceError, match="explicit human decision"):
            assert_order_permissible(**permissible(compliance_status=status))  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "status", [ComplianceStatus.DOUBTFUL, ComplianceStatus.UNKNOWN]
    )
    def test_human_approval_unlocks_doubtful_and_unknown(self, status: ComplianceStatus) -> None:
        assert_order_permissible(
            **permissible(compliance_status=status, human_approved=True)  # type: ignore[arg-type]
        )

    def test_human_approval_does_not_unlock_margin(self) -> None:
        """Approval relaxes the status rule only. It is not an override switch."""
        with pytest.raises(ComplianceError, match="would use margin"):
            assert_order_permissible(
                **permissible(  # type: ignore[arg-type]
                    cash_available=Decimal("10"),
                    order_notional=Decimal("90"),
                    human_approved=True,
                )
            )

    def test_human_approval_does_not_unlock_shorting(self) -> None:
        with pytest.raises(ComplianceError, match="open a short position"):
            assert_order_permissible(
                **permissible(  # type: ignore[arg-type]
                    side=Side.SELL,
                    quantity=Decimal("5"),
                    position_quantity=Decimal("0"),
                    human_approved=True,
                )
            )

    def test_human_approval_does_not_unlock_a_leveraged_etf(self) -> None:
        with pytest.raises(ComplianceError, match="forbidden instrument"):
            assert_order_permissible(
                **permissible(symbol=Symbol("TQQQ"), human_approved=True)  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("quantity", [Decimal("0"), Decimal("-1")])
    def test_non_positive_quantity_rejected(self, quantity: Decimal) -> None:
        with pytest.raises(ComplianceError, match="must be positive"):
            assert_order_permissible(**permissible(quantity=quantity))  # type: ignore[arg-type]
