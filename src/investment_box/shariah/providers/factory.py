"""Choosing the screening provider from configuration.

Before this existed, ``shariah.provider`` was a decorative config key: the
engine and the dashboard each constructed :class:`MockExternalProvider`
directly, whose default verdict is UNKNOWN, so no symbol could ever clear the
compliance gate and nothing could trade. The key is now actually read.

The composition is always the same shape. Certified funds are answered by
:class:`CertifiedFundProvider` from the board recorded in the universe file.
Anything else -- individual stocks, in Mode B -- goes to the configured
provider. In Mode A there is no "anything else" to screen, so no fallback is
built and the composite can honestly call itself a certified source.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from investment_box.config.schema import Settings
from investment_box.core.clock import Clock
from investment_box.core.logging import get_logger
from investment_box.core.types import UniverseMode
from investment_box.shariah.providers.base import ScreeningProvider
from investment_box.shariah.providers.fund_certification import CertifiedFundProvider
from investment_box.shariah.providers.internal_aaoifi import InternalAAOIFIProvider
from investment_box.shariah.providers.mock_external import MockExternalProvider

if TYPE_CHECKING:
    from investment_box.universe.builder import Instrument

log = get_logger(__name__)


class ProviderNotImplementedError(RuntimeError):
    """A configured provider exists as a name but not yet as an integration."""


def build_stock_screener(settings: Settings, *, clock: Clock | None = None) -> ScreeningProvider:
    """The provider for symbols that carry no fund certification of their own."""
    choice = settings.shariah.provider

    if choice == "internal_aaoifi":
        return InternalAAOIFIProvider(settings.shariah, clock=clock)
    if choice == "mock_external":
        return MockExternalProvider(clock=clock)
    raise ProviderNotImplementedError(
        f"shariah.provider is '{choice}', which is a planned integration with no "
        f"implementation yet. Implement it against the ScreeningProvider protocol, "
        f"or set shariah.provider to 'internal_aaoifi'."
    )


def build_screening_provider(
    settings: Settings,
    instruments: Iterable[Instrument],
    *,
    clock: Clock | None = None,
) -> ScreeningProvider:
    """The provider the engine and dashboard should both use."""
    funds = list(instruments)

    fallback: ScreeningProvider | None = None
    if settings.universe.mode is UniverseMode.ETF_AND_SCREENED_STOCKS:
        fallback = build_stock_screener(settings, clock=clock)

    provider = CertifiedFundProvider(funds, fallback=fallback, clock=clock)
    log.info(
        "shariah.provider_selected",
        provider=provider.name,
        fallback=fallback.name if fallback else None,
        certified_source=provider.is_certified_source,
        universe_mode=settings.universe.mode.value,
        certified_funds=sum(1 for f in funds if f.verified and f.certifying_board),
    )
    return provider
