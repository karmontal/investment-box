"""Properties of docker-compose.yml that protect the deployment.

The dashboard has no authentication of any kind. It carries a kill switch,
shows balances and positions, and edits autonomy level and allocated capital.
The only thing standing between it and the internet is which host address the
port is published on, so that address is worth a test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "docker-compose.yml"

#: Addresses that would accept connections from outside the host. Listed here
#: in order to REJECT them, which is why S104 does not apply.
PUBLIC_BINDS = ("0.0.0.0", "::", "*")  # noqa: S104


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text())


def _port_entries(compose: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for name, service in compose.get("services", {}).items():
        for entry in service.get("ports", []) or []:
            out.append((name, str(entry)))
    return out


class TestDashboardExposure:
    def test_the_default_bind_is_loopback_or_a_variable_defaulting_to_it(
        self, compose: dict
    ) -> None:
        for service, entry in _port_entries(compose):
            assert entry.startswith(("127.0.0.1:", "${")), (
                f"{service} publishes {entry!r} with no explicit host address. "
                f"Docker then binds every interface, and the dashboard has no login."
            )

    def test_no_service_binds_a_public_address(self, compose: dict) -> None:
        for service, entry in _port_entries(compose):
            for bad in PUBLIC_BINDS:
                assert not entry.startswith(f"{bad}:"), (
                    f"{service} publishes on {bad}, which is the whole internet"
                )

    def test_an_empty_override_still_falls_back_to_loopback(self, compose: dict) -> None:
        """`${VAR-default}` keeps an EMPTY value; `${VAR:-default}` replaces it.

        The difference is one character and decides whether
        `IB_DASHBOARD_BIND=` in a .env file publishes the dashboard to every
        interface. It must be the colon form.
        """
        for service, entry in _port_entries(compose):
            if not entry.startswith("${"):
                continue
            var = entry[2 : entry.index("}")]
            assert ":-" in var, (
                f"{service} uses ${{{var}}}; an empty value would survive and Docker "
                f"would bind all interfaces. Use the ${{VAR:-127.0.0.1}} form."
            )
            default = var.split(":-", 1)[1]
            assert default == "127.0.0.1", (
                f"{service} defaults its bind address to {default!r}; it must be loopback"
            )


class TestContainerHardening:
    def test_the_engine_gets_an_init_process(self, compose: dict) -> None:
        """Without it nothing reaps zombies, and PID 1 signal semantics bite."""
        assert compose["services"]["engine"].get("init") is True

    def test_config_is_mounted_read_only(self, compose: dict) -> None:
        for name, service in compose["services"].items():
            mounts = [m for m in service.get("volumes", []) if "./config" in str(m)]
            for mount in mounts:
                assert str(mount).endswith(":ro"), (
                    f"{name} mounts config writable; the engine must never rewrite it"
                )

    def test_every_service_restarts_unless_stopped(self, compose: dict) -> None:
        """A VPS reboots. The engine should come back without a human."""
        for name, service in compose["services"].items():
            assert service.get("restart") == "unless-stopped", f"{name} has no restart policy"
