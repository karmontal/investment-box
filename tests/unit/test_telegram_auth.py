"""The whitelist.

This is the security boundary for the interactive bot, so the tests are written
as adversarial cases rather than happy paths.
"""

from __future__ import annotations

import pytest

from investment_box.telegram.auth import (
    TELEGRAM_FORBIDDEN_ACTIONS,
    AuthGuard,
)

ALLOWED = 555000111
STRANGER = 999999999


class TestWhitelist:
    def test_allowed_user_passes(self) -> None:
        guard = AuthGuard(allowed_user_ids=frozenset({ALLOWED}))
        assert guard.check(ALLOWED, context="/balance")

    def test_stranger_rejected(self) -> None:
        guard = AuthGuard(allowed_user_ids=frozenset({ALLOWED}))
        assert not guard.check(STRANGER, context="/balance")

    def test_empty_whitelist_authorises_nobody(self) -> None:
        """A missing TELEGRAM_ALLOWED_USER_IDS must not mean 'allow everyone'."""
        guard = AuthGuard(allowed_user_ids=frozenset())
        assert not guard.is_enabled
        assert not guard.check(ALLOWED, context="/balance")
        assert not guard.check(STRANGER, context="/balance")

    def test_missing_user_id_rejected(self) -> None:
        """Channel posts and service messages carry no sender."""
        guard = AuthGuard(allowed_user_ids=frozenset({ALLOWED}))
        assert not guard.check(None, context="channel_post")

    def test_username_is_not_an_identity(self) -> None:
        """Usernames are user-settable; only the numeric id counts."""
        guard = AuthGuard(allowed_user_ids=frozenset({ALLOWED}))
        assert not guard.check(STRANGER, context="/balance", username="karmontal")

    def test_multiple_allowed_ids(self) -> None:
        guard = AuthGuard(allowed_user_ids=frozenset({111, 222}))
        assert guard.check(111, context="x")
        assert guard.check(222, context="x")
        assert not guard.check(333, context="x")


class TestAuditOfRejections:
    def test_unauthorised_attempt_is_audited(self, audit) -> None:
        guard = AuthGuard(allowed_user_ids=frozenset({ALLOWED}), audit=audit)
        guard.check(STRANGER, context="/balance", username="attacker")

        entries = audit.recent(event_type="telegram.unauthorised_access")
        assert len(entries) == 1
        assert str(STRANGER) in entries[0].summary

    def test_repeat_attempts_are_not_logged_repeatedly(self, audit) -> None:
        """One persistent stranger should not flood the audit log."""
        guard = AuthGuard(allowed_user_ids=frozenset({ALLOWED}), audit=audit)
        for _ in range(20):
            guard.check(STRANGER, context="/balance")
        assert len(audit.recent(event_type="telegram.unauthorised_access")) == 1

    def test_distinct_strangers_are_each_logged(self, audit) -> None:
        guard = AuthGuard(allowed_user_ids=frozenset({ALLOWED}), audit=audit)
        guard.check(111111, context="/balance")
        guard.check(222222, context="/balance")
        assert len(audit.recent(event_type="telegram.unauthorised_access")) == 2

    def test_authorised_use_is_not_audited_as_a_rejection(self, audit) -> None:
        guard = AuthGuard(allowed_user_ids=frozenset({ALLOWED}), audit=audit)
        guard.check(ALLOWED, context="/balance")
        assert audit.recent(event_type="telegram.unauthorised_access") == []


class TestForbiddenActions:
    @pytest.mark.parametrize("action", sorted(TELEGRAM_FORBIDDEN_ACTIONS))
    def test_live_switching_is_refused_from_telegram(self, action: str) -> None:
        """Enabling live trading is a dashboard-only, typed-confirmation action.

        A chat interface is the wrong place for an irreversible money-at-risk
        decision, and a stolen phone must not be able to make it.
        """
        guard = AuthGuard(allowed_user_ids=frozenset({ALLOWED}))
        with pytest.raises(PermissionError, match="dashboard"):
            guard.assert_action_allowed(action)

    def test_ordinary_actions_are_allowed(self) -> None:
        guard = AuthGuard(allowed_user_ids=frozenset({ALLOWED}))
        guard.assert_action_allowed("balance")

    def test_being_whitelisted_does_not_unlock_live_switching(self) -> None:
        """Authorisation and capability are separate. The whitelist is not a bypass."""
        guard = AuthGuard(allowed_user_ids=frozenset({ALLOWED}))
        assert guard.check(ALLOWED, context="x")
        with pytest.raises(PermissionError):
            guard.assert_action_allowed("switch_to_live")

    @pytest.mark.parametrize(
        "action", ["kill", "close_all_positions", "set_autonomy", "pause", "resume"]
    )
    def test_destructive_actions_need_confirmation(self, action: str) -> None:
        assert AuthGuard.requires_confirmation(action)

    def test_read_only_actions_need_no_confirmation(self) -> None:
        assert not AuthGuard.requires_confirmation("balance")
