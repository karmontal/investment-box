"""The approval state machine.

The central property under test: **a timeout is never an approval**, and that
holds whether or not the background sweeper ever runs.
"""

from __future__ import annotations

import pytest

from investment_box.core.clock import FrozenClock
from investment_box.core.types import ApprovalKind, ApprovalStatus
from investment_box.services.approvals import MAX_SNOOZES, SNOOZE_MINUTES, ApprovalService

PAYLOAD = {
    "symbol": "SPUS",
    "side": "buy",
    "quantity": "2",
    "entry_price": "59.83",
    "strategy": "etf_momentum_rotation",
    "probability": 0.58,
    "compliance_status": "compliant",
    "reason": "top-ranked on 3m risk-adjusted momentum",
}


def make(service: ApprovalService, key: str = "k1", **kwargs):
    return service.create(
        request_key=key, kind=ApprovalKind.TRADE_PROPOSAL, payload=dict(PAYLOAD), **kwargs
    )


class TestCreation:
    def test_starts_pending(self, approvals: ApprovalService) -> None:
        request = make(approvals)
        assert request.status is ApprovalStatus.PENDING
        assert not request.is_actionable

    def test_payload_round_trips(self, approvals: ApprovalService) -> None:
        assert make(approvals).payload["symbol"] == "SPUS"

    def test_deadline_respects_the_timeout(
        self, approvals: ApprovalService, clock: FrozenClock
    ) -> None:
        request = make(approvals, timeout_minutes=45)
        assert request.seconds_remaining(clock.now()) == pytest.approx(45 * 60, abs=2)

    def test_duplicate_key_returns_the_same_request(self, approvals: ApprovalService) -> None:
        """Asking the same question twice must not send two messages."""
        first = make(approvals, "same")
        second = make(approvals, "same")
        assert first.id == second.id

    def test_reasking_after_a_decision_opens_a_new_request(
        self, approvals: ApprovalService
    ) -> None:
        first = make(approvals, "same")
        approvals.respond(first.id, approved=False, responder_id="u1")
        second = make(approvals, "same")
        assert second.id != first.id
        assert second.status is ApprovalStatus.PENDING


class TestResponses:
    def test_approve(self, approvals: ApprovalService) -> None:
        request = make(approvals)
        updated = approvals.respond(request.id, approved=True, responder_id="555000111")
        assert updated is not None
        assert updated.status is ApprovalStatus.APPROVED
        assert updated.is_actionable
        assert updated.responder_id == "555000111"

    def test_reject(self, approvals: ApprovalService) -> None:
        request = make(approvals)
        updated = approvals.respond(request.id, approved=False, responder_id="u1")
        assert updated is not None
        assert updated.status is ApprovalStatus.REJECTED
        assert not updated.is_actionable

    def test_double_tap_does_not_overturn(self, approvals: ApprovalService) -> None:
        """A retried callback or an impatient second press must be a no-op."""
        request = make(approvals)
        approvals.respond(request.id, approved=True, responder_id="u1")
        second = approvals.respond(request.id, approved=False, responder_id="u1")
        assert second is not None
        assert second.status is ApprovalStatus.APPROVED

    def test_unknown_id(self, approvals: ApprovalService) -> None:
        assert approvals.respond(99999, approved=True, responder_id="u1") is None


class TestTimeoutNeverApproves:
    def test_pending_past_deadline_reads_as_expired(
        self, approvals: ApprovalService, clock: FrozenClock
    ) -> None:
        """Enforced on read, so a stalled sweeper cannot leave it answerable."""
        request = make(approvals, timeout_minutes=30)
        clock.advance(minutes=31)
        reloaded = approvals.get(request.id)
        assert reloaded is not None
        assert reloaded.status is ApprovalStatus.EXPIRED
        assert not reloaded.is_actionable

    def test_late_approval_is_not_honoured(
        self, approvals: ApprovalService, clock: FrozenClock
    ) -> None:
        """The dangerous case: the user taps Approve after the deadline."""
        request = make(approvals, timeout_minutes=30)
        clock.advance(minutes=31)
        updated = approvals.respond(request.id, approved=True, responder_id="u1")
        assert updated is not None
        assert updated.status is ApprovalStatus.EXPIRED
        assert not updated.is_actionable

    def test_sweeper_expires_due_requests(
        self, approvals: ApprovalService, clock: FrozenClock
    ) -> None:
        make(approvals, "a", timeout_minutes=30)
        make(approvals, "b", timeout_minutes=90)
        clock.advance(minutes=31)

        expired = approvals.expire_due()
        assert len(expired) == 1
        assert expired[0].status is ApprovalStatus.EXPIRED
        assert "defaulted to REJECT" in (expired[0].response_note or "")

    def test_sweeper_is_idempotent(self, approvals: ApprovalService, clock: FrozenClock) -> None:
        make(approvals, timeout_minutes=30)
        clock.advance(minutes=31)
        assert len(approvals.expire_due()) == 1
        assert approvals.expire_due() == []

    def test_expired_excluded_from_pending(
        self, approvals: ApprovalService, clock: FrozenClock
    ) -> None:
        """/pending must never invite approval of something already timed out."""
        make(approvals, "a", timeout_minutes=30)
        make(approvals, "b", timeout_minutes=120)
        clock.advance(minutes=31)
        assert len(approvals.pending()) == 1

    @pytest.mark.parametrize("status", list(ApprovalStatus))
    def test_only_approved_authorises(self, status: ApprovalStatus) -> None:
        assert status.authorises_action is (status is ApprovalStatus.APPROVED)


class TestModification:
    def test_modify_parks_the_request(self, approvals: ApprovalService) -> None:
        request = make(approvals)
        updated = approvals.request_modification(request.id, "u1")
        assert updated is not None
        assert updated.status is ApprovalStatus.AWAITING_MODIFICATION
        assert not updated.is_actionable

    def test_applying_a_size_approves_at_that_size(self, approvals: ApprovalService) -> None:
        request = make(approvals)
        approvals.request_modification(request.id, "u1")
        updated = approvals.apply_modification(request.id, new_quantity="1", responder_id="u1")
        assert updated is not None
        assert updated.status is ApprovalStatus.APPROVED
        assert updated.payload["quantity"] == "1"
        assert updated.payload["original_quantity"] == "2"

    @pytest.mark.parametrize("bad", ["0", "-1", "abc", "", "  "])
    def test_invalid_sizes_rejected(self, approvals: ApprovalService, bad: str) -> None:
        request = make(approvals)
        approvals.request_modification(request.id, "u1")
        assert approvals.apply_modification(request.id, new_quantity=bad, responder_id="u1") is None

    def test_cannot_modify_a_request_that_was_not_parked(
        self, approvals: ApprovalService
    ) -> None:
        request = make(approvals)
        updated = approvals.apply_modification(request.id, new_quantity="1", responder_id="u1")
        assert updated is not None
        assert updated.status is ApprovalStatus.PENDING

    def test_modification_still_expires(
        self, approvals: ApprovalService, clock: FrozenClock
    ) -> None:
        """Parking for a size change does not suspend the deadline."""
        request = make(approvals, timeout_minutes=30)
        approvals.request_modification(request.id, "u1")
        clock.advance(minutes=31)
        updated = approvals.apply_modification(request.id, new_quantity="1", responder_id="u1")
        assert updated is not None
        assert updated.status is ApprovalStatus.EXPIRED


class TestSnooze:
    def test_extends_the_deadline(self, approvals: ApprovalService, clock: FrozenClock) -> None:
        request = make(approvals, timeout_minutes=30)
        before = request.seconds_remaining(clock.now())
        updated = approvals.snooze(request.id, "u1")
        assert updated is not None
        assert updated.seconds_remaining(clock.now()) == pytest.approx(
            before + SNOOZE_MINUTES * 60, abs=2
        )

    def test_limited_number_of_snoozes(self, approvals: ApprovalService) -> None:
        """Snooze is 'give me a minute', not indefinite deferral."""
        request = make(approvals, timeout_minutes=30)
        for _ in range(MAX_SNOOZES):
            approvals.snooze(request.id, "u1")
        capped = approvals.snooze(request.id, "u1")
        assert capped is not None
        assert capped.payload["snoozes_used"] == MAX_SNOOZES

    def test_cannot_snooze_after_expiry(
        self, approvals: ApprovalService, clock: FrozenClock
    ) -> None:
        request = make(approvals, timeout_minutes=30)
        clock.advance(minutes=31)
        updated = approvals.snooze(request.id, "u1")
        assert updated is not None
        assert updated.status is ApprovalStatus.EXPIRED


class TestCancellation:
    def test_cancel_a_pending_request(self, approvals: ApprovalService) -> None:
        request = make(approvals)
        updated = approvals.cancel(request.id, "price moved")
        assert updated is not None
        assert updated.status is ApprovalStatus.CANCELLED
        assert not updated.is_actionable

    def test_cannot_cancel_a_decided_request(self, approvals: ApprovalService) -> None:
        request = make(approvals)
        approvals.respond(request.id, approved=True, responder_id="u1")
        updated = approvals.cancel(request.id, "too late")
        assert updated is not None
        assert updated.status is ApprovalStatus.APPROVED


class TestAuditTrail:
    def test_every_transition_is_recorded(self, approvals: ApprovalService, audit) -> None:
        request = make(approvals)
        approvals.respond(request.id, approved=True, responder_id="555000111")

        events = [entry.event_type for entry in audit.recent()]
        assert "approval.requested" in events
        assert "approval.approved" in events

    def test_rejection_records_the_responder(self, approvals: ApprovalService, audit) -> None:
        request = make(approvals)
        approvals.respond(request.id, approved=False, responder_id="555000222", note="too big")
        entry = audit.recent(event_type="approval.rejected")[0]
        assert entry.actor == "555000222"
        assert "too big" in entry.summary

    def test_timeout_is_audited(
        self, approvals: ApprovalService, clock: FrozenClock, audit
    ) -> None:
        make(approvals, timeout_minutes=30)
        clock.advance(minutes=31)
        approvals.expire_due()
        assert audit.recent(event_type="approval.expired")
