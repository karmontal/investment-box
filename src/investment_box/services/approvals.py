"""The approval state machine.

Deliberately transport-agnostic. Telegram is one way to ask a human; the
dashboard will be another. Both drive this, so the rules below hold no matter
where the answer comes from.

The rules, in order of importance:

1. **A timeout is never an approval.** An unanswered request becomes
   ``EXPIRED``, which does not authorise anything. This is enforced twice: by
   a sweeper that expires due requests, and by every read path, which treats a
   pending-but-past-deadline request as expired even if the sweeper never ran.
   Failing closed must not depend on a background job being alive.
2. **Responses are idempotent.** Only a ``PENDING`` request can be answered, so
   a double-tapped button, a retried callback or a duplicate webhook cannot
   approve something twice or overturn a decision.
3. **Every transition is audited** with who, when and what.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import select

from investment_box.core.clock import UTC, Clock, SystemClock
from investment_box.core.logging import get_logger
from investment_box.core.types import ApprovalKind, ApprovalStatus
from investment_box.db.models import Approval
from investment_box.db.session import Database
from investment_box.services.audit import AuditService

log = get_logger(__name__)

DEFAULT_TIMEOUT_MINUTES = 30
#: How long one Snooze adds. Kept short: snoozing is for "give me a minute",
#: not for deferring a decision indefinitely.
SNOOZE_MINUTES = 15
#: A request may be snoozed only this many times before it is left to expire.
MAX_SNOOZES = 2


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """A decision awaiting a human, as callers see it."""

    id: int
    request_key: str
    kind: ApprovalKind
    payload: dict[str, Any]
    status: ApprovalStatus
    expires_at: dt.datetime
    created_at: dt.datetime
    responded_at: dt.datetime | None = None
    responder_id: str | None = None
    response_note: str | None = None

    @property
    def is_actionable(self) -> bool:
        """Whether the engine may act on this request."""
        return self.status.authorises_action

    def seconds_remaining(self, now: dt.datetime) -> int:
        return max(0, int((self.expires_at - now).total_seconds()))

    @property
    def symbol(self) -> str | None:
        value = self.payload.get("symbol")
        return str(value) if value else None


class ApprovalService:
    """Create, answer and expire approval requests."""

    def __init__(
        self,
        database: Database,
        audit: AuditService,
        *,
        clock: Clock | None = None,
        timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES,
    ) -> None:
        self.db = database
        self.audit = audit
        self.clock = clock or SystemClock()
        self.timeout_minutes = timeout_minutes

    # ----------------------------------------------------------------- create

    def create(
        self,
        *,
        request_key: str,
        kind: ApprovalKind,
        payload: dict[str, Any],
        timeout_minutes: int | None = None,
    ) -> ApprovalRequest:
        """Open a request.

        ``request_key`` is the idempotency handle: asking the same question
        twice returns the existing request rather than spamming a second
        message. If the previous one already reached a terminal state, a new
        request is opened -- the question is being asked again, which is
        different from asking it twice at once.
        """
        now = self.clock.now()
        minutes = timeout_minutes if timeout_minutes is not None else self.timeout_minutes
        expires = now + dt.timedelta(minutes=minutes)

        existing = self.get_by_key(request_key)
        if existing is not None and not existing.status.is_terminal:
            log.info("approval.duplicate_suppressed", request_key=request_key)
            return existing

        stored_key = request_key
        if existing is not None:
            # Same question, new round. Suffix the key so the unique constraint
            # holds while the history of the earlier decision is preserved.
            stored_key = f"{request_key}#{int(now.timestamp())}"

        with self.db.session() as session:
            row = Approval(
                request_key=stored_key,
                kind=kind.value,
                payload_json=json.dumps(payload, default=str),
                status=ApprovalStatus.PENDING.value,
                expires_at=expires,
            )
            session.add(row)
            session.flush()
            session.expunge(row)

        self.audit.record(
            "approval.requested",
            f"Requested {kind.value} approval, expires in {minutes}m",
            actor="engine",
            symbol=payload.get("symbol"),
            detail={"request_key": stored_key, "payload": payload},
        )
        return self._to_request(row)

    # ------------------------------------------------------------------ reads

    def get(self, approval_id: int) -> ApprovalRequest | None:
        with self.db.session() as session:
            row = session.get(Approval, approval_id)
            if row is None:
                return None
            session.expunge(row)
        return self._to_request(row)

    def get_by_key(self, request_key: str) -> ApprovalRequest | None:
        with self.db.session() as session:
            row = session.scalar(
                select(Approval)
                .where(Approval.request_key.startswith(request_key))
                .order_by(Approval.id.desc())
                .limit(1)
            )
            if row is None:
                return None
            session.expunge(row)
        return self._to_request(row)

    def pending(self) -> list[ApprovalRequest]:
        """Requests still genuinely awaiting an answer.

        Anything past its deadline is excluded even if the sweeper has not run,
        so ``/pending`` can never invite a user to approve something already
        timed out.
        """
        now = self.clock.now()
        with self.db.session() as session:
            rows = list(
                session.scalars(
                    select(Approval)
                    .where(
                        Approval.status.in_(
                            [
                                ApprovalStatus.PENDING.value,
                                ApprovalStatus.AWAITING_MODIFICATION.value,
                            ]
                        )
                    )
                    .order_by(Approval.expires_at)
                ).all()
            )
            for row in rows:
                session.expunge(row)
        return [
            request
            for request in (self._to_request(row) for row in rows)
            if request.expires_at > now
        ]

    # --------------------------------------------------------------- transitions

    def respond(
        self,
        approval_id: int,
        *,
        approved: bool,
        responder_id: str,
        note: str | None = None,
    ) -> ApprovalRequest | None:
        """Record a human decision.

        Returns the updated request, or ``None`` if it no longer exists.
        A request that is already terminal is returned unchanged -- the second
        press of a double-tap is a no-op, not an error.
        """
        now = self.clock.now()
        with self.db.session() as session:
            row = session.get(Approval, approval_id)
            if row is None:
                return None

            current = ApprovalStatus(row.status)
            if current.is_terminal:
                log.info(
                    "approval.already_decided",
                    approval_id=approval_id,
                    status=current.value,
                )
                session.expunge(row)
                return self._to_request(row)

            if row.expires_at <= now:
                # The answer arrived after the deadline. It does not count, and
                # in particular a late "approve" must not authorise a trade
                # priced on stale information.
                row.status = ApprovalStatus.EXPIRED.value
                row.responded_at = now
                row.responder_id = responder_id
                row.response_note = "answered after the deadline; treated as expired"
                session.flush()
                session.expunge(row)
                self.audit.record(
                    "approval.expired",
                    "Response arrived after the deadline and was not honoured",
                    actor=responder_id,
                    detail={"approval_id": approval_id},
                )
                return self._to_request(row)

            row.status = (
                ApprovalStatus.APPROVED.value if approved else ApprovalStatus.REJECTED.value
            )
            row.responded_at = now
            row.responder_id = responder_id
            row.response_note = note
            payload = json.loads(row.payload_json)
            session.flush()
            session.expunge(row)

        self.audit.record(
            "approval.approved" if approved else "approval.rejected",
            f"{'Approved' if approved else 'Rejected'} by {responder_id}"
            + (f": {note}" if note else ""),
            actor=responder_id,
            symbol=payload.get("symbol"),
            detail={"approval_id": approval_id, "payload": payload},
        )
        return self._to_request(row)

    def request_modification(self, approval_id: int, responder_id: str) -> ApprovalRequest | None:
        """Mark a request as awaiting a replacement size.

        Does not authorise anything: ``AWAITING_MODIFICATION`` is not terminal
        and does not authorise action, and the original deadline still applies.
        """
        with self.db.session() as session:
            row = session.get(Approval, approval_id)
            if row is None or ApprovalStatus(row.status) is not ApprovalStatus.PENDING:
                return None if row is None else self._to_request(row)
            row.status = ApprovalStatus.AWAITING_MODIFICATION.value
            row.responder_id = responder_id
            session.flush()
            session.expunge(row)

        self.audit.record(
            "approval.modification_requested",
            f"{responder_id} asked to change the size",
            actor=responder_id,
            detail={"approval_id": approval_id},
        )
        return self._to_request(row)

    def apply_modification(
        self, approval_id: int, *, new_quantity: str, responder_id: str
    ) -> ApprovalRequest | None:
        """Approve a request at a different size.

        The engine re-checks risk, cash and compliance against the new size
        before acting, so this only records intent -- it does not bypass any
        limit.
        """
        try:
            quantity = Decimal(str(new_quantity))
        except (InvalidOperation, ValueError):
            return None
        if quantity <= 0:
            return None

        now = self.clock.now()
        with self.db.session() as session:
            row = session.get(Approval, approval_id)
            if row is None:
                return None
            if ApprovalStatus(row.status) is not ApprovalStatus.AWAITING_MODIFICATION:
                session.expunge(row)
                return self._to_request(row)
            if row.expires_at <= now:
                row.status = ApprovalStatus.EXPIRED.value
                session.flush()
                session.expunge(row)
                return self._to_request(row)

            payload = json.loads(row.payload_json)
            payload["original_quantity"] = payload.get("quantity")
            payload["quantity"] = str(quantity)
            payload["modified_by"] = responder_id
            row.payload_json = json.dumps(payload, default=str)
            row.status = ApprovalStatus.APPROVED.value
            row.responded_at = now
            row.responder_id = responder_id
            row.response_note = f"size changed to {quantity}"
            session.flush()
            session.expunge(row)

        self.audit.record(
            "approval.modified",
            f"{responder_id} approved at a changed size: {quantity}",
            actor=responder_id,
            symbol=payload.get("symbol"),
            detail={"approval_id": approval_id, "payload": payload},
        )
        return self._to_request(row)

    def snooze(self, approval_id: int, responder_id: str) -> ApprovalRequest | None:
        """Extend the deadline once, up to :data:`MAX_SNOOZES` times."""
        now = self.clock.now()
        with self.db.session() as session:
            row = session.get(Approval, approval_id)
            if row is None:
                return None
            if ApprovalStatus(row.status) is not ApprovalStatus.PENDING or row.expires_at <= now:
                session.expunge(row)
                return self._to_request(row)

            payload = json.loads(row.payload_json)
            used = int(payload.get("snoozes_used", 0))
            if used >= MAX_SNOOZES:
                session.expunge(row)
                return self._to_request(row)

            payload["snoozes_used"] = used + 1
            row.payload_json = json.dumps(payload, default=str)
            row.expires_at = row.expires_at + dt.timedelta(minutes=SNOOZE_MINUTES)
            session.flush()
            session.expunge(row)

        self.audit.record(
            "approval.snoozed",
            f"{responder_id} snoozed for {SNOOZE_MINUTES}m ({used + 1}/{MAX_SNOOZES})",
            actor=responder_id,
            detail={"approval_id": approval_id},
        )
        return self._to_request(row)

    def expire_due(self) -> list[ApprovalRequest]:
        """Expire every request past its deadline. Returns what was expired.

        Run on a schedule. The read paths do not depend on it having run --
        this exists so the audit log and the user get an explicit "this timed
        out" rather than silence.
        """
        now = self.clock.now()
        expired: list[Approval] = []
        with self.db.session() as session:
            rows = list(
                session.scalars(
                    select(Approval).where(
                        Approval.status.in_(
                            [
                                ApprovalStatus.PENDING.value,
                                ApprovalStatus.AWAITING_MODIFICATION.value,
                            ]
                        ),
                        Approval.expires_at <= now,
                    )
                ).all()
            )
            for row in rows:
                row.status = ApprovalStatus.EXPIRED.value
                row.responded_at = now
                row.response_note = "timed out without a response; defaulted to REJECT"
                expired.append(row)
            session.flush()
            for row in rows:
                session.expunge(row)

        for row in expired:
            self.audit.record(
                "approval.expired",
                "Timed out without a response; defaulted to REJECT",
                actor="system",
                detail={"approval_id": row.id, "request_key": row.request_key},
            )
        return [self._to_request(row) for row in expired]

    def cancel(self, approval_id: int, reason: str) -> ApprovalRequest | None:
        """Withdraw a request the engine no longer needs answered."""
        with self.db.session() as session:
            row = session.get(Approval, approval_id)
            if row is None:
                return None
            if ApprovalStatus(row.status).is_terminal:
                session.expunge(row)
                return self._to_request(row)
            row.status = ApprovalStatus.CANCELLED.value
            row.responded_at = self.clock.now()
            row.response_note = reason
            session.flush()
            session.expunge(row)

        self.audit.record(
            "approval.cancelled", f"Cancelled: {reason}", actor="engine",
            detail={"approval_id": approval_id},
        )
        return self._to_request(row)

    # -------------------------------------------------------------- internals

    def _to_request(self, row: Approval) -> ApprovalRequest:
        """Map a row to a request, applying the deadline defensively.

        A row still marked PENDING past its deadline is reported as EXPIRED
        regardless of what the table says, so a stalled sweeper can never make
        a stale request look answerable.
        """
        status = ApprovalStatus(row.status)
        expires_at = _as_utc(row.expires_at)
        if not status.is_terminal and expires_at <= self.clock.now():
            status = ApprovalStatus.EXPIRED

        return ApprovalRequest(
            id=row.id,
            request_key=row.request_key,
            kind=ApprovalKind(row.kind),
            payload=json.loads(row.payload_json),
            status=status,
            expires_at=expires_at,
            created_at=_as_utc(row.created_at),
            responded_at=_as_utc(row.responded_at) if row.responded_at else None,
            responder_id=row.responder_id,
            response_note=row.response_note,
        )


def _as_utc(value: dt.datetime) -> dt.datetime:
    """SQLite returns naive datetimes; everything stored was UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
