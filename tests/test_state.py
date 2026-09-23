import tempfile
import unittest
from pathlib import Path

from medium_pc_audit.state import (
    ACCEPTED,
    QUARANTINED,
    RECEIVED,
    REJECTED_DUPLICATE,
    StateHistoryError,
    append_state_event,
    read_state_history,
    state_history_path,
)


RUN_ID = "AUDIT_V0001__TEST__run001"


class StateHistoryTests(unittest.TestCase):

    def test_received_then_accepted_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            append_state_event(
                td,
                run_id=RUN_ID,
                state=RECEIVED,
                recorded_at="2026-09-23T09:00:00Z",
            )

            append_state_event(
                td,
                run_id=RUN_ID,
                state=ACCEPTED,
                recorded_at="2026-09-23T09:00:01Z",
            )

            history = read_state_history(
                td,
                RUN_ID,
            )

            self.assertEqual(
                [event["state"] for event in history],
                [RECEIVED, ACCEPTED],
            )

    def test_append_preserves_existing_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            append_state_event(
                td,
                run_id=RUN_ID,
                state=QUARANTINED,
                recorded_at="2026-09-23T09:00:00Z",
                reason_codes=["broken"],
            )

            path = state_history_path(
                td,
                RUN_ID,
            )

            before = path.read_bytes()

            append_state_event(
                td,
                run_id=RUN_ID,
                state=REJECTED_DUPLICATE,
                recorded_at="2026-09-23T09:00:01Z",
                reason_codes=["duplicate"],
            )

            after = path.read_bytes()

            self.assertTrue(
                after.startswith(before)
            )
            self.assertGreater(
                len(after),
                len(before),
            )

    def test_terminal_run_only_allows_duplicate_afterward(self):
        with tempfile.TemporaryDirectory() as td:
            append_state_event(
                td,
                run_id=RUN_ID,
                state=RECEIVED,
                recorded_at="2026-09-23T09:00:00Z",
            )

            append_state_event(
                td,
                run_id=RUN_ID,
                state=ACCEPTED,
                recorded_at="2026-09-23T09:00:01Z",
            )

            with self.assertRaises(
                StateHistoryError
            ):
                append_state_event(
                    td,
                    run_id=RUN_ID,
                    state=QUARANTINED,
                    recorded_at="2026-09-23T09:00:02Z",
                )

            append_state_event(
                td,
                run_id=RUN_ID,
                state=REJECTED_DUPLICATE,
                recorded_at="2026-09-23T09:00:03Z",
            )

    def test_accepted_cannot_be_first_state(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(
                StateHistoryError
            ):
                append_state_event(
                    td,
                    run_id=RUN_ID,
                    state=ACCEPTED,
                    recorded_at="2026-09-23T09:00:00Z",
                )

    def test_unsafe_run_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(
                StateHistoryError
            ):
                append_state_event(
                    td,
                    run_id="../escape",
                    state=QUARANTINED,
                    recorded_at="2026-09-23T09:00:00Z",
                )


if __name__ == "__main__":
    unittest.main()
