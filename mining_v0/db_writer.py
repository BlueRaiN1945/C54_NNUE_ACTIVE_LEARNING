from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from uci_parser import SearchResult, StaticEvalResult


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


class MiningDB:
    def __init__(self, path: str | Path):
        self.path = str(path)

        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row

        self.db.execute("PRAGMA foreign_keys = ON")

        enabled = self.db.execute(
            "PRAGMA foreign_keys"
        ).fetchone()[0]

        if enabled != 1:
            raise RuntimeError(
                "SQLite foreign_keys could not be enabled"
            )

    # ========================================================
    # Lifetime
    # ========================================================

    def close(self) -> None:
        self.db.close()

    def commit(self) -> None:
        self.db.commit()

    def rollback(self) -> None:
        self.db.rollback()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.db.commit()
        else:
            self.db.rollback()

        self.close()
        return False

    # ========================================================
    # Identity tables
    # ========================================================

    def register_engine(
        self,
        *,
        name: str,
        sha256: str,
        stockfish_commit: str | None,
        executable_path: str | None,
    ) -> int:
        self.db.execute(
            """
            INSERT OR IGNORE INTO engines (
                name,
                sha256,
                stockfish_commit,
                executable_path,
                created_utc
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                name,
                sha256,
                stockfish_commit,
                executable_path,
                utc_now(),
            ),
        )

        row = self.db.execute(
            """
            SELECT engine_id
            FROM engines
            WHERE sha256 = ?
            """,
            (sha256,),
        ).fetchone()

        if row is None:
            raise RuntimeError(
                "Failed to register engine"
            )

        return int(row["engine_id"])

    def register_network(
        self,
        *,
        role_hint: str | None,
        name: str,
        sha256: str,
        path: str | None,
        description: str | None = None,
    ) -> int:
        self.db.execute(
            """
            INSERT OR IGNORE INTO networks (
                role_hint,
                name,
                sha256,
                path,
                description,
                created_utc
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                role_hint,
                name,
                sha256,
                path,
                description,
                utc_now(),
            ),
        )

        row = self.db.execute(
            """
            SELECT network_id
            FROM networks
            WHERE sha256 = ?
            """,
            (sha256,),
        ).fetchone()

        if row is None:
            raise RuntimeError(
                "Failed to register network"
            )

        return int(row["network_id"])

    # ========================================================
    # Position identity
    # ========================================================

    def register_position(
        self,
        *,
        fen: str,
        source: str | None = None,
        split: str | None = None,
        original_score_type: str | None = None,
        original_score_value: int | None = None,
        original_wdl_w: int | None = None,
        original_wdl_d: int | None = None,
        original_wdl_l: int | None = None,
    ) -> int:
        fields = fen.split()

        if len(fields) != 6:
            raise ValueError(
                f"Expected six-field FEN, got {len(fields)} fields"
            )

        position_key = " ".join(fields[:4])
        side_to_move = fields[1]

        try:
            halfmove_clock = int(fields[4])
            fullmove_number = int(fields[5])
        except ValueError as exc:
            raise ValueError(
                "Invalid FEN move counters"
            ) from exc

        fen_hash = sha256_text(fen)

        self.db.execute(
            """
            INSERT OR IGNORE INTO positions (
                fen,
                fen_sha256,
                position_key,
                side_to_move,
                halfmove_clock,
                fullmove_number,
                source,
                split,
                original_score_type,
                original_score_value,
                original_wdl_w,
                original_wdl_d,
                original_wdl_l,
                created_utc
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                fen,
                fen_hash,
                position_key,
                side_to_move,
                halfmove_clock,
                fullmove_number,
                source,
                split,
                original_score_type,
                original_score_value,
                original_wdl_w,
                original_wdl_d,
                original_wdl_l,
                utc_now(),
            ),
        )

        row = self.db.execute(
            """
            SELECT position_id
            FROM positions
            WHERE fen = ?
            """,
            (fen,),
        ).fetchone()

        if row is None:
            raise RuntimeError(
                "Failed to register position"
            )

        return int(row["position_id"])

    # ========================================================
    # Mining run
    # ========================================================

    def create_run(
        self,
        *,
        run_tag: str,
        engine_id: int,
        candidate_network_id: int,
        official_network_id: int,
        teacher_network_id: int,
        candidate_nodes: int,
        candidate_multipv: int,
        teacher_nodes: int,
        threads: int,
        hash_mb: int,
        description: str | None = None,
    ) -> int:
        self.db.execute(
            """
            INSERT OR IGNORE INTO mining_runs (
                run_tag,
                created_utc,
                engine_id,
                candidate_network_id,
                official_network_id,
                teacher_network_id,
                candidate_nodes,
                candidate_multipv,
                teacher_nodes,
                threads,
                hash_mb,
                uci_show_wdl,
                description
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                run_tag,
                utc_now(),
                engine_id,
                candidate_network_id,
                official_network_id,
                teacher_network_id,
                candidate_nodes,
                candidate_multipv,
                teacher_nodes,
                threads,
                hash_mb,
                description,
            ),
        )

        row = self.db.execute(
            """
            SELECT *
            FROM mining_runs
            WHERE run_tag = ?
            """,
            (run_tag,),
        ).fetchone()

        if row is None:
            raise RuntimeError(
                "Failed to create mining run"
            )

        # Protect provenance if somebody accidentally reuses a tag.
        expected = {
            "engine_id": engine_id,
            "candidate_network_id": candidate_network_id,
            "official_network_id": official_network_id,
            "teacher_network_id": teacher_network_id,
            "candidate_nodes": candidate_nodes,
            "candidate_multipv": candidate_multipv,
            "teacher_nodes": teacher_nodes,
            "threads": threads,
            "hash_mb": hash_mb,
        }

        for key, value in expected.items():
            if row[key] != value:
                raise RuntimeError(
                    f"run_tag {run_tag!r} already exists "
                    f"with different {key}: "
                    f"{row[key]!r} != {value!r}"
                )

        return int(row["run_id"])

    # ========================================================
    # Search jobs/results
    # ========================================================

    def begin_search_job(
        self,
        *,
        run_id: int,
        position_id: int,
        engine_role: str,
        engine_id: int,
        network_id: int,
        search_kind: str,
        restricted_move: str,
        requested_nodes: int,
        requested_multipv: int,
    ) -> int:
        self.db.execute(
            """
            INSERT INTO search_jobs (
                run_id,
                position_id,
                engine_role,
                engine_id,
                network_id,
                search_kind,
                restricted_move,
                requested_nodes,
                requested_multipv,
                status,
                started_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)

            ON CONFLICT (
                run_id,
                position_id,
                engine_role,
                search_kind,
                restricted_move
            )
            DO UPDATE SET
                engine_id = excluded.engine_id,
                network_id = excluded.network_id,
                requested_nodes = excluded.requested_nodes,
                requested_multipv = excluded.requested_multipv,
                status = 'running',
                bestmove = NULL,
                ponder = NULL,
                error_text = NULL,
                started_utc = excluded.started_utc,
                finished_utc = NULL
            """,
            (
                run_id,
                position_id,
                engine_role,
                engine_id,
                network_id,
                search_kind,
                restricted_move,
                requested_nodes,
                requested_multipv,
                utc_now(),
            ),
        )

        row = self.db.execute(
            """
            SELECT job_id
            FROM search_jobs
            WHERE run_id = ?
              AND position_id = ?
              AND engine_role = ?
              AND search_kind = ?
              AND restricted_move = ?
            """,
            (
                run_id,
                position_id,
                engine_role,
                search_kind,
                restricted_move,
            ),
        ).fetchone()

        if row is None:
            raise RuntimeError(
                "Failed to create search job"
            )

        job_id = int(row["job_id"])

        # A rerun of the same logical job replaces old evidence.
        self.db.execute(
            """
            DELETE FROM search_results
            WHERE job_id = ?
            """,
            (job_id,),
        )

        return job_id

    def finish_search_job(
        self,
        *,
        job_id: int,
        result: SearchResult,
    ) -> None:
        for rank, info in sorted(result.infos.items()):
            self.db.execute(
                """
                INSERT INTO search_results (
                    job_id,
                    rank,
                    move,
                    score_type,
                    score_value,
                    bound,
                    wdl_w,
                    wdl_d,
                    wdl_l,
                    depth,
                    seldepth,
                    nodes,
                    time_ms,
                    nps,
                    pv
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    rank,
                    info.move,
                    info.score_type,
                    info.score_value,
                    info.bound,
                    info.wdl_w,
                    info.wdl_d,
                    info.wdl_l,
                    info.depth,
                    info.seldepth,
                    info.nodes,
                    info.time_ms,
                    info.nps,
                    info.pv,
                ),
            )

        status = (
            "ok"
            if result.bestmove is not None
            else "no_legal_moves"
        )

        self.db.execute(
            """
            UPDATE search_jobs
            SET status = ?,
                bestmove = ?,
                ponder = ?,
                finished_utc = ?
            WHERE job_id = ?
            """,
            (
                status,
                result.bestmove,
                result.ponder,
                utc_now(),
                job_id,
            ),
        )

    def fail_search_job(
        self,
        *,
        job_id: int,
        error_text: str,
    ) -> None:
        self.db.execute(
            """
            UPDATE search_jobs
            SET status = 'error',
                error_text = ?,
                finished_utc = ?
            WHERE job_id = ?
            """,
            (
                error_text,
                utc_now(),
                job_id,
            ),
        )

    # ========================================================
    # Static evidence
    # ========================================================

    def store_static_eval(
        self,
        *,
        run_id: int,
        position_id: int,
        network_role: str,
        network_id: int,
        move: str,
        child_fen: str | None,
        result: StaticEvalResult,
        score_root_pov: int | None,
    ) -> None:
        self.db.execute(
            """
            INSERT INTO static_evals (
                run_id,
                position_id,
                network_role,
                network_id,
                move,
                child_fen,
                status,
                score_raw_stm,
                score_root_pov,
                error_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT (
                run_id,
                position_id,
                network_role,
                move
            )
            DO UPDATE SET
                network_id = excluded.network_id,
                child_fen = excluded.child_fen,
                status = excluded.status,
                score_raw_stm = excluded.score_raw_stm,
                score_root_pov = excluded.score_root_pov,
                error_text = excluded.error_text
            """,
            (
                run_id,
                position_id,
                network_role,
                network_id,
                move,
                child_fen,
                result.status,
                result.score_raw_stm,
                score_root_pov,
                result.error_text,
            ),
        )
