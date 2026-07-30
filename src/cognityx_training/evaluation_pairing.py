"""Bounded SQLite index for deterministic prediction pairing."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Iterable, Iterator, Mapping

from cognityx_training.evaluation import PredictionPair
from cognityx_training.lineage import stable_json


class PredictionPairingStore:
    """Stream predictions into a disk-backed index and iterate stable pairs."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            handle = tempfile.NamedTemporaryFile(
                prefix="cognityx-evaluation-pairs-",
                suffix=".sqlite3",
                delete=False,
            )
            handle.close()
            path = Path(handle.name)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                candidate_id TEXT NOT NULL,
                record_id TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                duplicate_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (candidate_id, record_id, prediction_type)
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence_resolution (
                candidate_id TEXT NOT NULL,
                record_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (candidate_id, record_id)
            )
            """
        )
        self.connection.commit()
        self.inserted_rows = 0

    def ingest(
        self,
        candidate_id: str,
        prediction_type: str,
        rows: Iterable[Mapping[str, Any]],
    ) -> None:
        if prediction_type not in {"baseline", "candidate"}:
            raise ValueError(f"Unsupported prediction type: {prediction_type}")
        for row in rows:
            record_id = row.get("dataset_record_id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(
                    f"{prediction_type} prediction requires dataset_record_id"
                )
            payload = stable_json(dict(row))
            existing = self.connection.execute(
                """
                SELECT payload FROM predictions
                WHERE candidate_id = ? AND record_id = ? AND prediction_type = ?
                """,
                (candidate_id, record_id, prediction_type),
            ).fetchone()
            if existing is None:
                self.connection.execute(
                    """
                    INSERT INTO predictions
                    (candidate_id, record_id, prediction_type, payload)
                    VALUES (?, ?, ?, ?)
                    """,
                    (candidate_id, record_id, prediction_type, payload),
                )
                self.inserted_rows += 1
            elif existing[0] == payload:
                self.connection.execute(
                    """
                    UPDATE predictions
                    SET duplicate_count = duplicate_count + 1
                    WHERE candidate_id = ? AND record_id = ?
                      AND prediction_type = ?
                    """,
                    (candidate_id, record_id, prediction_type),
                )
            else:
                raise ValueError(
                    f"Conflicting {prediction_type} prediction for dataset "
                    f"record {record_id}"
                )
        self.connection.commit()

    def iter_pairs(self, candidate_id: str) -> Iterator[PredictionPair]:
        cursor = self.connection.execute(
            """
            SELECT record_id,
                   MAX(CASE WHEN prediction_type = 'baseline' THEN payload END),
                   MAX(CASE WHEN prediction_type = 'candidate' THEN payload END),
                   MAX(CASE WHEN prediction_type = 'baseline'
                            THEN duplicate_count ELSE 0 END),
                   MAX(CASE WHEN prediction_type = 'candidate'
                            THEN duplicate_count ELSE 0 END)
            FROM predictions
            WHERE candidate_id = ?
            GROUP BY record_id
            ORDER BY record_id
            """,
            (candidate_id,),
        )
        for record_id, baseline, candidate, baseline_duplicates, candidate_duplicates in cursor:
            yield PredictionPair(
                record_id=record_id,
                baseline=json.loads(baseline) if baseline is not None else None,
                candidate=json.loads(candidate) if candidate is not None else None,
                baseline_duplicate=bool(baseline_duplicates),
                candidate_duplicate=bool(candidate_duplicates),
            )

    def summary(self, candidate_id: str) -> dict[str, int]:
        rows = self.connection.execute(
            """
            SELECT COUNT(*),
                   SUM(CASE WHEN baseline_present = 1 AND candidate_present = 1
                            THEN 1 ELSE 0 END),
                   SUM(CASE WHEN baseline_present = 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN candidate_present = 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN baseline_duplicates > 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN candidate_duplicates > 0 THEN 1 ELSE 0 END)
            FROM (
                SELECT record_id,
                       MAX(prediction_type = 'baseline') AS baseline_present,
                       MAX(prediction_type = 'candidate') AS candidate_present,
                       MAX(CASE WHEN prediction_type = 'baseline'
                                THEN duplicate_count ELSE 0 END)
                           AS baseline_duplicates,
                       MAX(CASE WHEN prediction_type = 'candidate'
                                THEN duplicate_count ELSE 0 END)
                           AS candidate_duplicates
                FROM predictions
                WHERE candidate_id = ?
                GROUP BY record_id
            )
            """,
            (candidate_id,),
        ).fetchone()
        values = tuple(int(value or 0) for value in rows)
        return {
            "record_count": values[0],
            "paired_count": values[1],
            "missing_baseline_count": values[2],
            "missing_candidate_count": values[3],
            "duplicate_baseline_count": values[4],
            "duplicate_candidate_count": values[5],
        }

    def issues(self, candidate_id: str) -> Iterator[dict[str, Any]]:
        for pair in self.iter_pairs(candidate_id):
            if (
                pair.baseline is None
                or pair.candidate is None
                or pair.baseline_duplicate
                or pair.candidate_duplicate
            ):
                yield {
                    "dataset_record_id": pair.record_id,
                    "missing": (
                        "baseline"
                        if pair.baseline is None
                        else "candidate"
                        if pair.candidate is None
                        else None
                    ),
                    "baseline_duplicate": pair.baseline_duplicate,
                    "candidate_duplicate": pair.candidate_duplicate,
                }

    def put_evidence(
        self,
        candidate_id: str,
        record_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO evidence_resolution
            (candidate_id, record_id, payload) VALUES (?, ?, ?)
            """,
            (candidate_id, record_id, stable_json(dict(payload))),
        )
        self.connection.commit()

    def get_evidence(self, candidate_id: str, record_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            """
            SELECT payload FROM evidence_resolution
            WHERE candidate_id = ? AND record_id = ?
            """,
            (candidate_id, record_id),
        ).fetchone()
        if row is None:
            raise KeyError(
                f"Evidence resolution is absent for {candidate_id}/{record_id}"
            )
        return json.loads(row[0])

    def close(self, *, remove: bool = True) -> None:
        self.connection.close()
        if remove:
            self.path.unlink(missing_ok=True)
