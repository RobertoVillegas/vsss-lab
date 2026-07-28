"""Atomic schema-versioned policy registry and matchmaking."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

REGISTRY_SCHEMA = 1
PolicyCategory = Literal["main", "historical", "heuristic", "random", "exploiter", "friend"]
PolicyStatus = Literal["candidate", "main", "historical", "fixture"]


@dataclass(frozen=True)
class PolicyEntry:
    policy_id: str
    version: int
    category: PolicyCategory
    status: PolicyStatus
    checkpoint: str | None
    checkpoint_sha256: str | None
    algorithm: str
    rating: float
    parent: str | None
    created_at: str
    training_iteration: int

    @property
    def key(self) -> str:
        return f"{self.policy_id}@{self.version}"

    @classmethod
    def from_checkpoint(
        cls,
        *,
        policy_id: str,
        version: int,
        category: PolicyCategory,
        status: PolicyStatus,
        checkpoint: Path,
        algorithm: str,
        rating: float,
        parent: str | None,
        created_at: str,
        training_iteration: int,
    ) -> PolicyEntry:
        return cls(
            policy_id=policy_id,
            version=version,
            category=category,
            status=status,
            checkpoint=str(checkpoint.resolve()),
            checkpoint_sha256=_sha256(checkpoint),
            algorithm=algorithm,
            rating=rating,
            parent=parent,
            created_at=created_at,
            training_iteration=training_iteration,
        )


class LeagueRegistry:
    """Single-writer local registry with atomic persistence."""

    def __init__(self, path: Path, entries: tuple[PolicyEntry, ...] = ()) -> None:
        self.path = path
        self.entries = entries

    @classmethod
    def load(cls, path: Path) -> LeagueRegistry:
        if not path.exists():
            return cls(path)
        document = json.loads(path.read_text())
        if document.get("schema_version") != REGISTRY_SCHEMA:
            raise ValueError("unsupported league registry schema")
        return cls(path, tuple(PolicyEntry(**entry) for entry in document["policies"]))

    def register(self, entry: PolicyEntry) -> None:
        if any(existing.key == entry.key for existing in self.entries):
            raise ValueError(f"duplicate policy entry: {entry.key}")
        if entry.checkpoint is not None:
            checkpoint = Path(entry.checkpoint)
            if not checkpoint.is_file() or _sha256(checkpoint) != entry.checkpoint_sha256:
                raise ValueError("checkpoint hash mismatch")
        self.entries = (*self.entries, entry)
        self.save()

    def save(self) -> None:
        document = {
            "schema_version": REGISTRY_SCHEMA,
            "policies": [asdict(entry) for entry in self.entries],
        }
        payload = json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            dir=self.path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def current_main(self) -> PolicyEntry:
        mains = [entry for entry in self.entries if entry.status == "main"]
        if len(mains) != 1:
            raise ValueError("league must contain exactly one main policy")
        return mains[0]

    def promote(self, candidate_key: str) -> None:
        candidate = self.get(candidate_key)
        if candidate.status != "candidate":
            raise ValueError("only a candidate can be promoted")
        current = self.current_main()
        self.entries = tuple(
            replace(entry, status="historical", category="historical")
            if entry.key == current.key
            else replace(entry, status="main", category="main")
            if entry.key == candidate.key
            else entry
            for entry in self.entries
        )
        self.save()

    def get(self, key: str) -> PolicyEntry:
        try:
            return next(entry for entry in self.entries if entry.key == key)
        except StopIteration as error:
            raise KeyError(key) from error

    def select_opponent(
        self,
        *,
        seed: int,
        weights: dict[PolicyCategory, float],
        exclude: frozenset[str] = frozenset(),
    ) -> PolicyEntry:
        eligible = [
            entry
            for entry in self.entries
            if entry.key not in exclude and weights.get(entry.category, 0.0) > 0.0
        ]
        eligible.sort(key=lambda entry: entry.key)
        if not eligible:
            raise ValueError("no eligible league opponents")
        realized_weights = [weights[entry.category] for entry in eligible]
        return random.Random(seed).choices(eligible, weights=realized_weights, k=1)[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
