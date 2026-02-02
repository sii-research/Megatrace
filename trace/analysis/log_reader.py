#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log reading and parsing utilities for distributed training analysis.
Extracted from DistributedLogAnalyzer so the analyzer can focus on
processing already-parsed rank entries.
"""

from __future__ import annotations

import bz2
import gzip
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class LogEntry:
    """Single log entry data structure."""

    raw_line: str
    node_ip: str
    hostname: str
    gpu_pci: str
    save_count: int
    timestamp: float
    rank: int
    function: str
    data_size: int
    stream: str
    op_count: int
    group_hash: Optional[str] = None

    def __str__(self) -> str:
        base = (
            f"[save_count {self.save_count}] [{self.timestamp}] [Rank {self.rank}] "
            f"Fun {self.function} Data {self.data_size} stream {self.stream} "
            f"opCount {self.op_count}"
        )
        return f"{base} groupHash {self.group_hash}" if self.group_hash else base


class LogReader:
    """Encapsulates log discovery, reading, and parsing logic."""

    LOG_EXTENSIONS = {".log", ".txt", ".out", ".err"}

    def __init__(
        self,
        log_path: Optional[str] = None,
        max_save_count_groups: Optional[int] = 2,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.log_path = Path(log_path) if log_path else None
        self.max_save_count_groups = (
            max_save_count_groups
            if (max_save_count_groups is None or max_save_count_groups >= 0)
            else None
        )
        self.log_files: List[Path] = []
        self.logger = logger or logging.getLogger(__name__)
        # Single flexible pattern: optional prefix [..][..]..., then last two brackets before save_count = node_ip, hostname.
        # Convention: any extra env vars (train_job_id, running_round, etc.) go before node_ip/hostname; only node_ip and hostname are required immediately before [save_count N].
        self.log_pattern = re.compile(
            r"(?:\[[^\]]+\]\s*)*"
            r"\[([^\]]+)\] \[([^\]]+)\] \[save_count (\d+)\] "
            r"\[([\d.]+)\] \[Rank (\d+)\] \[PCI ([^\]]+)\] "
            r"\[Fun[c]? (\w+)\] \[Data (\d+)\] \[stream (0x[0-9a-fA-F]+|\(nil\))\] \[opCount (\d+)\] "
            r"(?:\[groupHash (0x[0-9a-fA-F]+)\])?"
        )

    def discover_log_files(self) -> List[Path]:
        """
        Traverse log files in the configured folder.
        Returns a list containing all log file paths.
        """
        if not self.log_path:
            self.logger.error("Log path is not specified")
            self.log_files = []
            return self.log_files

        if self.log_path.is_file():
            if self.log_path.suffix.lower() in self.LOG_EXTENSIONS:
                self.log_files = [self.log_path]
                self.logger.info("Found single log file: %s", self.log_path)
            else:
                self.logger.warning("File %s is not a log file", self.log_path)
                self.log_files = []
        elif self.log_path.is_dir():
            collected: List[Path] = []
            for file_path in self.log_path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in self.LOG_EXTENSIONS:
                    collected.append(file_path)
            self.log_files = collected
            self.logger.info(
                "Found %d log files in directory %s", len(self.log_files), self.log_path
            )
        else:
            self.logger.error("Path %s does not exist", self.log_path)
            self.log_files = []

        return self.log_files

    def parse_log_files(self) -> Dict[int, List[LogEntry]]:
        """
        Read and parse log files into rank-indexed entries.
        Returns a dict keyed by rank with timestamp-sorted LogEntry lists.
        """
        rank_entries: Dict[int, List[LogEntry]] = defaultdict(list)

        for log_file in self.log_files:
            self.logger.info("Starting to parse file: %s", log_file)
            lines = self.read_log_file(log_file)
            for line in lines:
                entry = self.parse_log_line(line)
                if entry:
                    rank_entries[entry.rank].append(entry)

        for entries in rank_entries.values():
            entries.sort(key=lambda x: x.timestamp)

        self.logger.info("Parsing completed, found logs for %d ranks", len(rank_entries))
        for rank, entries in rank_entries.items():
            self.logger.info("Rank %d: %d records", rank, len(entries))

        return rank_entries

    def read_log_file(
        self, file_path: Path, max_save_count_groups: Optional[int] = None
    ) -> List[str]:
        """
        Read lines from a log file. When `max_save_count_groups` is positive,
        only the most recent save_count groups are loaded (best-effort for text files).
        """
        effective_groups = (
            self.max_save_count_groups if max_save_count_groups is None else max_save_count_groups
        )
        if effective_groups is None or effective_groups <= 0:
            return self._read_full_file(file_path)
        lines, used_partial = self._read_recent_savecount_lines(file_path, effective_groups)
        if used_partial:
            self.logger.info(
                "File %s read partially (latest %d save_count groups), total %d lines",
                file_path,
                effective_groups,
                len(lines),
            )
        return lines

    def parse_log_line(self, line: str) -> Optional[LogEntry]:
        """Parse a single log line. Prefix: optional [..] segments, then node_ip and hostname as last two before [save_count N]; body fixed."""
        match = self.log_pattern.match(line)
        if not match:
            return None
        node_ip = match.group(1).strip()
        hostname = match.group(2).strip()
        save_count = int(match.group(3))
        timestamp = float(match.group(4))
        rank = int(match.group(5))
        gpu_pci = match.group(6)
        function = match.group(7)
        data_size = int(match.group(8))
        stream = match.group(9)
        op_count = int(match.group(10))
        group_hash = match.group(11)
        if stream == "(nil)":
            stream = "0"
        return LogEntry(
            raw_line=line,
            node_ip=node_ip,
            hostname=hostname,
            gpu_pci=gpu_pci,
            save_count=save_count,
            timestamp=timestamp,
            rank=rank,
            function=function,
            data_size=data_size,
            stream=stream,
            op_count=op_count,
            group_hash=group_hash,
        )

    def _extract_save_count(self, line: str) -> Optional[int]:
        match = self.log_pattern.match(line)
        if not match:
            return None
        try:
            return int(match.group(3))
        except (TypeError, ValueError):
            return None

    def _read_recent_savecount_lines(
        self, file_path: Path, max_groups: int
    ) -> Tuple[List[str], bool]:
        """
        Read only the lines belonging to the most recent `max_groups` save_count values.
        Falls back to full read when the file is compressed or when parsing fails.
        """
        if max_groups <= 0:
            max_groups = 0
        if file_path.suffix.lower() in {".gz", ".bz2"} or max_groups == 0:
            return self._read_full_file(file_path), False

        chunk_size = 4096
        collected: List[str] = []
        save_counts_order: List[int] = []
        save_counts_set: Set[int] = set()
        stop_reading = False

        try:
            with open(file_path, "rb") as f:
                position = f.seek(0, os.SEEK_END)
                remainder = b""

                while position > 0 and not stop_reading:
                    read_size = min(chunk_size, position)
                    position -= read_size
                    f.seek(position)
                    chunk = f.read(read_size)
                    if not chunk:
                        break

                    data = chunk + remainder
                    parts = data.split(b"\n")
                    remainder = parts[0]

                    for part in reversed(parts[1:]):
                        if not part and not save_counts_order:
                            continue
                        line = part.decode("utf-8", errors="ignore").rstrip("\n\r")
                        if not line:
                            continue
                        save_count = self._extract_save_count(line)
                        if save_count is None:
                            continue
                        if save_count in save_counts_set:
                            collected.append(line)
                            continue
                        if len(save_counts_order) < max_groups:
                            save_counts_order.append(save_count)
                            save_counts_set.add(save_count)
                            collected.append(line)
                            continue
                        stop_reading = True
                        break

                if not stop_reading and remainder:
                    line = remainder.decode("utf-8", errors="ignore").rstrip("\n\r")
                    if line:
                        save_count = self._extract_save_count(line)
                        if save_count is not None and (
                            save_count in save_counts_set or len(save_counts_order) < max_groups
                        ):
                            if save_count not in save_counts_set:
                                save_counts_order.append(save_count)
                                save_counts_set.add(save_count)
                            collected.append(line)

        except Exception as exc:
            self.logger.error("Error reading file partially %s: %s", file_path, exc)
            return self._read_full_file(file_path), False

        collected.reverse()
        return collected, True

    def _read_full_file(self, file_path: Path) -> List[str]:
        lines: List[str] = []
        try:
            if file_path.suffix.lower() == ".gz":
                with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            elif file_path.suffix.lower() == ".bz2":
                with bz2.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            lines = [line.rstrip("\n\r") for line in lines]
            self.logger.info("File %s read completed, total %d lines", file_path, len(lines))
        except Exception as exc:
            self.logger.error("Error reading file %s: %s", file_path, exc)
            return []
        return lines
