from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class EDFHeader:
    path: str
    header_bytes: int
    data_records: int
    record_duration: float
    signal_count: int
    labels: list[str]
    physical_dimensions: list[str]
    physical_min: np.ndarray
    physical_max: np.ndarray
    digital_min: np.ndarray
    digital_max: np.ndarray
    samples_per_record: np.ndarray

    @property
    def bytes_per_record(self) -> int:
        return int(np.sum(self.samples_per_record) * 2)

    @property
    def duration_seconds(self) -> float:
        return self.data_records * self.record_duration

    def sample_rate(self, signal: int | str) -> float:
        index = self.signal_index(signal)
        return float(self.samples_per_record[index] / self.record_duration)

    def signal_index(self, signal: int | str) -> int:
        if isinstance(signal, int):
            if 0 <= signal < self.signal_count:
                return signal
            if 1 <= signal <= self.signal_count:
                return signal - 1
            raise IndexError(f"Signal index out of range: {signal}")
        normalized = str(signal).strip().lower()
        for idx, label in enumerate(self.labels):
            if label.strip().lower() == normalized:
                return idx
        raise KeyError(f"Signal label not found: {signal}")


def read_edf_header(path: str | Path) -> EDFHeader:
    path = Path(path)
    with path.open("rb") as handle:
        fixed = handle.read(256)
        if len(fixed) != 256:
            raise ValueError(f"Not enough bytes for EDF fixed header: {path}")
        header_bytes = _parse_int(fixed[184:192])
        data_records = _parse_int(fixed[236:244])
        record_duration = _parse_float(fixed[244:252])
        signal_count = _parse_int(fixed[252:256])
        variable = handle.read(header_bytes - 256)
    if data_records <= 0:
        raise ValueError(f"EDF has non-positive data record count: {path}")
    if signal_count <= 0:
        raise ValueError(f"EDF has non-positive signal count: {path}")
    fields = _split_variable_header(variable, signal_count)
    return EDFHeader(
        path=str(path),
        header_bytes=header_bytes,
        data_records=data_records,
        record_duration=record_duration,
        signal_count=signal_count,
        labels=[value.strip() for value in fields["label"]],
        physical_dimensions=[value.strip() for value in fields["physical_dimension"]],
        physical_min=np.array([float(v or 0) for v in fields["physical_min"]], dtype=float),
        physical_max=np.array([float(v or 0) for v in fields["physical_max"]], dtype=float),
        digital_min=np.array([float(v or 0) for v in fields["digital_min"]], dtype=float),
        digital_max=np.array([float(v or 0) for v in fields["digital_max"]], dtype=float),
        samples_per_record=np.array([int(float(v or 0)) for v in fields["samples_per_record"]], dtype=int),
    )


def iter_signal_digital_chunks(
    header: EDFHeader,
    signal: int | str,
    *,
    start_record: int = 0,
    chunk_records: int = 64,
    max_samples: int = 2_000_000,
) -> Iterable[np.ndarray]:
    signal_index = header.signal_index(signal)
    samples_before = int(np.sum(header.samples_per_record[:signal_index]))
    samples = int(header.samples_per_record[signal_index])
    with Path(header.path).open("rb") as handle:
        for record_start in range(start_record, header.data_records, chunk_records):
            count = min(chunk_records, header.data_records - record_start)
            offset = header.header_bytes + record_start * header.bytes_per_record
            for rec in range(count):
                signal_offset = offset + rec * header.bytes_per_record + samples_before * 2
                for sample_start in range(0, samples, max_samples):
                    sample_count = min(max_samples, samples - sample_start)
                    handle.seek(signal_offset + sample_start * 2)
                    raw = handle.read(sample_count * 2)
                    if len(raw) != sample_count * 2:
                        raise ValueError(f"Short EDF read in {header.path}")
                    yield np.frombuffer(raw, dtype="<i2").copy()


def digital_to_physical(header: EDFHeader, signal: int | str, digital: np.ndarray) -> np.ndarray:
    idx = header.signal_index(signal)
    dig_min = header.digital_min[idx]
    dig_max = header.digital_max[idx]
    phys_min = header.physical_min[idx]
    phys_max = header.physical_max[idx]
    if dig_max == dig_min:
        return digital.astype(float)
    scale = (phys_max - phys_min) / (dig_max - dig_min)
    return (digital.astype(float) - dig_min) * scale + phys_min


def unit_scale_to_microvolts(header: EDFHeader, signal: int | str) -> float:
    idx = header.signal_index(signal)
    unit = header.physical_dimensions[idx].strip().lower().replace("µ", "u")
    if unit in {"mv", "millivolt", "millivolts"}:
        return 1000.0
    if unit in {"uv", "uV".lower(), "microvolt", "microvolts"}:
        return 1.0
    if unit in {"v", "volt", "volts"}:
        return 1_000_000.0
    return 1.0


def physical_dimension(header: EDFHeader, signal: int | str) -> str:
    return header.physical_dimensions[header.signal_index(signal)]


def digital_to_microvolts(header: EDFHeader, signal: int | str, digital: np.ndarray) -> np.ndarray:
    return digital_to_physical(header, signal, digital) * unit_scale_to_microvolts(header, signal)


def read_signal_window(
    header: EDFHeader,
    signal: int | str,
    start_seconds: float,
    duration_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    signal_index = header.signal_index(signal)
    sample_rate = header.sample_rate(signal_index)
    start_sample = max(0, int(round(start_seconds * sample_rate)))
    sample_count = max(0, int(round(duration_seconds * sample_rate)))
    samples_per_record = int(header.samples_per_record[signal_index])
    start_record = start_sample // samples_per_record
    end_sample = start_sample + sample_count
    end_record = min(header.data_records, int(np.ceil(end_sample / samples_per_record)))
    if sample_count == 0 or start_record >= header.data_records:
        return np.array([]), np.array([])
    samples_before = int(np.sum(header.samples_per_record[:signal_index]))
    pieces = []
    remaining = sample_count
    with Path(header.path).open("rb") as handle:
        for record in range(start_record, end_record):
            local_start = start_sample - record * samples_per_record if record == start_record else 0
            local_start = max(0, local_start)
            available = samples_per_record - local_start
            take = min(available, remaining)
            if take <= 0:
                continue
            offset = (
                header.header_bytes
                + record * header.bytes_per_record
                + (samples_before + local_start) * 2
            )
            handle.seek(offset)
            raw = handle.read(take * 2)
            if len(raw) != take * 2:
                raise ValueError(f"Short EDF window read in {header.path}")
            pieces.append(np.frombuffer(raw, dtype="<i2").copy())
            remaining -= take
            if remaining <= 0:
                break
    if not pieces:
        return np.array([]), np.array([])
    segment = np.concatenate(pieces)
    values = digital_to_microvolts(header, signal_index, segment)
    times = start_seconds + np.arange(len(values)) / sample_rate
    return times, values


def envelope_for_signal(
    header: EDFHeader,
    signal: int | str,
    *,
    bin_seconds: float = 5.0,
    chunk_records: int = 64,
) -> pd.DataFrame:
    sample_rate = header.sample_rate(signal)
    bin_samples = max(1, int(round(sample_rate * bin_seconds)))
    rows = []
    pending = np.array([], dtype=np.int16)
    signal_index = header.signal_index(signal)
    bin_index = 0
    for chunk in iter_signal_digital_chunks(header, signal_index, chunk_records=chunk_records):
        if pending.size:
            chunk = np.concatenate([pending, chunk])
        usable = (chunk.size // bin_samples) * bin_samples
        reshaped = chunk[:usable].reshape(-1, bin_samples) if usable else np.empty((0, bin_samples))
        for row in reshaped:
            physical = digital_to_microvolts(header, signal_index, row)
            rows.append(
                {
                    "time_seconds": bin_index * bin_seconds,
                    "signal": header.labels[signal_index],
                    "min": float(np.min(physical)),
                    "max": float(np.max(physical)),
                    "mean": float(np.mean(physical)),
                    "rms": float(np.sqrt(np.mean(np.square(physical)))),
                    "unit": "uV",
                }
            )
            bin_index += 1
        pending = chunk[usable:]
    if pending.size:
        physical = digital_to_microvolts(header, signal_index, pending)
        rows.append(
            {
                "time_seconds": bin_index * bin_seconds,
                "signal": header.labels[signal_index],
                "min": float(np.min(physical)),
                "max": float(np.max(physical)),
                "mean": float(np.mean(physical)),
                "rms": float(np.sqrt(np.mean(np.square(physical)))),
                "unit": "uV",
            }
        )
    return pd.DataFrame(rows)


def _split_variable_header(variable: bytes, signal_count: int) -> dict[str, list[str]]:
    layout = [
        ("label", 16),
        ("transducer", 80),
        ("physical_dimension", 8),
        ("physical_min", 8),
        ("physical_max", 8),
        ("digital_min", 8),
        ("digital_max", 8),
        ("prefiltering", 80),
        ("samples_per_record", 8),
        ("reserved", 32),
    ]
    fields: dict[str, list[str]] = {}
    offset = 0
    for name, width in layout:
        values = []
        for _ in range(signal_count):
            raw = variable[offset : offset + width]
            values.append(raw.decode("ascii", errors="ignore").strip())
            offset += width
        fields[name] = values
    return fields


def _parse_int(raw: bytes) -> int:
    return int(raw.decode("ascii", errors="ignore").strip() or "0")


def _parse_float(raw: bytes) -> float:
    return float(raw.decode("ascii", errors="ignore").strip() or "0")
