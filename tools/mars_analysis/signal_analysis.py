from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd

from .edf_stream import EDFHeader, digital_to_microvolts, iter_signal_digital_chunks, read_signal_window


def compute_streaming_psd(
    header: EDFHeader,
    signal: int | str,
    *,
    chunk_seconds: float = 300.0,
    nperseg: int = 4096,
    max_hz: float = 60.0,
) -> pd.DataFrame:
    from scipy.signal import welch

    sample_rate = header.sample_rate(signal)
    max_samples = max(1, int(round(chunk_seconds * sample_rate)))
    spectra = []
    weights = []
    for digital in iter_signal_digital_chunks(header, signal, max_samples=max_samples):
        if len(digital) < max(8, nperseg // 2):
            continue
        values = digital_to_microvolts(header, signal, digital)
        freqs, power = welch(
            values,
            fs=sample_rate,
            nperseg=min(nperseg, len(values)),
            noverlap=min(nperseg // 2, max(0, len(values) // 2 - 1)),
        )
        mask = freqs <= max_hz
        spectra.append(power[mask])
        weights.append(len(values))
    if not spectra:
        return pd.DataFrame(columns=["FrequencyHz", "Power", "Signal"])
    matrix = np.vstack(spectra)
    avg = np.average(matrix, axis=0, weights=np.array(weights, dtype=float))
    return pd.DataFrame(
        {
            "FrequencyHz": freqs[mask],
            "Power": avg,
            "Signal": header.labels[header.signal_index(signal)],
        }
    )


def compute_hourly_streaming_psd(
    header: EDFHeader,
    signal: int | str,
    *,
    hour_bin_size: float = 1.0,
    chunk_seconds: float = 300.0,
    nperseg: int = 4096,
    max_hz: float = 60.0,
) -> pd.DataFrame:
    from scipy.signal import welch

    sample_rate = header.sample_rate(signal)
    max_samples = max(1, int(round(chunk_seconds * sample_rate)))
    bin_seconds = max(float(hour_bin_size) * 3600.0, 1.0)
    per_bin: dict[int, list[np.ndarray]] = {}
    weights: dict[int, list[int]] = {}
    freqs_by_bin: dict[int, np.ndarray] = {}
    elapsed = 0.0
    for digital in iter_signal_digital_chunks(header, signal, max_samples=max_samples):
        chunk_start = elapsed
        elapsed += len(digital) / sample_rate
        if len(digital) < max(8, nperseg // 2):
            continue
        values = digital_to_microvolts(header, signal, digital)
        freqs, power = welch(
            values,
            fs=sample_rate,
            nperseg=min(nperseg, len(values)),
            noverlap=min(nperseg // 2, max(0, len(values) // 2 - 1)),
        )
        mask = freqs <= max_hz
        hour_bin = int(chunk_start // bin_seconds)
        per_bin.setdefault(hour_bin, []).append(power[mask])
        weights.setdefault(hour_bin, []).append(len(values))
        freqs_by_bin[hour_bin] = freqs[mask]
    rows = []
    signal_label = header.labels[header.signal_index(signal)]
    for hour_bin, spectra in sorted(per_bin.items()):
        avg = np.average(np.vstack(spectra), axis=0, weights=np.array(weights[hour_bin], dtype=float))
        for freq, power in zip(freqs_by_bin[hour_bin], avg, strict=False):
            rows.append(
                {
                    "HourBin": hour_bin,
                    "BinStartSeconds": hour_bin * bin_seconds,
                    "BinSizeHours": hour_bin_size,
                    "FrequencyHz": float(freq),
                    "Power": float(power),
                    "Signal": signal_label,
                }
            )
    return pd.DataFrame(rows)


def compute_spectrogram_tiles(
    header: EDFHeader,
    signal: int | str,
    *,
    output_dir: str | Path,
    chunk_seconds: float = 300.0,
    nperseg: int = 1024,
    max_hz: float = 20.0,
) -> pd.DataFrame:
    from scipy.signal import spectrogram

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = header.sample_rate(signal)
    max_samples = max(1, int(round(chunk_seconds * sample_rate)))
    signal_label = header.labels[header.signal_index(signal)]
    index_rows = []
    elapsed = 0.0
    tile_idx = 0
    for digital in iter_signal_digital_chunks(header, signal, max_samples=max_samples):
        if len(digital) < max(8, nperseg // 2):
            elapsed += len(digital) / sample_rate
            continue
        values = digital_to_microvolts(header, signal, digital)
        freqs, times, power = spectrogram(
            values,
            fs=sample_rate,
            nperseg=min(nperseg, len(values)),
            noverlap=min(nperseg // 2, max(0, len(values) // 2 - 1)),
            scaling="density",
            mode="psd",
        )
        mask = freqs <= max_hz
        tile_path = output_dir / f"{signal_label}_tile_{tile_idx:04d}.npz"
        np.savez_compressed(
            tile_path,
            frequency_hz=freqs[mask],
            time_seconds=times + elapsed,
            power=power[mask, :],
        )
        index_rows.append(
            {
                "Signal": signal_label,
                "Tile": tile_idx,
                "Path": str(tile_path),
                "StartSeconds": float(elapsed),
                "EndSeconds": float(elapsed + len(values) / sample_rate),
                "FrequencyMaxHz": max_hz,
            }
        )
        elapsed += len(values) / sample_rate
        tile_idx += 1
    index = pd.DataFrame(index_rows)
    index.to_csv(output_dir / f"{signal_label}_spectrogram_index.csv", index=False)
    write_spectrogram_preview(index, output_dir / f"{signal_label}_spectrogram.html")
    return index


def write_spectrogram_preview(index: pd.DataFrame, output_path: str | Path, *, max_tiles: int = 96) -> None:
    output_path = Path(output_path)
    if index.empty:
        output_path.write_text("<html><body><h1>Spectrogram</h1><p>No tiles generated.</p></body></html>", encoding="utf-8")
        return
    try:
        import plotly.graph_objects as go

        powers = []
        times = []
        freqs = None
        for path in index["Path"].head(max_tiles):
            data = np.load(path)
            freqs = data["frequency_hz"]
            times.extend(data["time_seconds"].tolist())
            powers.append(data["power"])
        matrix = np.concatenate(powers, axis=1)
        matrix, times = _time_bin_spectrogram(matrix, np.asarray(times, dtype=float))
        fig = go.Figure(data=go.Heatmap(x=times, y=freqs, z=10 * np.log10(matrix + 1e-12), colorscale="Viridis"))
        fig.update_layout(title=f"Spectrogram preview ({min(len(index), max_tiles)} tile(s))", xaxis_title="Time (s)", yaxis_title="Frequency (Hz)")
        fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    except Exception as exc:  # noqa: BLE001
        output_path.write_text(
            f"<html><body><h1>Spectrogram preview unavailable</h1><p>{exc}</p>{index.to_html(index=False)}</body></html>",
            encoding="utf-8",
        )


def write_psd_plot(frame: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        plot_frame = frame.copy()
        plot_frame["FrequencyHz"] = pd.to_numeric(plot_frame["FrequencyHz"], errors="coerce")
        plot_frame["Power"] = pd.to_numeric(plot_frame["Power"], errors="coerce")
        plot_frame = plot_frame.dropna(subset=["FrequencyHz", "Power"])
        plot_frame = plot_frame[(plot_frame["FrequencyHz"] >= 0.5) & (plot_frame["FrequencyHz"] <= 60.0)]
        signals = list(plot_frame["Signal"].dropna().astype(str).unique())
        if not signals:
            output_path.write_text("<html><body><h1>Power spectra</h1><p>No displayable PSD rows.</p></body></html>", encoding="utf-8")
            return
        fig = make_subplots(rows=len(signals), cols=1, shared_xaxes=True, subplot_titles=signals, vertical_spacing=0.1)
        colors = ["#1f4fbf", "#d62728", "#2ca02c"]
        for row, signal in enumerate(signals, start=1):
            subset = plot_frame[plot_frame["Signal"].astype(str) == signal]
            color = colors[(row - 1) % len(colors)]
            if "HourBin" in subset.columns:
                for hour, group in subset.groupby("HourBin", dropna=False):
                    fig.add_trace(
                        go.Scatter(
                            x=group["FrequencyHz"],
                            y=group["Power"],
                            mode="lines",
                            line={"color": _rgba_from_hex(color, 0.22), "width": 0.8},
                            name=f"{signal} hour {hour}",
                            showlegend=False,
                        ),
                        row=row,
                        col=1,
                    )
                mean = subset.groupby("FrequencyHz", as_index=False)["Power"].mean()
                fig.add_trace(
                    go.Scatter(x=mean["FrequencyHz"], y=mean["Power"], mode="lines", line={"color": color, "width": 2.2}, name=f"{signal} mean"),
                    row=row,
                    col=1,
                )
            else:
                fig.add_trace(go.Scatter(x=subset["FrequencyHz"], y=subset["Power"], mode="lines", line={"color": color}, name=signal), row=row, col=1)
            fig.update_yaxes(title_text="uV^2/Hz", row=row, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", row=len(signals), col=1)
        fig.update_layout(
            title="Power spectra",
            template="plotly_white",
            height=max(420, 300 * len(signals)),
            font={"family": "Arial, Helvetica, sans-serif", "size": 12},
            margin={"l": 78, "r": 34, "t": 72, "b": 68},
        )
        fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    except Exception as exc:  # noqa: BLE001
        output_path.write_text(
            f"<html><body><h1>Power spectra unavailable</h1><p>{exc}</p>{frame.to_html(index=False)}</body></html>",
            encoding="utf-8",
        )


def write_epoch_inspector_plot(
    header: EDFHeader,
    eeg_signal: int | str,
    emg_signal: int | str,
    output_path: str | Path,
    *,
    epoch_index: int = 1,
    epoch_length_sec: float = 2.5,
    labels: dict[str, str] | None = None,
) -> None:
    start = max(0, epoch_index - 1) * epoch_length_sec
    eeg_t, eeg_v = read_signal_window(header, eeg_signal, start, epoch_length_sec)
    emg_t, emg_v = read_signal_window(header, emg_signal, start, epoch_length_sec)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eeg_x = eeg_t - eeg_t[0] if len(eeg_t) else eeg_t
        emg_x = emg_t - emg_t[0] if len(emg_t) else emg_t
        label_text = ""
        if labels:
            label_text = " | " + " | ".join(f"{name}: {value}" for name, value in labels.items())
        fig, axes = plt.subplots(2, 1, figsize=(10.8, 5.4), dpi=180, sharex=True)
        axes[0].plot(eeg_x, eeg_v, color="#111111", linewidth=0.8)
        axes[1].plot(emg_x, emg_v, color="#d62728", linewidth=0.8)
        axes[0].set_ylabel("EEG (uV)")
        axes[1].set_ylabel("EMG (uV)")
        axes[1].set_xlabel("Seconds in epoch")
        for ax in axes:
            ax.grid(True, color="#e8edf5", linewidth=0.55)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
        title = f"Epoch {epoch_index} raw window ({start:.1f}s){label_text}"
        fig.suptitle(title, x=0.045, y=0.995, ha="left", fontsize=12, fontweight="bold")
        fig.subplots_adjust(top=0.88, bottom=0.12, hspace=0.22)
        for suffix in (".png", ".svg"):
            fig.savefig(output_path.with_suffix(suffix), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        asset = output_path.with_suffix(".svg") if output_path.with_suffix(".svg").exists() else output_path.with_suffix(".png")
        output_path.write_text(
            f"""
<html>
<head>
<title>{html.escape(title)}</title>
<style>
body {{ margin:0; background:#fff; color:#111111; font-family:Arial,Helvetica,sans-serif; }}
main {{ padding:18px 22px; }}
h1 {{ font-size:20px; margin:0 0 12px; }}
img {{ width:100%; height:auto; display:block; }}
</style>
</head>
<body><main><img src="{html.escape(asset.name)}" alt="{html.escape(title)}" /></main></body>
</html>
""",
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        frame = pd.DataFrame({"EEGTime": eeg_t[:1000], "EEG": eeg_v[:1000]})
        output_path.write_text(
            f"<html><body><h1>Epoch inspector unavailable</h1><p>{html.escape(str(exc))}</p>{frame.to_html(index=False)}</body></html>",
            encoding="utf-8",
        )


def _rgba_from_hex(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    red, green, blue = tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def _time_bin_spectrogram(matrix: np.ndarray, times: np.ndarray, *, max_columns: int = 288) -> tuple[np.ndarray, np.ndarray]:
    """Average adjacent STFT columns for a responsive, scientifically readable preview."""
    if matrix.shape[1] <= max_columns:
        return matrix, times
    groups = np.array_split(np.arange(matrix.shape[1]), max_columns)
    return (
        np.column_stack([matrix[:, group].mean(axis=1) for group in groups]),
        np.asarray([times[group].mean() for group in groups], dtype=float),
    )


def write_spectrogram_hour_index(index: pd.DataFrame, output_path: str | Path, *, hour_bin_size: float = 1.0) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if index.empty:
        output_path.write_text("<html><body><h1>Hourly spectrogram tiles</h1><p>No tiles generated.</p></body></html>", encoding="utf-8")
        return
    frame = index.copy()
    bin_seconds = max(float(hour_bin_size) * 3600.0, 1.0)
    frame["HourBin"] = (pd.to_numeric(frame["StartSeconds"], errors="coerce") // bin_seconds).astype("Int64")
    counts = frame.groupby(["Signal", "HourBin"], dropna=False).size().rename("TileCount").reset_index()
    counts["BinStartSeconds"] = counts["HourBin"].astype(float) * bin_seconds
    output_path.write_text(
        "<html><body><h1>Hourly spectrogram tile index</h1>"
        + counts.to_html(index=False, escape=True)
        + "</body></html>",
        encoding="utf-8",
    )
