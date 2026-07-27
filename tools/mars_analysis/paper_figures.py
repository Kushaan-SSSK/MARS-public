from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from .edf_stream import EDFHeader, digital_to_microvolts, iter_signal_digital_chunks, read_signal_window


STATE_COLORS = {
    "Wake": "#d62728",
    "NREM": "#1f4fbf",
    "REM": "#2ca02c",
    "DISCO-T": "#6a3d9a",
    "Unknown": "#7f7f7f",
}


def select_representative_epochs(
    predictions: pd.DataFrame,
    *,
    states: tuple[str, ...] = ("Wake", "NREM", "REM"),
) -> dict[str, int]:
    frame = predictions.copy()
    frame["EpochIndex"] = pd.to_numeric(frame["EpochIndex"], errors="coerce")
    frame["Confidence"] = pd.to_numeric(frame.get("Confidence", 0), errors="coerce").fillna(0)
    out: dict[str, int] = {}
    for state in states:
        subset = frame[frame["PredLabel"].astype(str) == state]
        if subset.empty:
            continue
        row = subset.sort_values(["Confidence", "EpochIndex"], ascending=[False, True]).iloc[0]
        out[state] = int(row["EpochIndex"])
    return out


def write_paper_trace_figure(
    *,
    recording_id: str,
    header: EDFHeader,
    eeg_signal: int | str,
    emg_signal: int | str,
    eeg_envelope: pd.DataFrame,
    emg_envelope: pd.DataFrame,
    predictions: pd.DataFrame,
    output_path: str | Path,
    epoch_length_sec: float,
    zoom_seconds: float = 8.0,
    zoom_epochs: list[int] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_static_trace_figure(
            recording_id=recording_id,
            header=header,
            eeg_signal=eeg_signal,
            emg_signal=emg_signal,
            eeg_envelope=eeg_envelope,
            emg_envelope=emg_envelope,
            predictions=predictions,
            output_path=output_path,
            epoch_length_sec=epoch_length_sec,
            zoom_seconds=zoom_seconds,
            zoom_epochs=zoom_epochs,
        )
        _write_static_asset_page(
            output_path,
            title=f"{recording_id} EEG/EMG trace overview",
            note="Paper-style static trace. Channel-specific EDF names are intentionally hidden; verify channels in the Overview tab.",
        )
    except Exception as exc:  # noqa: BLE001
        output_path.write_text(
            f"<html><body><h1>{recording_id} paper trace unavailable</h1><p>{exc}</p></body></html>",
            encoding="utf-8",
        )


def write_paper_spectral_figure(
    *,
    recording_id: str,
    spectrogram_index: pd.DataFrame,
    psd_frames: list[pd.DataFrame],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=("EEG spectrogram", "Power spectra"), horizontal_spacing=0.12)
        if not spectrogram_index.empty:
            powers = []
            times = []
            freqs = None
            for path in spectrogram_index["Path"].head(48):
                data = np.load(path)
                freqs = data["frequency_hz"]
                times.extend(data["time_seconds"].tolist())
                powers.append(data["power"])
            if powers and freqs is not None:
                matrix = np.concatenate(powers, axis=1)
                fig.add_trace(
                    go.Heatmap(x=times, y=freqs, z=10 * np.log10(matrix + 1e-12), colorscale="Viridis", colorbar={"title": "dB"}),
                    row=1,
                    col=1,
                )
        if psd_frames:
            psd = pd.concat(psd_frames, ignore_index=True)
            for signal, group in psd.groupby("Signal", dropna=False):
                fig.add_trace(
                    go.Scatter(x=group["FrequencyHz"], y=group["Power"], mode="lines", name=str(signal)),
                    row=1,
                    col=2,
                )
        fig.update_layout(
            title=f"{recording_id} spectral summary",
            template="plotly_white",
            height=520,
            font={"family": "Arial, Helvetica, sans-serif", "size": 12},
        )
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
        fig.update_yaxes(title_text="Power", row=1, col=2)
        fig.write_html(output_path, include_plotlyjs=True, full_html=True)
        try:
            _write_static_spectral_figure(
                recording_id=recording_id,
                spectrogram_index=spectrogram_index,
                psd_frames=psd_frames,
                output_path=output_path,
            )
        except Exception:
            _try_write_static(fig, output_path)
    except Exception as exc:  # noqa: BLE001
        output_path.write_text(
            f"<html><body><h1>{recording_id} spectral figure unavailable</h1><p>{exc}</p></body></html>",
            encoding="utf-8",
        )


def write_paper_spectrogram_figure(
    *,
    recording_id: str,
    spectrogram_index: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np
        import plotly.graph_objects as go

        matrix, freqs, times = _load_spectrogram_preview(spectrogram_index, max_tiles=96)
        if matrix is None or freqs is None or times is None:
            output_path.write_text("<html><body><h1>Spectrogram unavailable</h1><p>No spectrogram tiles found.</p></body></html>", encoding="utf-8")
            return
        db = 10 * np.log10(matrix + 1e-12)
        vmin, vmax = np.nanpercentile(db, [5, 99])
        fig = go.Figure(
            data=go.Heatmap(
                x=times / 3600.0,
                y=freqs,
                z=db,
                zmin=vmin,
                zmax=vmax,
                colorscale="Viridis",
                colorbar={"title": "Power (dB)"},
            )
        )
        fig.update_layout(
            title=f"{recording_id} EEG spectrogram",
            template="plotly_white",
            height=620,
            font={"family": "Arial, Helvetica, sans-serif", "size": 12},
            margin={"l": 78, "r": 34, "t": 72, "b": 68},
            xaxis_title="Time from recording start (h)",
            yaxis_title="Frequency (Hz)",
        )
        fig.update_yaxes(range=[0, float(max(freqs))])
        fig.write_html(output_path, include_plotlyjs=True, full_html=True)
        _write_static_spectrogram_figure(recording_id=recording_id, spectrogram_index=spectrogram_index, output_path=output_path)
    except Exception as exc:  # noqa: BLE001
        output_path.write_text(
            f"<html><body><h1>{recording_id} spectrogram unavailable</h1><p>{exc}</p></body></html>",
            encoding="utf-8",
        )


def write_paper_psd_figure(
    *,
    recording_id: str,
    psd_frames: list[pd.DataFrame],
    output_path: str | Path,
    header: EDFHeader | None = None,
    eeg_signal: int | str | None = None,
    emg_signal: int | str | None = None,
    predictions: pd.DataFrame | None = None,
    epoch_length_sec: float | None = None,
    min_hz: float = 0.5,
    max_hz: float = 60.0,
    max_faint_curves_per_state: int = 2,
    show_sem_band: bool = True,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if header is not None and eeg_signal is not None and predictions is not None and epoch_length_sec:
            _write_static_state_psd_figure(
                recording_id=recording_id,
                header=header,
                signals=[("EEG", eeg_signal), ("EMG", emg_signal)] if emg_signal is not None else [("EEG", eeg_signal)],
                predictions=predictions,
                output_path=output_path,
                epoch_length_sec=float(epoch_length_sec),
                min_hz=min_hz,
                max_hz=max_hz,
                max_faint_curves_per_state=max_faint_curves_per_state,
                show_sem_band=show_sem_band,
            )
            _write_static_asset_page(
                output_path,
                title=f"{recording_id} state-conditioned power spectra",
                note="Thin lines are sampled label-aligned epochs; bold lines are state means.",
            )
            return
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        psd = pd.concat(psd_frames, ignore_index=True) if psd_frames else pd.DataFrame()
        psd = _filter_psd_for_display(psd, min_hz=min_hz, max_hz=max_hz)
        signals = list(psd["Signal"].dropna().astype(str).unique()) if not psd.empty else []
        if not signals:
            output_path.write_text("<html><body><h1>Power spectra unavailable</h1><p>No PSD rows found.</p></body></html>", encoding="utf-8")
            return
        fig = make_subplots(rows=len(signals), cols=1, shared_xaxes=True, subplot_titles=signals, vertical_spacing=0.11)
        for row, signal in enumerate(signals, start=1):
            subset = psd[psd["Signal"].astype(str) == signal]
            if "HourBin" in subset.columns:
                for hour, group in subset.groupby("HourBin", dropna=False):
                    fig.add_trace(
                        go.Scatter(
                            x=group["FrequencyHz"],
                            y=group["Power"],
                            mode="lines",
                            line={"width": 0.7, "color": "rgba(70,95,220,0.22)"},
                            name=f"{signal} hour {hour}",
                            showlegend=False,
                        ),
                        row=row,
                        col=1,
                    )
                mean = subset.groupby("FrequencyHz", as_index=False)["Power"].mean()
                fig.add_trace(
                    go.Scatter(x=mean["FrequencyHz"], y=mean["Power"], mode="lines", line={"width": 2.2}, name=f"{signal} mean"),
                    row=row,
                    col=1,
                )
            else:
                fig.add_trace(go.Scatter(x=subset["FrequencyHz"], y=subset["Power"], mode="lines", name=signal), row=row, col=1)
            fig.update_yaxes(title_text="uV^2/Hz", row=row, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", row=len(signals), col=1)
        fig.update_layout(
            title=f"{recording_id} power spectra",
            template="plotly_white",
            height=max(420, 300 * len(signals)),
            font={"family": "Arial, Helvetica, sans-serif", "size": 12},
            margin={"l": 78, "r": 34, "t": 72, "b": 68},
        )
        fig.write_html(output_path, include_plotlyjs=True, full_html=True)
        _write_static_psd_figure(recording_id=recording_id, psd_frames=psd_frames, output_path=output_path, min_hz=min_hz, max_hz=max_hz)
    except Exception as exc:  # noqa: BLE001
        output_path.write_text(
            f"<html><body><h1>{recording_id} PSD unavailable</h1><p>{exc}</p></body></html>",
            encoding="utf-8",
        )


def _add_envelope(fig, envelope: pd.DataFrame, *, row: int, color: str, name: str) -> None:
    import plotly.graph_objects as go

    fig.add_trace(
        go.Scatter(
            x=envelope["time_seconds"],
            y=envelope["max"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=envelope["time_seconds"],
            y=envelope["min"],
            mode="lines",
            fill="tonexty",
            fillcolor=_rgba(color, 0.32),
            line={"width": 0},
            name=name,
            hovertemplate="Time %{x:.1f}s<br>min/max envelope<extra></extra>",
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=envelope["time_seconds"],
            y=envelope["mean"],
            mode="lines",
            line={"color": color, "width": 0.8},
            showlegend=False,
        ),
        row=row,
        col=1,
    )


def _add_state_strip(fig, predictions: pd.DataFrame, *, row: int, epoch_length_sec: float) -> None:
    import plotly.graph_objects as go

    frame = predictions.copy()
    frame["EpochStartSeconds"] = pd.to_numeric(frame["EpochStartSeconds"], errors="coerce")
    frame = frame.iloc[:: max(1, len(frame) // 12000)]
    for state, group in frame.groupby("PredLabel", dropna=False):
        fig.add_trace(
            go.Bar(
                x=group["EpochStartSeconds"],
                y=[1] * len(group),
                width=epoch_length_sec,
                marker_color=STATE_COLORS.get(str(state), "#7f7f7f"),
                name=str(state),
                hovertemplate=f"{state}<br>%{{x:.1f}}s<extra></extra>",
            ),
            row=row,
            col=1,
        )


def _rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    red, green, blue = tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def _write_static_trace_figure(
    *,
    recording_id: str,
    header: EDFHeader,
    eeg_signal: int | str,
    emg_signal: int | str,
    eeg_envelope: pd.DataFrame,
    emg_envelope: pd.DataFrame,
    predictions: pd.DataFrame,
    output_path: Path,
    epoch_length_sec: float,
    zoom_seconds: float,
    zoom_epochs: list[int] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _apply_publication_matplotlib_style()
    reps = _resolve_zoom_epochs(predictions, zoom_epochs)
    zoom_count = max(1, len(reps))
    figure_height = 4.7 + 1.72 * zoom_count
    fig = plt.figure(figsize=(13.5, figure_height), dpi=180)
    grid = fig.add_gridspec(
        2 + zoom_count,
        1,
        height_ratios=[1.0, 1.0, *([0.92] * zoom_count)],
        hspace=0.72,
    )
    ax_eeg = fig.add_subplot(grid[0, 0])
    ax_emg = fig.add_subplot(grid[1, 0], sharex=ax_eeg)

    eeg_x, eeg_y = _downsample_signal_trace(header, eeg_signal)
    emg_x, emg_y = _downsample_signal_trace(header, emg_signal)
    _draw_full_trace_axis(ax_eeg, eeg_x, eeg_y, "#111111", "", "EEG (uV)")
    _draw_full_trace_axis(ax_emg, emg_x, emg_y, "#d62728", "", "EMG (uV)")

    duration_hours = max(float(header.duration_seconds) / 3600.0, 1e-9)
    ax_emg.set_xlabel("Time from recording start (h)", fontstyle="italic", labelpad=7)
    ax_eeg.set_xlim(0, duration_hours)
    for ax in (ax_eeg, ax_emg):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color="#e8edf5", linewidth=0.55)
        ax.tick_params(labelsize=8)

    for row_offset, (state, epoch) in enumerate(reps):
        ax = fig.add_subplot(grid[2 + row_offset, 0])
        start = max(0.0, (epoch - 1) * epoch_length_sec)
        eeg_t, eeg_v = read_signal_window(header, eeg_signal, start, zoom_seconds)
        emg_t, emg_v = read_signal_window(header, emg_signal, start, zoom_seconds)
        if len(eeg_t):
            x = eeg_t - eeg_t[0]
            ax.plot(x, _normalize_zoom_trace(eeg_v) + 0.58, color="#111111", linewidth=0.65)
        if len(emg_t):
            x = emg_t - emg_t[0]
            ax.plot(x, _normalize_zoom_trace(emg_v) - 0.58, color="#d62728", linewidth=0.65)
        ax.text(
            0.01,
            1.07,
            f"{state} epoch {epoch} ({start:.1f}s)",
            transform=ax.transAxes,
            color=STATE_COLORS.get(state, "#17213a"),
            fontweight="bold",
            va="bottom",
            clip_on=False,
        )
        ax.text(-0.012, 0.69, "EEG", transform=ax.transAxes, ha="right", va="center", fontweight="bold", fontsize=8)
        ax.text(-0.012, 0.30, "EMG", transform=ax.transAxes, ha="right", va="center", fontweight="bold", fontsize=8)
        ax.set_xlim(0, zoom_seconds)
        ax.set_ylim(-1.1, 1.1)
        ax.set_yticks([])
        ax.grid(True, axis="x", color="#e8edf5", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor(STATE_COLORS.get(state, "#95a3b8"))
        ax.tick_params(axis="x", labelsize=8)
        if row_offset == zoom_count - 1:
            ax.set_xlabel("Seconds in zoom window", fontstyle="italic")
        else:
            ax.set_xticklabels([])

    fig.suptitle(f"{recording_id} EEG/EMG trace overview", x=0.055, y=0.992, ha="left", fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.94, bottom=0.07)
    _save_static_matplotlib(fig, output_path)


def _write_static_spectral_figure(
    *,
    recording_id: str,
    spectrogram_index: pd.DataFrame,
    psd_frames: list[pd.DataFrame],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _apply_publication_matplotlib_style()
    import numpy as np

    fig = plt.figure(figsize=(12.8, 5.5), dpi=180)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.36, 1.0], wspace=0.22)
    ax_spec = fig.add_subplot(grid[0, 0])
    psd_grid = grid[0, 1].subgridspec(2, 1, hspace=0.28)
    ax_psd_a = fig.add_subplot(psd_grid[0, 0])
    ax_psd_b = fig.add_subplot(psd_grid[1, 0], sharex=ax_psd_a)

    if not spectrogram_index.empty:
        matrix, freqs, times = _load_spectrogram_preview(spectrogram_index)
        if matrix is not None and freqs is not None and times is not None:
            db = 10 * np.log10(matrix + 1e-12)
            vmin, vmax = np.nanpercentile(db, [3, 98])
            image = ax_spec.imshow(
                db,
                aspect="auto",
                origin="lower",
                extent=[float(np.min(times)) / 3600.0, float(np.max(times)) / 3600.0, float(np.min(freqs)), float(np.max(freqs))],
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            cbar = fig.colorbar(image, ax=ax_spec, fraction=0.046, pad=0.02)
            cbar.set_label("Power (dB)", fontsize=8)
            cbar.ax.tick_params(labelsize=7)
    ax_spec.set_title("EEG spectrogram", fontsize=11, fontweight="bold")
    ax_spec.set_xlabel("Time from recording start (h)", fontstyle="italic")
    ax_spec.set_ylabel("Frequency (Hz)")

    psd = pd.concat(psd_frames, ignore_index=True) if psd_frames else pd.DataFrame()
    if not psd.empty:
        signals = list(psd["Signal"].dropna().astype(str).unique())
        first_signal = signals[0] if signals else ""
        second_signal = signals[1] if len(signals) > 1 else ""
        _plot_psd_signal(ax_psd_a, psd, first_signal, "#1f4fbf")
        _plot_psd_signal(ax_psd_b, psd, second_signal, "#d62728")
        ax_psd_a.set_title(first_signal or "Power spectra", fontsize=10, fontweight="bold")
        ax_psd_b.set_title(second_signal or "Power spectra", fontsize=10, fontweight="bold")
    for ax in (ax_psd_a, ax_psd_b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color="#e8edf5", linewidth=0.55)
        ax.set_ylabel("Power")
        ax.tick_params(labelsize=8)
    ax_psd_b.set_xlabel("Frequency (Hz)", fontstyle="italic")
    fig.suptitle(f"{recording_id} spectral summary", x=0.055, y=0.99, ha="left", fontsize=13, fontweight="bold")
    _save_static_matplotlib(fig, output_path)


def _write_static_spectrogram_figure(
    *,
    recording_id: str,
    spectrogram_index: pd.DataFrame,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)
    matrix, freqs, times = _load_spectrogram_preview(spectrogram_index, max_tiles=96)
    if matrix is not None and freqs is not None and times is not None:
        db = 10 * np.log10(matrix + 1e-12)
        vmin, vmax = np.nanpercentile(db, [5, 99])
        image = ax.imshow(
            db,
            aspect="auto",
            origin="lower",
            extent=[float(np.min(times)) / 3600.0, float(np.max(times)) / 3600.0, float(np.min(freqs)), float(np.max(freqs))],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Power (dB)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    ax.set_title(f"{recording_id} EEG spectrogram", fontsize=12, fontweight="bold", loc="left")
    ax.set_xlabel("Time from recording start (h)", fontstyle="italic")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, 20)
    _save_static_matplotlib(fig, output_path)


def _write_static_psd_figure(
    *,
    recording_id: str,
    psd_frames: list[pd.DataFrame],
    output_path: Path,
    min_hz: float,
    max_hz: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _apply_publication_matplotlib_style()

    psd = pd.concat(psd_frames, ignore_index=True) if psd_frames else pd.DataFrame()
    psd = _filter_psd_for_display(psd, min_hz=min_hz, max_hz=max_hz)
    signals = list(psd["Signal"].dropna().astype(str).unique()) if not psd.empty else []
    if not signals:
        fig, ax = plt.subplots(figsize=(7, 2.2), dpi=180)
        ax.text(0.5, 0.5, "No PSD rows available", ha="center", va="center")
        ax.axis("off")
        _save_static_matplotlib(fig, output_path)
        return
    fig, axes = plt.subplots(max(1, len(signals)), 1, figsize=(8.5, max(3.2, 2.7 * max(1, len(signals)))), dpi=180, sharex=True)
    if len(signals) == 1:
        axes = [axes]
    colors = ["#1f4fbf", "#d62728", "#2ca02c"]
    for idx, signal in enumerate(signals):
        ax = axes[idx]
        subset = psd[psd["Signal"].astype(str) == signal]
        color = colors[idx % len(colors)]
        if "HourBin" in subset.columns:
            for _hour, group in subset.groupby("HourBin", dropna=False):
                ax.plot(group["FrequencyHz"], group["Power"], color=color, alpha=0.16, linewidth=0.55)
            mean = subset.groupby("FrequencyHz", as_index=False)["Power"].mean()
            ax.plot(mean["FrequencyHz"], mean["Power"], color=color, linewidth=1.6)
        else:
            ax.plot(subset["FrequencyHz"], subset["Power"], color=color, linewidth=1.2)
        ax.set_title(str(signal), fontsize=10, fontweight="bold")
        ax.set_ylabel("uV^2/Hz")
        ax.grid(True, color="#e8edf5", linewidth=0.55)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("Frequency (Hz)", fontstyle="italic")
    fig.suptitle(f"{recording_id} power spectra", x=0.02, y=0.995, ha="left", fontsize=13, fontweight="bold")
    _save_static_matplotlib(fig, output_path)


def _write_static_state_psd_figure(
    *,
    recording_id: str,
    header: EDFHeader,
    signals: list[tuple[str, int | str | None]],
    predictions: pd.DataFrame,
    output_path: Path,
    epoch_length_sec: float,
    min_hz: float,
    max_hz: float,
    max_faint_curves_per_state: int,
    show_sem_band: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _apply_publication_matplotlib_style()

    valid_signals = [(role, signal) for role, signal in signals if signal is not None]
    fig, axes = plt.subplots(
        max(1, len(valid_signals)),
        1,
        figsize=(11.8, max(4.6, 2.85 * max(1, len(valid_signals)))),
        dpi=180,
        sharex=True,
    )
    if len(valid_signals) == 1:
        axes = [axes]
    for ax, (role, signal) in zip(axes, valid_signals, strict=False):
        spectra = _state_psd_spectra(
            header,
            signal,
            predictions,
            epoch_length_sec=epoch_length_sec,
            min_hz=min_hz,
            max_hz=max_hz,
            max_windows_per_state=160,
        )
        mean_values = []
        for state in ["DISCO-T", "REM", "NREM", "Wake", "Unknown"]:
            state_spectra = spectra.get(state, [])
            if not state_spectra:
                continue
            color = STATE_COLORS.get(state, "#7f7f7f")
            for freqs, power in _evenly_sample_spectra(state_spectra, max_faint_curves_per_state):
                ax.plot(freqs, power, color=color, alpha=0.20, linewidth=0.45)
            aligned = _align_spectra(state_spectra)
            if aligned is not None:
                freqs, matrix = aligned
                mean = _smooth_power(np.nanmean(matrix, axis=0), window=5)
                if show_sem_band and len(matrix) > 1:
                    sem = _smooth_power(np.nanstd(matrix, axis=0) / np.sqrt(len(matrix)), window=5)
                    ax.fill_between(freqs, mean - sem, mean + sem, color=color, alpha=0.12, linewidth=0)
                ax.plot(freqs, mean, color=color, alpha=0.98, linewidth=2.0, label=state)
                mean_values.extend(mean.tolist())
        if mean_values:
            ymax = float(np.nanpercentile(np.asarray(mean_values, dtype=float), 99.0))
            if np.isfinite(ymax) and ymax > 0:
                ax.set_ylim(0, ymax * 1.18)
        ax.set_title(role, loc="right", fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Power (uV$^2$/Hz)", fontsize=9)
        ax.grid(True, color="#e8edf5", linewidth=0.45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_linewidth(0.75)
        ax.tick_params(labelsize=8.5, width=0.75, length=3)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right", handlelength=1.7)
    axes[-1].set_xlim(min_hz, max_hz)
    axes[-1].set_xlabel("Frequency (Hz)", fontsize=9.5, fontstyle="italic")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.965, bottom=0.105, hspace=0.34)
    _save_static_matplotlib(fig, output_path)


def _state_psd_spectra(
    header: EDFHeader,
    signal: int | str,
    predictions: pd.DataFrame,
    *,
    epoch_length_sec: float,
    min_hz: float,
    max_hz: float,
    max_windows_per_state: int = 4,
) -> dict[str, list[tuple[object, object]]]:
    import numpy as np
    from scipy.signal import welch

    frame = predictions.copy()
    frame["EpochIndex"] = pd.to_numeric(frame.get("EpochIndex"), errors="coerce")
    frame["EpochStartSeconds"] = pd.to_numeric(frame.get("EpochStartSeconds"), errors="coerce")
    frame["PredLabel"] = frame.get("PredLabel", pd.Series(dtype=str)).astype(str)
    frame = frame.dropna(subset=["EpochIndex", "EpochStartSeconds"])
    out: dict[str, list[tuple[object, object]]] = {}
    sample_rate = header.sample_rate(signal)
    for state, group in frame.groupby("PredLabel", dropna=False):
        group = group.sort_values("EpochIndex")
        if group.empty:
            continue
        if len(group) > max_windows_per_state:
            take = np.linspace(0, len(group) - 1, max_windows_per_state, dtype=int)
            group = group.iloc[take]
        spectra = []
        for start in group["EpochStartSeconds"].to_numpy(dtype=float):
            _t, values = read_signal_window(header, signal, float(start), epoch_length_sec)
            if len(values) < 16:
                continue
            values = np.asarray(values, dtype=float)
            values = values - np.nanmedian(values)
            nperseg = min(1024, len(values))
            freqs, power = welch(values, fs=sample_rate, nperseg=nperseg, noverlap=max(0, nperseg // 2))
            mask = (freqs >= min_hz) & (freqs <= max_hz)
            if mask.any():
                spectra.append((freqs[mask], power[mask]))
        if spectra:
            out[str(state)] = spectra
    return out


def _align_spectra(spectra: list[tuple[object, object]]):
    import numpy as np

    if not spectra:
        return None
    base_freqs = np.asarray(spectra[0][0], dtype=float)
    rows = []
    for freqs, power in spectra:
        freqs = np.asarray(freqs, dtype=float)
        power = np.asarray(power, dtype=float)
        if len(freqs) == len(base_freqs) and np.allclose(freqs, base_freqs):
            rows.append(power)
        else:
            rows.append(np.interp(base_freqs, freqs, power))
    return base_freqs, np.vstack(rows)


def _evenly_sample_spectra(
    spectra: list[tuple[object, object]],
    max_count: int,
) -> list[tuple[object, object]]:
    if max_count <= 0:
        return []
    if len(spectra) <= max_count:
        return spectra
    import numpy as np

    take = np.linspace(0, len(spectra) - 1, max_count, dtype=int)
    return [spectra[int(index)] for index in take]


def _smooth_power(values, *, window: int):
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(arr, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _write_static_asset_page(output_path: Path, *, title: str, note: str = "") -> None:
    svg = output_path.with_suffix(".svg")
    png = output_path.with_suffix(".png")
    asset = svg if svg.exists() else png if png.exists() else None
    if asset is None:
        return
    rel = html.escape(asset.name)
    note_html = f"<p class='note'>{html.escape(note)}</p>" if note else ""
    output_path.write_text(
        f"""
<html>
<head>
<style>
body {{ margin: 0; background: #ffffff; color: #17213a; font-family: Arial, Helvetica, sans-serif; }}
main {{ padding: 18px 22px 28px; }}
h1 {{ margin: 0 0 8px; font-size: 20px; font-weight: 700; }}
.note {{ margin: 0 0 14px; color: #59657a; font-size: 13px; }}
img {{ display: block; width: 100%; height: auto; }}
</style>
</head>
<body><main><h1>{html.escape(title)}</h1>{note_html}<img src="{rel}" /></main></body>
</html>
""",
        encoding="utf-8",
    )


def _try_write_static(fig, html_path: Path) -> None:
    for suffix in (".png", ".svg"):
        try:
            fig.write_image(html_path.with_suffix(suffix))
        except Exception:
            continue


def _draw_envelope_axis(ax, envelope: pd.DataFrame, color: str, label: str, ylabel: str) -> None:
    x = pd.to_numeric(envelope["time_seconds"], errors="coerce").to_numpy(dtype=float) / 3600.0
    y_min = pd.to_numeric(envelope["min"], errors="coerce").to_numpy(dtype=float)
    y_max = pd.to_numeric(envelope["max"], errors="coerce").to_numpy(dtype=float)
    y_mean = pd.to_numeric(envelope["mean"], errors="coerce").to_numpy(dtype=float)
    ax.fill_between(x, y_min, y_max, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, y_mean, color=color, linewidth=0.55)
    ax.text(0.005, 0.86, label, transform=ax.transAxes, fontsize=9, fontweight="bold", color=color)
    ax.set_ylabel(ylabel)


def _draw_static_state_strip(ax, predictions: pd.DataFrame, epoch_length_sec: float) -> None:
    import numpy as np
    from matplotlib.colors import ListedColormap

    state_order = ["Wake", "NREM", "REM", "Unknown", "TransitionalOrUnclassified"]
    colors = [STATE_COLORS.get(state, "#b8bec8") for state in state_order]
    values = {state: idx for idx, state in enumerate(state_order)}
    frame = predictions.copy()
    frame["EpochStartSeconds"] = pd.to_numeric(frame["EpochStartSeconds"], errors="coerce")
    frame = frame.dropna(subset=["EpochStartSeconds"]).sort_values("EpochStartSeconds")
    if frame.empty:
        return
    step = max(1, len(frame) // 16000)
    frame = frame.iloc[::step]
    strip = np.array([values.get(str(label), values["Unknown"]) for label in frame["PredLabel"]], dtype=float)[None, :]
    start = float(frame["EpochStartSeconds"].min()) / 3600.0
    end = float((frame["EpochStartSeconds"].max() + epoch_length_sec * step)) / 3600.0
    ax.imshow(strip, aspect="auto", interpolation="nearest", cmap=ListedColormap(colors), extent=[start, end, 0, 1])
    ax.set_ylabel("State", rotation=0, ha="right", va="center", labelpad=18, fontsize=8, fontweight="bold")


def _normalize_zoom_trace(values) -> object:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    arr = arr - np.nanmedian(arr)
    scale = np.nanpercentile(np.abs(arr), 98)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanmax(np.abs(arr)) or 1.0
    return np.clip(arr / scale, -1.0, 1.0) * 0.42


def _load_spectrogram_preview(spectrogram_index: pd.DataFrame, *, max_tiles: int = 48):
    import numpy as np

    powers = []
    times = []
    freqs = None
    paths = spectrogram_index.get("Path", pd.Series(dtype=str)).dropna().head(max_tiles)
    for path in paths:
        data = np.load(path)
        freqs = data["frequency_hz"]
        times.extend(data["time_seconds"].tolist())
        powers.append(data["power"])
    if not powers or freqs is None:
        return None, None, None
    matrix = np.concatenate(powers, axis=1)
    time_values = np.asarray(times, dtype=float)
    if matrix.shape[1] > 288:
        groups = np.array_split(np.arange(matrix.shape[1]), 288)
        matrix = np.column_stack([matrix[:, group].mean(axis=1) for group in groups])
        time_values = np.asarray([time_values[group].mean() for group in groups], dtype=float)
    return matrix, freqs, time_values


def _resolve_zoom_epochs(predictions: pd.DataFrame, zoom_epochs: list[int] | None = None) -> list[tuple[str, int]]:
    frame = predictions.copy()
    frame["EpochIndex"] = pd.to_numeric(frame["EpochIndex"], errors="coerce")
    frame = frame.dropna(subset=["EpochIndex"])
    if zoom_epochs:
        out = []
        labels = frame.set_index(frame["EpochIndex"].astype(int))["PredLabel"].astype(str).to_dict()
        max_epoch = int(frame["EpochIndex"].max()) if not frame.empty else 1
        for epoch in zoom_epochs[:3]:
            resolved = min(max(int(epoch), 1), max_epoch)
            out.append((labels.get(resolved, "Epoch"), resolved))
        return out
    reps = select_representative_epochs(predictions)
    return [(state, reps[state]) for state in ("Wake", "NREM", "REM") if state in reps] or [(state, epoch) for state, epoch in reps.items()]


def _downsample_signal_trace(header: EDFHeader, signal: int | str, *, max_points: int = 24000) -> tuple[object, object]:
    import numpy as np

    sample_rate = header.sample_rate(signal)
    total_samples = max(1, int(round(header.duration_seconds * sample_rate)))
    step = max(1, int(np.ceil(total_samples / max_points)))
    xs = []
    ys = []
    offset = 0
    for digital in iter_signal_digital_chunks(header, signal, max_samples=max(step * 4096, step)):
        values = digital_to_microvolts(header, signal, digital)
        sampled = values[::step]
        indices = np.arange(offset, offset + len(values), step)[: len(sampled)]
        xs.append(indices / sample_rate / 3600.0)
        ys.append(sampled)
        offset += len(values)
    if not xs:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.concatenate(xs), np.concatenate(ys)


def _draw_full_trace_axis(ax, x, y, color: str, label: str, ylabel: str) -> None:
    ax.plot(x, y, color=color, linewidth=0.35)
    if label:
        ax.text(0.005, 0.84, label, transform=ax.transAxes, fontsize=9, fontweight="bold", color=color)
    ax.set_ylabel(ylabel)


def _filter_psd_for_display(psd: pd.DataFrame, *, min_hz: float, max_hz: float) -> pd.DataFrame:
    if psd.empty:
        return psd
    frame = psd.copy()
    frame["FrequencyHz"] = pd.to_numeric(frame["FrequencyHz"], errors="coerce")
    frame["Power"] = pd.to_numeric(frame["Power"], errors="coerce")
    frame = frame.dropna(subset=["FrequencyHz", "Power"])
    return frame[(frame["FrequencyHz"] >= min_hz) & (frame["FrequencyHz"] <= max_hz)]


def _plot_psd_signal(ax, psd: pd.DataFrame, signal: str, color: str) -> None:
    if not signal:
        return
    subset = psd[psd["Signal"].astype(str) == signal].copy()
    if subset.empty:
        return
    group_cols = ["HourBin"] if "HourBin" in subset.columns else []
    if group_cols:
        for _, group in subset.groupby(group_cols, dropna=False):
            ax.plot(group["FrequencyHz"], group["Power"], color=color, alpha=0.28, linewidth=0.7)
    else:
        ax.plot(subset["FrequencyHz"], subset["Power"], color=color, alpha=0.8, linewidth=0.9)


def _save_static_matplotlib(fig, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    for suffix in (".png", ".svg"):
        fig.savefig(output_path.with_suffix(suffix), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _apply_publication_matplotlib_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "axes.titleweight": "bold",
            "axes.labelweight": "normal",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "text.color": "#111111",
            "axes.labelcolor": "#111111",
            "xtick.color": "#111111",
            "ytick.color": "#111111",
            "axes.edgecolor": "#111111",
            "svg.fonttype": "none",
        }
    )
