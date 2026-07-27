from __future__ import annotations

import html
import json
from pathlib import Path
import traceback

import pandas as pd

from .config import AnalysisConfig, RecordingSpec
from .edf_stream import read_edf_header
from .labels import discover_label_files, inventory_label_files, load_label_frame
from .metrics import sanitize_name
from .paper_figures import write_paper_trace_figure
from .pipeline import AnalysisRunner
from .signal_analysis import write_epoch_inspector_plot


def parse_epoch_text(text: str, *, max_epoch: int) -> tuple[int, str]:
    cleaned = str(text or "").strip().replace(",", "")
    try:
        value = int(float(cleaned))
    except ValueError:
        return 1, f"Invalid epoch; loaded epoch 1. Valid range: 1-{max_epoch}"
    if value < 1:
        return 1, f"Epoch {value} is below range; loaded epoch 1. Valid range: 1-{max_epoch}"
    if value > max_epoch:
        return max_epoch, f"Epoch {value} is above range; loaded epoch {max_epoch}. Valid range: 1-{max_epoch}"
    return value, f"Loaded epoch {value}. Valid range: 1-{max_epoch}"


def run_gui(config_path: str | Path = "analysis_config.json") -> int:
    try:
        from PySide6.QtCore import QFileSystemWatcher, QObject, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot
        from PySide6.QtWidgets import (
            QApplication,
            QFileDialog,
            QHBoxLayout,
            QLabel,
            QListWidget,
            QMainWindow,
            QPushButton,
            QLineEdit,
            QPlainTextEdit,
            QSplitter,
            QTabWidget,
            QToolBar,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        print(f"PySide6 is required for the GUI: {exc}")
        return 2

    try:
        from PySide6.QtCore import QUrl
        from PySide6.QtWebEngineWidgets import QWebEngineView
    except ImportError:
        QUrl = None
        QWebEngineView = None

    class WorkerSignals(QObject):
        finished = Signal(str)
        failed = Signal(str)

    class PipelineWorker(QRunnable):
        def __init__(self, command: str, cfg_path: Path):
            super().__init__()
            self.command = command
            self.cfg_path = cfg_path
            self.signals = WorkerSignals()

        @Slot()
        def run(self) -> None:
            try:
                runner = AnalysisRunner(AnalysisConfig.from_json(self.cfg_path))
                if self.command == "inventory":
                    rows = runner.run_inventory()
                    self.signals.finished.emit(f"Inventoried {len(rows)} label rows.")
                else:
                    outputs = runner.run_analysis()
                    self.signals.finished.emit("Analysis complete:\n" + "\n".join(f"{k}: {v}" for k, v in outputs.items()))
            except Exception:  # noqa: BLE001
                self.signals.failed.emit(traceback.format_exc())

    class HtmlPane(QWidget):
        def __init__(self, label: str):
            super().__init__()
            self.label = label
            layout = QVBoxLayout(self)
            if QWebEngineView is not None:
                self.viewer = QWebEngineView()
                layout.addWidget(self.viewer)
            else:
                self.viewer = QPlainTextEdit()
                self.viewer.setReadOnly(True)
                layout.addWidget(self.viewer)

        def load_file(self, path: Path) -> None:
            if QWebEngineView is not None and hasattr(self.viewer, "load") and QUrl is not None:
                self.viewer.load(QUrl.fromLocalFile(str(path.resolve())))
            elif path.suffix.lower() in {".png", ".svg", ".jpg", ".jpeg"}:
                self.viewer.setPlainText(str(path.resolve()))
            else:
                self.viewer.setPlainText(path.read_text(encoding="utf-8", errors="replace"))

        def load_html(self, title: str, body: str) -> None:
            text = f"<html><body><h1>{html.escape(title)}</h1>{body}</body></html>"
            if QWebEngineView is not None and hasattr(self.viewer, "setHtml"):
                self.viewer.setHtml(text)
            else:
                self.viewer.setPlainText(text)

    class MainWindow(QMainWindow):
        def __init__(self, cfg_path: Path):
            super().__init__()
            self.setWindowTitle("MARS Analysis")
            self.resize(1480, 900)
            self.cfg_path = cfg_path
            self.cfg: AnalysisConfig | None = None
            self.current_epoch_artifact: Path | None = None
            self.current_epoch_recording_id = ""
            self.current_recording_id = ""
            self.label_index_cache: dict[str, dict[str, pd.DataFrame]] = {}
            self.thread_pool = QThreadPool.globalInstance()
            self.watcher = QFileSystemWatcher()
            self.watcher.fileChanged.connect(self.on_config_changed)
            self.refresh_timer = QTimer()
            self.refresh_timer.setInterval(10_000)
            self.refresh_timer.timeout.connect(self.refresh_plots)

            self.recording_list = QListWidget()
            self.recording_list.currentTextChanged.connect(lambda _text: self.refresh_plots())
            self.meta = QLabel()
            self.meta.setWordWrap(True)
            self.max_epoch = 10_000_000
            self.epoch_input = QLineEdit("1")
            self.epoch_input.setPlaceholderText("Type epoch number")
            self.epoch_input.returnPressed.connect(self.load_epoch)
            epoch_button = QPushButton("Load Epoch")
            epoch_button.clicked.connect(self.load_epoch)
            self.epoch_status = QLabel("Epoch range: unknown")
            self.epoch_status.setWordWrap(True)
            self.zoom_epochs_input = QLineEdit("")
            self.zoom_epochs_input.setPlaceholderText("Zoom epochs, e.g. 11738,34158,27435")
            self.zoom_epochs_input.returnPressed.connect(self.update_trace_zooms)
            zoom_button = QPushButton("Update Trace Zooms")
            zoom_button.clicked.connect(self.update_trace_zooms)

            sidebar = QWidget()
            sidebar_layout = QVBoxLayout(sidebar)
            sidebar_layout.addWidget(QLabel("Recordings"))
            sidebar_layout.addWidget(self.recording_list, 1)
            sidebar_layout.addWidget(QLabel("Selected Metadata"))
            sidebar_layout.addWidget(self.meta)
            sidebar_layout.addWidget(QLabel("Epoch"))
            sidebar_layout.addWidget(self.epoch_input)
            sidebar_layout.addWidget(epoch_button)
            sidebar_layout.addWidget(self.epoch_status)
            sidebar_layout.addWidget(QLabel("Trace Zoom Epochs"))
            sidebar_layout.addWidget(self.zoom_epochs_input)
            sidebar_layout.addWidget(zoom_button)

            self.log = QPlainTextEdit()
            self.log.setReadOnly(True)
            self.config_text = QPlainTextEdit()
            self.config_text.setReadOnly(True)
            self.dataset_status = HtmlPane("Dataset Status")
            self.overview = HtmlPane("Overview")
            self.classification = HtmlPane("Classification")
            self.trace_qc = HtmlPane("Trace QC")
            self.spectrogram = HtmlPane("Spectrogram")
            self.power = HtmlPane("Power Spectra")
            self.epoch = HtmlPane("Epoch Inspector")
            self.group = HtmlPane("Group Analysis")

            epoch_tab = QWidget()
            self.epoch_tab = epoch_tab
            epoch_tab_layout = QVBoxLayout(epoch_tab)
            epoch_controls = QWidget()
            epoch_controls_layout = QHBoxLayout(epoch_controls)
            self.epoch_tab_input = QLineEdit("1")
            self.epoch_tab_input.setPlaceholderText("Type epoch number")
            self.epoch_tab_input.returnPressed.connect(self.load_epoch_from_tab)
            epoch_tab_button = QPushButton("Load Epoch")
            epoch_tab_button.clicked.connect(self.load_epoch_from_tab)
            self.epoch_tab_status = QLabel("Epoch range: unknown")
            self.epoch_tab_status.setWordWrap(True)
            epoch_controls_layout.addWidget(QLabel("Epoch"))
            epoch_controls_layout.addWidget(self.epoch_tab_input)
            epoch_controls_layout.addWidget(epoch_tab_button)
            epoch_controls_layout.addWidget(self.epoch_tab_status)
            epoch_tab_layout.addWidget(epoch_controls)
            epoch_tab_layout.addWidget(self.epoch)

            dataset_tab = QWidget()
            dataset_layout = QVBoxLayout(dataset_tab)
            dataset_layout.addWidget(self.dataset_status, 3)
            dataset_layout.addWidget(QLabel("Run Log"))
            dataset_layout.addWidget(self.log, 1)

            self.tabs = QTabWidget()
            self.tabs.addTab(dataset_tab, "Dataset Status")
            self.tabs.addTab(self.overview, "Recording Overview")
            self.tabs.addTab(self.classification, "Classification")
            self.tabs.addTab(self.trace_qc, "Trace QC")
            self.tabs.addTab(self.spectrogram, "Spectrogram")
            self.tabs.addTab(self.power, "Power Spectra")
            self.tabs.addTab(epoch_tab, "Epoch Inspector")
            self.tabs.addTab(self.group, "Group Analysis")

            splitter = QSplitter()
            splitter.addWidget(sidebar)
            splitter.addWidget(self.tabs)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            self.setCentralWidget(splitter)

            toolbar = QToolBar("Actions")
            self.addToolBar(toolbar)
            for text, command in [
                ("Open Config", "open"),
                ("Inventory", "inventory"),
                ("Analyze", "analyze"),
                ("Refresh", "refresh"),
            ]:
                button = QPushButton(text)
                button.clicked.connect(lambda _checked=False, cmd=command: self.handle_command(cmd))
                toolbar.addWidget(button)

            self.load_config()
            self.refresh_timer.start()

        def handle_command(self, command: str) -> None:
            if command == "open":
                path, _ = QFileDialog.getOpenFileName(self, "Open analysis config", str(self.cfg_path), "JSON (*.json)")
                if path:
                    self.cfg_path = Path(path)
                    self.load_config()
                return
            if command == "refresh":
                self.load_config()
                return
            self.log.appendPlainText(f"Running {command}...")
            worker = PipelineWorker(command, self.cfg_path)
            worker.signals.finished.connect(self.on_finished)
            worker.signals.failed.connect(self.on_failed)
            self.thread_pool.start(worker)

        def on_finished(self, message: str) -> None:
            self.log.appendPlainText(message)
            self.load_config()

        def on_failed(self, message: str) -> None:
            self.log.appendPlainText(message)

        def on_config_changed(self, path: str) -> None:
            QTimer.singleShot(250, self.load_config)

        def load_config(self) -> None:
            try:
                self.cfg = AnalysisConfig.from_json(self.cfg_path)
                self.config_text.setPlainText(self.cfg_path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                self.log.appendPlainText(f"Could not load config: {exc}")
                return
            watched = set(self.watcher.files())
            cfg_str = str(self.cfg_path)
            if cfg_str not in watched and self.cfg_path.exists():
                self.watcher.addPath(cfg_str)
            previous = self.recording_list.currentItem().text() if self.recording_list.currentItem() else ""
            self.recording_list.blockSignals(True)
            self.recording_list.clear()
            for spec in self.cfg.recordings:
                self.recording_list.addItem(spec.recording_id)
            matches = self.recording_list.findItems(previous, Qt.MatchExactly)
            if matches:
                self.recording_list.setCurrentItem(matches[0])
            elif self.recording_list.count():
                self.recording_list.setCurrentRow(0)
            self.recording_list.blockSignals(False)
            self.refresh_plots()

        def refresh_plots(self) -> None:
            if self.cfg is None:
                return
            spec = self.selected_recording()
            output_dir = self.cfg.dataset_output_dir
            plot_dir = output_dir / "plots"
            self.dataset_status.load_html("Dataset Status", _dataset_status_html(self.cfg))
            if spec is None:
                self.meta.setText("No recording selected.")
                self._load_or_note(self.overview, output_dir / "qc_summary.csv")
                return
            rec_dir = output_dir / spec.recording_id
            if self.current_recording_id != spec.recording_id:
                self.current_recording_id = spec.recording_id
                self.current_epoch_artifact = None
                self.current_epoch_recording_id = ""
            self.meta.setText(_metadata_text(spec))
            self._update_epoch_range(spec)
            self.overview.load_html("Overview", _recording_overview_html(self.cfg, spec) + _channel_verification_html(self.cfg, spec))
            self._load_first_existing(
                self.classification,
                [
                    rec_dir / "classification_dashboard.html",
                    rec_dir / "scored" / "classification_dashboard.html",
                ],
                fallback_html=lambda: _classification_html(self.cfg, spec),
            )
            self._load_first_existing(
                self.trace_qc,
                [
                    rec_dir / "paper_figures" / "paper_trace_overview.svg",
                    rec_dir / "paper_figures" / "paper_trace_overview.png",
                    rec_dir / "paper_figures" / "paper_trace_overview.html",
                    *sorted((rec_dir / "plots").glob("*trace_envelope.html")),
                    output_dir / "qc_summary.csv",
                ],
            )
            self._load_first_existing(
                self.spectrogram,
                [
                    rec_dir / "paper_figures" / "paper_spectrogram.svg",
                    rec_dir / "paper_figures" / "paper_spectrogram.png",
                    rec_dir / "paper_figures" / "paper_spectrogram.html",
                    *sorted((rec_dir / "spectrogram_tiles").glob("*_spectrogram.html")),
                ],
            )
            self._load_first_existing(
                self.power,
                [
                    rec_dir / "paper_figures" / "paper_state_power_spectra.svg",
                    rec_dir / "paper_figures" / "paper_state_power_spectra.png",
                    rec_dir / "paper_figures" / "paper_state_power_spectra.html",
                    rec_dir / "paper_figures" / "paper_power_spectra.html",
                    *sorted((rec_dir / "plots").glob("*_hourly_power_spectra.html")),
                    output_dir / "power_spectra.html",
                ],
            )
            self._load_first_existing(
                self.group,
                [
                    output_dir / "plots" / "group_hourly_state_composition.html",
                    output_dir / "plots" / "group_animal_state_summary.html",
                    output_dir / "hourly_bins.csv",
                ],
            )
            if self.current_epoch_recording_id == spec.recording_id and self.current_epoch_artifact and self.current_epoch_artifact.exists():
                self.epoch.load_file(self.current_epoch_artifact)
            else:
                self._load_or_note(self.epoch, rec_dir / "epoch_inspector.html")

        def load_epoch_from_tab(self) -> None:
            self.epoch_input.setText(self.epoch_tab_input.text())
            self.load_epoch()

        def load_epoch(self) -> None:
            if self.cfg is None:
                return
            spec = self.selected_recording()
            if spec is None or not spec.edf_path:
                self.epoch.load_html("Epoch Inspector", "<p>No EDF recording selected.</p>")
                return
            try:
                header = read_edf_header(spec.edf_path)
                eeg = _first_available_signal(header, spec.channel_map.eeg, self.cfg.visualizer.eeg_channel, 0)
                emg = _first_available_signal(header, spec.channel_map.emg, self.cfg.visualizer.emg_channel, min(1, header.signal_count - 1))
                epoch_index, message = parse_epoch_text(self.epoch_input.text(), max_epoch=self.max_epoch)
                self.epoch_input.setText(str(epoch_index))
                self.epoch_tab_input.setText(str(epoch_index))
                loading_message = f"Loading epoch {epoch_index}..."
                self.epoch_status.setText(loading_message)
                self.epoch_tab_status.setText(loading_message)
                self.epoch.load_html("Epoch Inspector", f"<p>{html.escape(loading_message)}</p>")
                labels = self._labels_for_epoch_cached(spec.recording_id, epoch_index)
                epoch_length = _scored_epoch_length(self.cfg, spec) or spec.epoch_length_sec
                out = self.cfg.dataset_output_dir / spec.recording_id / "plots" / f"epoch_inspector_epoch_{epoch_index}.html"
                write_epoch_inspector_plot(
                    header,
                    eeg,
                    emg,
                    out,
                    epoch_index=epoch_index,
                    epoch_length_sec=epoch_length,
                    labels=labels,
                )
                self.current_epoch_artifact = out
                self.current_epoch_recording_id = spec.recording_id
                self.epoch.load_file(out)
                self.epoch_status.setText(message)
                self.epoch_tab_status.setText(message)
                self.tabs.setCurrentWidget(self.epoch_tab)
            except Exception as exc:  # noqa: BLE001
                self.epoch.load_html("Epoch Inspector Error", f"<pre>{html.escape(traceback.format_exc())}</pre>")
                self.log.appendPlainText(f"Epoch inspector failed: {exc}")

        def selected_recording(self) -> RecordingSpec | None:
            if self.cfg is None or self.recording_list.currentItem() is None:
                return None
            rec_id = self.recording_list.currentItem().text()
            return next((spec for spec in self.cfg.recordings if spec.recording_id == rec_id), None)

        def _update_epoch_range(self, spec: RecordingSpec) -> None:
            if self.cfg is None or not spec.edf_path:
                return
            try:
                header = read_edf_header(spec.edf_path)
                epoch_length = _scored_epoch_length(self.cfg, spec) or spec.epoch_length_sec
                max_epoch = max(1, int(header.duration_seconds // max(float(epoch_length), 1e-9)))
                self.max_epoch = max_epoch
                current, _message = parse_epoch_text(self.epoch_input.text(), max_epoch=max_epoch)
                self.epoch_input.setText(str(current))
                self.epoch_status.setText(f"Epoch range: 1-{max_epoch}")
                self.epoch_tab_input.setText(str(current))
                self.epoch_tab_status.setText(f"Epoch range: 1-{max_epoch}")
            except Exception:
                self.max_epoch = 10_000_000
                self.epoch_status.setText("Epoch range unavailable")
                self.epoch_tab_status.setText("Epoch range unavailable")

        def update_trace_zooms(self) -> None:
            if self.cfg is None:
                return
            spec = self.selected_recording()
            if spec is None or not spec.edf_path:
                return
            try:
                epochs = _parse_epoch_list(self.zoom_epochs_input.text(), max_epoch=self.max_epoch)
                header = read_edf_header(spec.edf_path)
                eeg = _first_available_signal(header, spec.channel_map.eeg, self.cfg.visualizer.eeg_channel, 0)
                emg = _first_available_signal(header, spec.channel_map.emg, self.cfg.visualizer.emg_channel, min(1, header.signal_count - 1))
                rec_dir = self.cfg.dataset_output_dir / spec.recording_id
                pred_path = rec_dir / "scored" / "predictions_standardized.csv"
                eeg_env = rec_dir / "trace_envelopes" / f"{sanitize_name(str(eeg))}_envelope.parquet"
                emg_env = rec_dir / "trace_envelopes" / f"{sanitize_name(str(emg))}_envelope.parquet"
                if not pred_path.exists() or not eeg_env.exists() or not emg_env.exists():
                    self.trace_qc.load_html("Trace Zooms", "<p>Run analysis before updating trace zooms.</p>")
                    return
                predictions = pd.read_csv(pred_path)
                epoch_lengths = pd.to_numeric(predictions.get("EpochLengthSeconds", pd.Series(dtype=float)), errors="coerce").dropna()
                epoch_length = float(epoch_lengths.mode().iloc[0]) if not epoch_lengths.empty else spec.epoch_length_sec
                out = rec_dir / "paper_figures" / "paper_trace_custom.html"
                write_paper_trace_figure(
                    recording_id=spec.recording_id,
                    header=header,
                    eeg_signal=eeg,
                    emg_signal=emg,
                    eeg_envelope=pd.read_parquet(eeg_env),
                    emg_envelope=pd.read_parquet(emg_env),
                    predictions=predictions,
                    output_path=out,
                    epoch_length_sec=epoch_length,
                    zoom_epochs=epochs or None,
                )
                preferred = out.with_suffix(".svg")
                self._load_or_note(self.trace_qc, preferred if preferred.exists() else out)
                self.tabs.setCurrentWidget(self.trace_qc)
                self.log.appendPlainText(f"Updated trace zooms for {spec.recording_id}: {epochs or 'representative epochs'}")
            except Exception as exc:  # noqa: BLE001
                self.trace_qc.load_html("Trace Zoom Error", f"<pre>{html.escape(traceback.format_exc())}</pre>")
                self.log.appendPlainText(f"Trace zoom update failed: {exc}")

        def _labels_for_epoch_cached(self, recording_id: str, epoch_index: int) -> dict[str, str]:
            if self.cfg is None:
                return {}
            if recording_id not in self.label_index_cache:
                cache: dict[str, pd.DataFrame] = {}
                predictions = self.cfg.dataset_output_dir / recording_id / "scored" / "predictions_standardized.csv"
                if predictions.exists():
                    try:
                        frame = pd.read_csv(predictions, usecols=["EpochIndex", "PredLabel", "Model"])
                        frame["EpochIndex"] = pd.to_numeric(frame["EpochIndex"], errors="coerce").astype("Int64")
                        model = str(frame["Model"].dropna().astype(str).iloc[0]) if "Model" in frame.columns and not frame.empty else ""
                        cache[f"offline_accusleepy:{model}"] = frame.set_index("EpochIndex", drop=False)
                    except Exception:
                        pass
                self.label_index_cache[recording_id] = cache
            labels: dict[str, str] = {}
            for key, frame in self.label_index_cache.get(recording_id, {}).items():
                try:
                    if epoch_index in frame.index:
                        value = frame.loc[epoch_index]
                        if isinstance(value, pd.DataFrame):
                            value = value.iloc[0]
                        labels[key] = str(value["PredLabel"])
                except Exception:
                    continue
            return labels

        def _load_first_existing(self, pane: HtmlPane, paths: list[Path], fallback_html=None) -> None:
            for path in paths:
                if path.exists():
                    self._load_or_note(pane, path)
                    return
            if fallback_html is not None:
                pane.load_html(pane.label, fallback_html())
                return
            self._load_or_note(pane, paths[0])

        def _load_first_matching(self, pane: HtmlPane, folder: Path, pattern: str, fallback: Path | None) -> None:
            matches = sorted(folder.glob(pattern)) if folder.exists() else []
            if matches:
                pane.load_file(matches[0])
            elif fallback is not None:
                self._load_or_note(pane, fallback)
            else:
                pane.load_html(pane.label, f"<p>No artifact matching {html.escape(pattern)} in {html.escape(str(folder))}.</p>")

        def _load_or_note(self, pane: HtmlPane, path: Path) -> None:
            if path.exists() and path.suffix.lower() == ".html":
                pane.load_file(path)
            elif path.exists() and path.suffix.lower() in {".png", ".svg", ".jpg", ".jpeg"}:
                pane.load_file(path)
            elif path.exists() and path.suffix.lower() == ".json":
                pane.load_html(path.name, f"<pre>{html.escape(path.read_text(encoding='utf-8', errors='replace'))}</pre>")
            elif path.exists() and path.suffix.lower() == ".csv":
                pane.load_html(path.name, _csv_preview(path))
            elif path.exists():
                pane.load_html(path.name, f"<pre>{html.escape(path.read_text(encoding='utf-8', errors='replace')[:20000])}</pre>")
            else:
                pane.load_html(pane.label, f"<p>Run analysis to create {html.escape(path.name)}.</p><p>{html.escape(str(path))}</p>")

    def _csv_preview(path: Path) -> str:
        try:
            return _styled_table(pd.read_csv(path).head(300))
        except Exception as exc:  # noqa: BLE001
            return f"<pre>{html.escape(str(exc))}</pre>"

    def _metadata_text(spec: RecordingSpec) -> str:
        lines = [
            f"Animal: {spec.animal_id or ''}",
            f"Condition: {spec.condition or ''}",
            f"Dose: {spec.dose or ''}",
            f"Session: {spec.session or ''}",
            f"EEG: {spec.channel_map.eeg or ''}",
            f"EMG: {spec.channel_map.emg or ''}",
        ]
        return "\n".join(lines)

    def _parse_epoch_list(text: str, *, max_epoch: int) -> list[int]:
        epochs: list[int] = []
        for part in str(text or "").replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                value = int(float(part.replace(",", "")))
            except ValueError:
                continue
            epochs.append(min(max(value, 1), max_epoch))
        return epochs[:3]

    def _channel_verification_html(cfg: AnalysisConfig, spec: RecordingSpec) -> str:
        try:
            if not spec.edf_path:
                return ""
            header = read_edf_header(spec.edf_path)
            eeg = _first_available_signal(header, spec.channel_map.eeg, cfg.visualizer.eeg_channel, 0)
            emg = _first_available_signal(header, spec.channel_map.emg, cfg.visualizer.emg_channel, min(1, header.signal_count - 1))
            rows = []
            for role, signal in [("EEG", eeg), ("EMG", emg)]:
                idx = header.signal_index(signal)
                env_path = cfg.dataset_output_dir / spec.recording_id / "trace_envelopes" / f"{sanitize_name(str(signal))}_envelope.parquet"
                min_uv = max_uv = rms_uv = ""
                if env_path.exists():
                    env = pd.read_parquet(env_path, columns=["min", "max", "rms"])
                    min_uv = float(pd.to_numeric(env["min"], errors="coerce").min())
                    max_uv = float(pd.to_numeric(env["max"], errors="coerce").max())
                    rms_uv = float(pd.to_numeric(env["rms"], errors="coerce").mean())
                rows.append(
                    {
                        "Role": role,
                        "EDFLabel": header.labels[idx],
                        "PhysicalUnit": header.physical_dimensions[idx],
                        "SampleRateHz": header.sample_rate(idx),
                        "MinUV": min_uv,
                        "MaxUV": max_uv,
                        "MeanRMSUV": rms_uv,
                    }
                )
            return "<h2>Channel Verification</h2>" + _styled_table(pd.DataFrame(rows))
        except Exception as exc:  # noqa: BLE001
            return f"<h2>Channel Verification</h2><p class='note'>Unavailable: {html.escape(str(exc))}</p>"

    def _recording_overview_html(cfg: AnalysisConfig, spec: RecordingSpec) -> str:
        rows = {
            "RecordingID": spec.recording_id,
            "AnimalID": spec.animal_id,
            "Condition": spec.condition,
            "Dose": spec.dose,
            "Session": spec.session,
            "EDF": spec.edf_path,
            "Model": cfg.default_model_alias,
            "Output": str(cfg.dataset_output_dir / spec.recording_id),
        }
        return _styled_page("Overview", [_styled_table(pd.DataFrame([rows]))])

    def _dataset_status_html(cfg: AnalysisConfig) -> str:
        output_dir = cfg.dataset_output_dir
        status_path = output_dir / "classification_status.csv"
        cards = [("Dataset", cfg.dataset_id), ("Model", cfg.default_model_alias), ("Configured recordings", str(len(cfg.recordings)))]
        if status_path.exists():
            status = pd.read_csv(status_path)
            if not status.empty and "Status" in status:
                counts = status["Status"].astype(str).value_counts().rename_axis("Status").reset_index(name="Recordings")
                cards.append(("Scored recordings", f"{int((status['Status'].astype(str) == 'ok').sum())}/{len(cfg.recordings)}"))
                return _styled_page("Dataset Status", [_cards(cards), "<h2>Classification Status</h2>", _styled_table(counts)])
        return _styled_page("Dataset Status", [_cards(cards), "<p>Run analysis to create model predictions and review artifacts.</p>"])

        # Retained below only for private-workspace compatibility; public UI returns above.
        pieces: list[str] = []
        summary_path = output_dir / "mars_vs_reference_summary.csv"
        alignment_path = output_dir / "reference_alignment_summary.csv"
        status_path = output_dir / "classification_status.csv"
        sanity_path = output_dir / "prediction_sanity.csv"
        cards: list[tuple[str, str]] = [
            ("Dataset", cfg.dataset_id),
            ("Model", cfg.default_model_alias),
            ("Configured Recordings", f"{len(cfg.recordings):,}"),
        ]

        if status_path.exists():
            try:
                status = pd.read_csv(status_path)
                status_counts = status["Status"].astype(str).value_counts().rename_axis("Status").reset_index(name="Recordings")
                ok_count = int((status["Status"].astype(str) == "ok").sum())
                cards.append(("Scored Recordings", f"{ok_count:,}/{len(cfg.recordings):,}"))
                pieces.append("<h2>Classification Status</h2>" + _styled_table(status_counts))
            except Exception as exc:  # noqa: BLE001
                pieces.append(f"<p class='note'>Could not read classification status: {html.escape(str(exc))}</p>")

        if summary_path.exists():
            try:
                summary = pd.read_csv(summary_path)
                ok = summary[summary.get("status", pd.Series(dtype=str)).astype(str) == "ok"] if not summary.empty else pd.DataFrame()
                mars_epochs = int(pd.to_numeric(ok.get("mars_epoch_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
                aligned = int(pd.to_numeric(ok.get("aligned_epoch_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
                strict = int(pd.to_numeric(ok.get("strict_valid_epoch_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
                epoch_lengths = pd.to_numeric(ok.get("mars_epoch_length_sec", pd.Series(dtype=float)), errors="coerce").dropna()
                if epoch_lengths.empty and not ok.empty:
                    epoch_lengths = pd.Series([2.5])
                epoch_length = float(epoch_lengths.mode().iloc[0]) if not epoch_lengths.empty else 2.5
                two_second_equiv = int(round(mars_epochs * epoch_length / 2.0)) if mars_epochs else 0
                cards.extend(
                    [
                        ("MARS Epochs", f"{mars_epochs:,}"),
                        ("Aligned reference Epochs", f"{aligned:,}"),
                        ("Strict-Valid Epochs", f"{strict:,}"),
                        ("Epoch Length", f"{epoch_length:g} s"),
                    ]
                )
                metrics = pd.DataFrame(
                    [
                        {
                            "View": "All aligned",
                            "Accuracy": _weighted(ok, "accuracy", "aligned_epoch_count"),
                            "BalancedAccuracy": _weighted(ok, "balanced_accuracy", "aligned_epoch_count"),
                            "MacroF1": _weighted(ok, "macro_f1", "aligned_epoch_count"),
                            "WeightedF1": _weighted(ok, "weighted_f1", "aligned_epoch_count"),
                            "Kappa": _weighted(ok, "kappa", "aligned_epoch_count"),
                        },
                        {
                            "View": "Strict valid",
                            "Accuracy": _weighted(ok, "strict_accuracy", "strict_valid_epoch_count"),
                            "BalancedAccuracy": _weighted(ok, "strict_balanced_accuracy", "strict_valid_epoch_count"),
                            "MacroF1": _weighted(ok, "strict_macro_f1", "strict_valid_epoch_count"),
                            "WeightedF1": _weighted(ok, "strict_weighted_f1", "strict_valid_epoch_count"),
                            "Kappa": _weighted(ok, "strict_kappa", "strict_valid_epoch_count"),
                        },
                    ]
                )
                pieces.append(
                    "<p class='note'>"
                    f"E2.5W9 writes one prediction every {epoch_length:g} seconds. "
                    "The raw reference source export is on a 2.0-second grid, but the active benchmark labels are reference-derived "
                    "and remapped to the E2.5W9 2.5-second grid by max-overlap before comparison. "
                    f"The {mars_epochs:,} MARS rows are therefore the correct 2.5-second denominator; "
                    f"that is roughly {two_second_equiv:,} rows if expressed on a 2.0-second time grid."
                    "</p>"
                )
                pieces.append("<h2>MARS vs reference Metrics</h2>" + _styled_table(metrics))
            except Exception as exc:  # noqa: BLE001
                pieces.append(f"<p class='note'>Could not read MARS-vs-reference summary: {html.escape(str(exc))}</p>")

        if alignment_path.exists():
            try:
                alignment = pd.read_csv(alignment_path)
                align_counts = alignment["Status"].astype(str).value_counts().rename_axis("AlignmentStatus").reset_index(name="Recordings")
                pieces.append("<h2>reference Alignment</h2>" + _styled_table(align_counts))
            except Exception as exc:  # noqa: BLE001
                pieces.append(f"<p class='note'>Could not read alignment summary: {html.escape(str(exc))}</p>")

        if sanity_path.exists():
            try:
                sanity = pd.read_csv(sanity_path)
                sanity_counts = sanity["Status"].astype(str).value_counts().rename_axis("PredictionSanity").reset_index(name="Recordings")
                pieces.append("<h2>Prediction Sanity</h2>" + _styled_table(sanity_counts))
            except Exception as exc:  # noqa: BLE001
                pieces.append(f"<p class='note'>Could not read prediction sanity: {html.escape(str(exc))}</p>")

        pieces.insert(0, _cards(cards))
        pieces.append(
            "<h2>Artifacts</h2>"
            + _styled_table(
                pd.DataFrame(
                    [
                        {"Artifact": "MARS vs reference dashboard", "Path": str(output_dir / "plots" / "mars_vs_reference_dashboard.html")},
                        {"Artifact": "State percentages", "Path": str(output_dir / "state_percentages.csv")},
                        {"Artifact": "Hourly bins", "Path": str(output_dir / "hourly_bins.csv")},
                        {"Artifact": "Classification status", "Path": str(status_path)},
                    ]
                )
            )
        )
        return _styled_page("Dataset Status", pieces)

    def _group_analysis_html(cfg: AnalysisConfig, output_dir: Path) -> str:
        hourly_path = output_dir / "hourly_bins.csv"
        state_path = output_dir / "state_percentages.csv"
        sanity_path = output_dir / "prediction_sanity.csv"
        if not hourly_path.exists():
            return f"<p>Run analysis to create {html.escape(str(hourly_path))}.</p>"
        hourly = pd.read_csv(hourly_path)
        if hourly.empty:
            return "<p>No hourly state bins are available yet.</p>"
        pieces = []
        if sanity_path.exists():
            sanity = pd.read_csv(sanity_path)
            failed = sanity[sanity["Status"].astype(str).str.lower() != "pass"] if not sanity.empty else sanity
            if not failed.empty:
                pieces.append(
                    "<h2>Prediction Sanity Failed</h2>"
                    "<p>Group analysis is not biologically interpretable until these scoring issues are resolved.</p>"
                    + _styled_table(failed.head(200))
                )
        meta = pd.DataFrame(
            [
                {
                    "RecordingID": spec.recording_id,
                    "AnimalID": spec.animal_id,
                    "Condition": spec.condition,
                    "Dose": spec.dose,
                    "Session": spec.session,
                }
                for spec in cfg.recordings
            ]
        )
        merged = hourly.merge(meta, on="RecordingID", how="left")
        keys = ["Condition", "Dose", "Session", "HourBin", "PredLabel"]
        grouped = (
            merged.groupby(keys, dropna=False)["EpochCount"]
            .sum()
            .rename("EpochCount")
            .reset_index()
        )
        totals = grouped.groupby(["Condition", "Dose", "Session", "HourBin"])["EpochCount"].transform("sum")
        grouped["Percent"] = grouped["EpochCount"] / totals
        animal_summary = (
            merged.groupby(["AnimalID", "Condition", "Dose", "PredLabel"], dropna=False)["EpochCount"]
            .sum()
            .rename("EpochCount")
            .reset_index()
        )
        animal_totals = animal_summary.groupby(["AnimalID", "Condition", "Dose"])["EpochCount"].transform("sum")
        animal_summary["Percent"] = animal_summary["EpochCount"] / animal_totals
        chart_html = ""
        try:
            import plotly.express as px

            chart_frame = grouped.copy()
            chart_frame["Group"] = (
                chart_frame["Condition"].fillna("").astype(str)
                + " / "
                + chart_frame["Dose"].fillna("").astype(str)
                + " / "
                + chart_frame["Session"].fillna("").astype(str)
            )
            chart_frame = chart_frame.groupby(["Group", "HourBin", "PredLabel"], dropna=False)["EpochCount"].sum().reset_index()
            totals = chart_frame.groupby(["Group", "HourBin"])["EpochCount"].transform("sum")
            chart_frame["Percent"] = chart_frame["EpochCount"] / totals
            fig = px.bar(
                chart_frame,
                x="HourBin",
                y="Percent",
                color="PredLabel",
                facet_row="Group",
                barmode="stack",
                color_discrete_map={"Wake": "#d62728", "NREM": "#1f77b4", "REM": "#2ca02c"},
                title="Hourly vigilant-state composition",
            )
            fig.update_layout(
                template="plotly_white",
                height=max(460, 210 * max(1, chart_frame["Group"].nunique())),
                font={"family": "Inter, Segoe UI, Arial", "size": 13},
                yaxis_tickformat=".0%",
                margin={"l": 70, "r": 30, "t": 70, "b": 55},
            )
            fig.update_yaxes(tickformat=".0%", range=[0, 1])
            chart_html = fig.to_html(include_plotlyjs=True, full_html=False)
        except Exception as exc:  # noqa: BLE001
            chart_html = f"<p class='note'>Group chart unavailable: {html.escape(str(exc))}</p>"
        pieces.extend(
            [
                chart_html,
                "<h2>Condition/Dose/Session by Hour</h2>",
                _styled_table(grouped.head(500)),
            ]
        )
        pieces.extend(["<h2>Animal-Level State Percentages</h2>", _styled_table(animal_summary.head(500))])
        if state_path.exists():
            state = pd.read_csv(state_path)
            if not state.empty:
                dataset = (
                    state.groupby("PredLabel", dropna=False)["EpochCount"]
                    .sum()
                    .rename("EpochCount")
                    .reset_index()
                )
                dataset["Percent"] = dataset["EpochCount"] / dataset["EpochCount"].sum()
                pieces.extend(["<h2>Dataset State Percentages</h2>", _styled_table(dataset)])
        return _styled_page("Group Analysis", pieces)

    def _classification_html(cfg: AnalysisConfig, spec: RecordingSpec) -> str:
        output_dir = cfg.dataset_output_dir
        rec_dir = output_dir / spec.recording_id / "scored"
        predictions_path = rec_dir / "predictions_standardized.csv"
        if predictions_path.exists():
            predictions = pd.read_csv(predictions_path)
            counts = predictions["PredLabel"].astype(str).value_counts().rename_axis("State").reset_index(name="Epochs")
            counts["Percent"] = (counts["Epochs"] / counts["Epochs"].sum() * 100).round(2)
            summary = json.loads((rec_dir / "summary.json").read_text(encoding="utf-8")) if (rec_dir / "summary.json").exists() else {}
            cards = [
                ("Model", str(summary.get("ModelAlias", cfg.default_model_alias))),
                ("Epochs", str(len(predictions))),
                ("Epoch length", str(summary.get("EpochLengthSeconds", spec.epoch_length_sec))),
                ("Input units", str(summary.get("InputUnit", "uV"))),
            ]
            return _styled_page(f"{spec.recording_id} Classification", [_cards(cards), "<h2>Prediction Distribution</h2>", _styled_table(counts)])
        return _styled_page(f"{spec.recording_id} Classification", ["<p>Run analysis to generate predictions.</p>"])

        # Retained below only for private-workspace compatibility; public UI returns above.
        summary_path = rec_dir / "summary.json"
        manifest_path = rec_dir / "classification_manifest.json"
        sanity_path = output_dir / "prediction_sanity.csv"
        pieces = []
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                rows = {
                    "Model": summary.get("ModelAlias", ""),
                    "Model packageFile": summary.get("Model packageFile", ""),
                    "InputUnit": summary.get("InputUnit", ""),
                    "UnitScaleApplied": json.dumps(summary.get("UnitScaleApplied", {}), sort_keys=True),
                    "EpochLengthSeconds": summary.get("EpochLengthSeconds", ""),
                    "AggregateScoredEpochs": summary.get("AggregateScoredEpochs", ""),
                    "InputParity": json.dumps(summary.get("InputParity", {}), sort_keys=True),
                }
                pieces.append(
                    _cards(
                        [
                            ("Model", str(rows["Model"])),
                            ("Epochs", str(rows["AggregateScoredEpochs"])),
                            ("Input", str(rows["InputUnit"])),
                            ("Model package", Path(str(rows["Model packageFile"])).name),
                        ]
                    )
                )
                pieces.append("<h2>Run Details</h2>")
                pieces.append(_styled_table(pd.DataFrame([rows])))
            except Exception as exc:  # noqa: BLE001
                pieces.append(f"<p>Could not read summary.json: {html.escape(str(exc))}</p>")
        elif manifest_path.exists():
            pieces.append(f"<pre>{html.escape(manifest_path.read_text(encoding='utf-8', errors='replace'))}</pre>")
        else:
            pieces.append(f"<p>Run analysis to create {html.escape(str(summary_path))}.</p>")
        predictions = rec_dir / "predictions_standardized.csv"
        if predictions.exists():
            try:
                frame = pd.read_csv(predictions, usecols=["PredLabel", "Confidence"])
                counts = frame["PredLabel"].astype(str).value_counts().rename_axis("PredLabel").reset_index(name="EpochCount")
                counts["Percent"] = counts["EpochCount"] / counts["EpochCount"].sum()
                chart_html = ""
                try:
                    import plotly.express as px

                    fig = px.pie(
                        counts,
                        names="PredLabel",
                        values="EpochCount",
                        hole=0.42,
                        color="PredLabel",
                        color_discrete_map={"Wake": "#d62728", "NREM": "#1f77b4", "REM": "#2ca02c"},
                        title="Prediction distribution",
                    )
                    fig.update_layout(template="plotly_white", height=360, font={"family": "Inter, Segoe UI, Arial"})
                    chart_html = fig.to_html(include_plotlyjs=True, full_html=False)
                except Exception as exc:  # noqa: BLE001
                    chart_html = f"<p class='note'>Prediction chart unavailable: {html.escape(str(exc))}</p>"
                pieces.extend(["<h2>Prediction Distribution</h2>", chart_html, _styled_table(counts)])
                pieces.append(
                    "<h2>Confidence</h2>"
                    + _styled_table(pd.DataFrame(
                        [
                            {
                                "Min": pd.to_numeric(frame["Confidence"], errors="coerce").min(),
                                "Max": pd.to_numeric(frame["Confidence"], errors="coerce").max(),
                                "Std": pd.to_numeric(frame["Confidence"], errors="coerce").std(),
                                "UniqueRounded": pd.to_numeric(frame["Confidence"], errors="coerce").round(6).nunique(),
                            }
                        ]
                    ))
                )
            except Exception as exc:  # noqa: BLE001
                pieces.append(f"<p>Could not summarize predictions: {html.escape(str(exc))}</p>")
        if sanity_path.exists():
            try:
                sanity = pd.read_csv(sanity_path)
                match = sanity[sanity["RecordingID"].astype(str) == spec.recording_id]
                if not match.empty:
                    status = str(match.iloc[0].get("Status", ""))
                    if status.lower() != "pass":
                        pieces.append("<h2>Prediction Sanity Failed</h2>")
                    else:
                        pieces.append("<h2>Prediction Sanity Passed</h2>")
                    pieces.append(_styled_table(match))
            except Exception as exc:  # noqa: BLE001
                pieces.append(f"<p>Could not read prediction_sanity.csv: {html.escape(str(exc))}</p>")
        parity_path = rec_dir / "staging" / "input_parity.csv"
        if parity_path.exists():
            pieces.extend(["<h2>EDF vs AccuSleePy Parquet Parity</h2>", _csv_preview(parity_path)])
        return _styled_page(f"{spec.recording_id} Classification", pieces)

    def _styled_page(title: str, pieces: list[str]) -> str:
        return (
            "<html><head>"
            "<style>"
            "body{margin:0;background:#f6f8fb;color:#111827;font-family:Arial,Helvetica,sans-serif;}"
            "main{padding:26px 30px 42px;}"
            "h1{font-size:24px;line-height:1.2;margin:0 0 18px;font-weight:700;}"
            "h2{font-size:16px;line-height:1.25;margin:24px 0 10px;font-weight:700;color:#111827;}"
            ".note{color:#4b5563;font-size:13px;line-height:1.45;}"
            ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:0 0 20px;}"
            ".card{background:white;border:1px solid #dfe5ee;border-radius:8px;padding:13px 15px;box-shadow:0 1px 2px rgba(20,30,50,.04);}"
            ".card .label{font-size:11px;text-transform:uppercase;letter-spacing:.035em;color:#5b667a;margin-bottom:6px;}"
            ".card .value{font-size:18px;font-weight:700;color:#111827;overflow-wrap:anywhere;}"
            "table{border-collapse:collapse;width:100%;background:white;border:1px solid #dfe5ee;border-radius:8px;overflow:hidden;font-size:12px;}"
            "th{background:#eef3f8;color:#111827;text-align:left;font-weight:700;padding:8px 10px;border-bottom:1px solid #dfe5ee;position:sticky;top:0;}"
            "td{padding:8px 10px;border-bottom:1px solid #edf1f5;vertical-align:top;}"
            "tr:nth-child(even) td{background:#fafbfd;}"
            "</style></head><body><main>"
            f"<h1>{html.escape(title)}</h1>"
            + "\n".join(pieces)
            + "</main></body></html>"
        )

    def _styled_table(frame: pd.DataFrame) -> str:
        out = frame.copy()
        for col in out.columns:
            if pd.api.types.is_float_dtype(out[col]):
                out[col] = out[col].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
        return out.to_html(index=False, escape=True, classes="dataframe")

    def _cards(items: list[tuple[str, str]]) -> str:
        cards = []
        for label, value in items:
            cards.append(
                "<div class='card'>"
                f"<div class='label'>{html.escape(label)}</div>"
                f"<div class='value'>{html.escape(value)}</div>"
                "</div>"
            )
        return "<div class='cards'>" + "".join(cards) + "</div>"

    def _weighted(frame: pd.DataFrame, value_column: str, weight_column: str) -> float | str:
        if frame.empty or value_column not in frame.columns or weight_column not in frame.columns:
            return ""
        values = pd.to_numeric(frame[value_column], errors="coerce")
        weights = pd.to_numeric(frame[weight_column], errors="coerce").fillna(0)
        mask = values.notna() & (weights > 0)
        if not mask.any():
            return ""
        return float((values[mask] * weights[mask]).sum() / weights[mask].sum())

    def _scored_epoch_length(cfg: AnalysisConfig, spec: RecordingSpec) -> float | None:
        path = cfg.dataset_output_dir / spec.recording_id / "scored" / "summary.json"
        if not path.exists():
            return None
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            return float(summary.get("EpochLengthSeconds"))
        except Exception:
            return None

    def _labels_for_epoch(cfg: AnalysisConfig, recording_id: str, epoch_index: int) -> dict[str, str]:
        labels: dict[str, str] = {}
        roots = [*cfg.label_roots, cfg.dataset_output_dir]
        rows = [
            row
            for row in inventory_label_files(discover_label_files(roots))
            if row.status == "ok" and row.recording_id == recording_id
        ]
        for row in rows:
            try:
                frame = load_label_frame(row.path, recording_id=recording_id, default_epoch_length=row.epoch_length_sec or 2.5)
                match = frame[pd.to_numeric(frame["EpochIndex"], errors="coerce") == epoch_index]
                if not match.empty:
                    key = f"{row.label_source}:{row.model}"
                    labels[key] = str(match.iloc[0]["PredLabel"])
            except Exception:
                continue
        return labels

    def _first_available_signal(header, *candidates):
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                header.signal_index(candidate)
                return candidate
            except (KeyError, IndexError):
                continue
        return 0

    app = QApplication.instance() or QApplication([])
    window = MainWindow(Path(config_path))
    window.show()
    return app.exec()


