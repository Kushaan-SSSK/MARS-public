from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

STATE_COLORS = {
    "Wake": "#d62728",
    "NREM": "#1f77b4",
    "REM": "#2ca02c",
    "Unknown": "#7f7f7f",
    "TransitionalOrUnclassified": "#9467bd",
}

PLOT_TEMPLATE = "plotly_white"


def write_state_pie(
    state_counts: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "State Percentages",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import plotly.express as px

        fig = px.pie(
            state_counts,
            names="PredLabel",
            values="EpochCount",
            title=title,
            hole=0.35,
        )
        fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    except Exception as exc:  # noqa: BLE001
        _write_fallback_html(output_path, title, state_counts, exc)


def write_confusion_heatmap(
    matrix: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "Label Confusion Matrix",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import plotly.express as px

        fig = px.imshow(
            matrix,
            text_auto=True,
            aspect="auto",
            title=title,
            color_continuous_scale="Blues",
            labels={"x": "Compared label", "y": "AccuSleePy label", "color": "Epochs"},
        )
        fig.update_layout(template=PLOT_TEMPLATE, font={"family": "Arial, Helvetica, sans-serif", "size": 12})
        fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    except Exception as exc:  # noqa: BLE001
        _write_fallback_html(output_path, title, matrix.reset_index(), exc)


def write_disagreement_timeline(
    differences: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "Label Disagreements",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if differences.empty:
        output_path.write_text(
            _empty_page("Label Disagreements", "No disagreements were found for aligned comparison sources."),
            encoding="utf-8",
        )
        return
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        frame = differences.copy()
        frame["EpochStartSeconds"] = pd.to_numeric(frame["EpochStartSeconds"], errors="coerce")
        frame["Hour"] = (frame["EpochStartSeconds"] // 3600).astype("Int64")
        frame["Mismatch"] = frame["AccuSleePyLabel"].astype(str) + " -> " + frame["ComparedLabel"].astype(str)

        top_recordings = (
            frame.groupby("RecordingID")
            .size()
            .sort_values(ascending=False)
            .head(35)
            .index
        )
        heat = (
            frame[frame["RecordingID"].isin(top_recordings)]
            .groupby(["RecordingID", "Hour"], dropna=False)
            .size()
            .rename("Disagreements")
            .reset_index()
        )
        heat_matrix = heat.pivot(index="RecordingID", columns="Hour", values="Disagreements").fillna(0)
        heat_matrix = heat_matrix.reindex(top_recordings)

        source_counts = (
            frame.groupby(["ComparedSource", "ComparedLabel"], dropna=False)
            .size()
            .rename("Disagreements")
            .reset_index()
        )
        mismatch_counts = (
            frame.groupby("Mismatch", dropna=False)
            .size()
            .sort_values(ascending=False)
            .head(12)
            .rename("Disagreements")
            .reset_index()
        )

        fig = make_subplots(
            rows=3,
            cols=2,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "heatmap", "colspan": 2}, None],
                [{"type": "bar"}, {"type": "bar"}],
            ],
            subplot_titles=(
                "",
                "",
                "Hourly disagreement density by recording",
                "Disagreements by compared label",
                "Most common mismatch directions",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.12,
            row_heights=[0.18, 0.48, 0.34],
        )
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=int(len(frame)),
                title={"text": "Disagreement epochs"},
                number={"font": {"size": 34}},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=int(frame["RecordingID"].nunique()),
                title={"text": "Recordings with disagreements"},
                number={"font": {"size": 34}},
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Heatmap(
                z=heat_matrix.to_numpy(),
                x=[str(col) for col in heat_matrix.columns],
                y=heat_matrix.index.tolist(),
                colorscale="YlOrRd",
                colorbar={"title": "Epochs"},
                hovertemplate="Recording %{y}<br>Hour %{x}<br>Disagreements %{z}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        for label, sub in source_counts.groupby("ComparedLabel", dropna=False):
            fig.add_trace(
                go.Bar(
                    x=sub["ComparedSource"],
                    y=sub["Disagreements"],
                    name=str(label),
                    marker_color=STATE_COLORS.get(str(label), "#8a8f98"),
                    hovertemplate="%{x}<br>%{y} disagreements<extra></extra>",
                ),
                row=3,
                col=1,
            )
        fig.add_trace(
            go.Bar(
                x=mismatch_counts["Disagreements"],
                y=mismatch_counts["Mismatch"],
                orientation="h",
                marker_color="#3b6ea8",
                showlegend=False,
                hovertemplate="%{y}<br>%{x} disagreements<extra></extra>",
            ),
            row=3,
            col=2,
        )
        fig.update_layout(
            title=title,
            template=PLOT_TEMPLATE,
            barmode="stack",
            height=980,
            margin={"l": 90, "r": 40, "t": 90, "b": 60},
            font={"family": "Arial, Helvetica, sans-serif", "size": 12, "color": "#111111"},
            legend={"orientation": "h", "y": -0.08},
        )
        fig.update_xaxes(title_text="Hour", row=2, col=1)
        fig.update_xaxes(title_text="Compared source", row=3, col=1)
        fig.update_xaxes(title_text="Disagreement epochs", row=3, col=2)
        fig.update_yaxes(title_text="", row=2, col=1, automargin=True)
        fig.update_yaxes(autorange="reversed", row=3, col=2, automargin=True)
        fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    except Exception as exc:  # noqa: BLE001
        _write_fallback_html(output_path, title, differences, exc)


def write_label_comparison_dashboard(
    summary: pd.DataFrame,
    differences: pd.DataFrame,
    output_path: str | Path,
    *,
    coverage: pd.DataFrame | None = None,
    title: str = "Label Comparison Summary",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if summary.empty and differences.empty:
        output_path.write_text(
            _empty_page(
                "Label Comparison",
                "No comparison label sources were available for this run.",
            ),
            encoding="utf-8",
        )
        return
    _write_lightweight_comparison_dashboard(summary, differences, output_path, coverage=coverage, title=title)
    return
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        s = summary.copy()
        c = coverage.copy() if coverage is not None else pd.DataFrame()
        for col in ["accuracy", "disagreement_pct", "aligned_epoch_count", "disagreement_count"]:
            if col in s.columns:
                s[col] = pd.to_numeric(s[col], errors="coerce")
        for col in ["BaselineEpochs", "AlignedEpochs", "DisagreementEpochs", "CoveragePercent"]:
            if col in c.columns:
                c[col] = pd.to_numeric(c[col], errors="coerce").fillna(0)
        ok = s[s.get("status", pd.Series(dtype=str)).astype(str) == "ok"] if "status" in s.columns else s
        invalid = s[s.get("status", pd.Series(dtype=str)).astype(str) != "ok"] if "status" in s.columns else pd.DataFrame()
        baseline_total = int(c["BaselineEpochs"].max()) if "BaselineEpochs" in c.columns and not c.empty else 0
        full_aligned = int(c.loc[c.get("Status", pd.Series(dtype=str)).astype(str) == "full_dataset", "AlignedEpochs"].sum()) if not c.empty else 0
        partial_aligned = int(c.loc[c.get("Status", pd.Series(dtype=str)).astype(str) == "partial_reference", "AlignedEpochs"].sum()) if not c.empty else int(ok["aligned_epoch_count"].sum()) if not ok.empty else 0
        source = (
            ok.groupby("compared_source", dropna=False)
            .agg(
                Pairs=("recording_id", "count"),
                MeanAccuracy=("accuracy", "mean"),
                MeanDisagreement=("disagreement_pct", "mean"),
                AlignedEpochs=("aligned_epoch_count", "sum"),
            )
            .reset_index()
            if not ok.empty
            else pd.DataFrame(columns=["compared_source", "Pairs", "MeanAccuracy", "MeanDisagreement", "AlignedEpochs"])
        )
        top = (
            ok.sort_values("disagreement_pct", ascending=False)
            .head(25)
            if not ok.empty and "disagreement_pct" in ok.columns
            else pd.DataFrame(columns=["recording_id", "compared_source", "disagreement_pct"])
        )
        status_counts = (
            s.groupby("status", dropna=False)
            .size()
            .rename("Count")
            .reset_index()
            if "status" in s.columns and not s.empty
            else pd.DataFrame(columns=["status", "Count"])
        )

        fig = make_subplots(
            rows=3,
            cols=3,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "bar", "colspan": 3}, None, None],
            ],
            subplot_titles=(
                "",
                "",
                "",
                "Coverage by comparison source",
                "Average accuracy by source",
                "Comparison status counts",
                "Highest disagreement aligned pairs",
            ),
            vertical_spacing=0.13,
            horizontal_spacing=0.10,
            row_heights=[0.2, 0.36, 0.44],
        )
        fig.add_trace(
            go.Indicator(mode="number", value=baseline_total, title={"text": "Total scored baseline epochs"}),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Indicator(mode="number", value=full_aligned, title={"text": "Full-dataset aligned epochs"}),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Indicator(mode="number", value=partial_aligned, title={"text": "Partial/reference aligned epochs"}),
            row=1,
            col=3,
        )
        if not c.empty:
            filtered_coverage = c[c["Status"].astype(str) != "baseline"].copy()
            fig.add_trace(
                go.Bar(
                    x=filtered_coverage["ComparedSource"].astype(str) + " / " + filtered_coverage["ComparedModel"].astype(str),
                    y=filtered_coverage["CoveragePercent"] * 100,
                    marker_color=[
                        "#2f6f4e" if status == "full_dataset" else "#b45309" if status == "partial_reference" else "#8a8f98"
                        for status in filtered_coverage["Status"].astype(str)
                    ],
                    name="Coverage",
                    hovertemplate="%{x}<br>%{y:.2f}% coverage<extra></extra>",
                ),
                row=2,
                col=1,
            )
        fig.add_trace(
            go.Bar(
                x=source["compared_source"],
                y=source["MeanAccuracy"],
                marker_color="#2f6f4e",
                name="Mean accuracy",
                hovertemplate="%{x}<br>Mean accuracy %{y:.3f}<extra></extra>",
            ),
            row=2,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=status_counts["status"],
                y=status_counts["Count"],
                marker_color=["#2f6f4e" if str(v) == "ok" else "#b45309" for v in status_counts["status"]],
                name="Status",
                showlegend=False,
            ),
            row=2,
            col=3,
        )
        labels = top["recording_id"].astype(str) + " / " + top["compared_source"].astype(str)
        fig.add_trace(
            go.Bar(
                x=top["disagreement_pct"] * 100,
                y=labels,
                orientation="h",
                marker_color="#3b6ea8",
                name="Disagreement %",
                hovertemplate="%{y}<br>%{x:.1f}% disagreement<extra></extra>",
            ),
            row=3,
            col=1,
        )
        fig.update_layout(
            title=title,
            template=PLOT_TEMPLATE,
            height=900,
            margin={"l": 110, "r": 40, "t": 90, "b": 60},
            font={"family": "Arial, Helvetica, sans-serif", "size": 12, "color": "#111111"},
            showlegend=False,
            annotations=[
                *list(fig.layout.annotations),
                {
                    "text": (
                        f"{len(invalid)} invalid-alignment pairs are listed in label_comparison_summary.csv. "
                        "Full-dataset metrics require coverage of all baseline epochs."
                    ),
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0,
                    "y": -0.08,
                    "showarrow": False,
                    "align": "left",
                    "font": {"size": 12, "color": "#59657a"},
                },
            ],
        )
        fig.update_yaxes(autorange="reversed", row=3, col=1, automargin=True)
        fig.update_yaxes(range=[0, 100], row=2, col=1)
        fig.update_yaxes(range=[0, 1], row=2, col=2)
        fig.update_xaxes(title_text="Coverage (%)", row=2, col=1)
        fig.update_xaxes(title_text="Accuracy", row=2, col=2)
        fig.update_xaxes(title_text="Pairs", row=2, col=3)
        fig.update_xaxes(title_text="Disagreement (%)", row=3, col=1)
        fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    except Exception as exc:  # noqa: BLE001
        _write_fallback_html(output_path, title, summary, exc)


def write_mars_vs_reference_dashboard(
    summary: pd.DataFrame,
    differences: pd.DataFrame,
    confusion: pd.DataFrame,
    alignment: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    title = "MARS E2.5W9 vs reference Source Labels"
    if summary.empty and alignment.empty:
        output_path.write_text(
            _empty_page(title, "No MARS predictions or reference labels were available for this run."),
            encoding="utf-8",
        )
        return
    s = summary.copy()
    a = alignment.copy()
    d = differences.copy()
    for frame, columns in [
        (s, ["mars_epoch_count", "aligned_epoch_count", "strict_valid_epoch_count", "coverage_percent", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "kappa", "strict_accuracy", "disagreement_count", "disagreement_pct"]),
        (a, ["MarsEpochs", "AlignedEpochs", "MissingEpochs", "CoveragePercent", "StrictValidEpochs"]),
    ]:
        for column in columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
    ok = s[s.get("status", pd.Series(dtype=str)).astype(str) == "ok"] if not s.empty else pd.DataFrame()
    total_mars = int(a["MarsEpochs"].sum()) if "MarsEpochs" in a.columns else int(s["mars_epoch_count"].sum()) if "mars_epoch_count" in s.columns else 0
    aligned = int(a["AlignedEpochs"].sum()) if "AlignedEpochs" in a.columns else int(ok["aligned_epoch_count"].sum()) if not ok.empty else 0
    strict = int(a["StrictValidEpochs"].sum()) if "StrictValidEpochs" in a.columns else int(ok["strict_valid_epoch_count"].sum()) if not ok.empty else 0
    coverage_pct = aligned / total_mars if total_mars else 0.0
    strict_metrics = _aggregate_metric_row(ok, strict_only=True)
    all_metrics = _aggregate_metric_row(ok, strict_only=False)
    status_counts = (
        a.groupby("Status", dropna=False).size().rename("Recordings").reset_index()
        if "Status" in a.columns and not a.empty
        else pd.DataFrame()
    )
    pieces = [
        _cards(
            [
                ("MARS Epochs", f"{total_mars:,}"),
                ("Aligned reference Epochs", f"{aligned:,}"),
                ("Strict-Valid Epochs", f"{strict:,}"),
                ("Coverage", f"{coverage_pct * 100:.2f}%"),
            ]
        ),
        "<p class='note'>This page compares only our trained offline MARS/AccuSleePy model against reference source-code labels. REST, IntelliSleepScorer, and external/original AccuSleePy labels are excluded.</p>",
    ]
    metric_rows = pd.DataFrame(
        [
            {"View": "All aligned epochs", **all_metrics},
            {"View": "Strict valid reference epochs", **strict_metrics},
        ]
    )
    pieces.append("<h2>Metrics</h2>" + _mini_table(metric_rows))
    if not status_counts.empty:
        pieces.append("<h2>Alignment Status</h2>" + _mini_table(status_counts))
    if not a.empty:
        pieces.append("<h2>Per-Recording Alignment</h2>" + _mini_table(a.head(200)))
    if not confusion.empty:
        pieces.append("<h2>Confusion Matrix Long Form</h2>" + _mini_table(confusion.head(200)))
    if not d.empty:
        mismatch = (
            d.groupby(["ReferenceLabel", "MarsLabel"], dropna=False)
            .size()
            .rename("MismatchEpochs")
            .reset_index()
            .sort_values("MismatchEpochs", ascending=False)
        )
        pieces.append("<h2>Mismatches</h2>" + _mini_table(mismatch.head(100)))
    output_path.write_text(_page(title, pieces), encoding="utf-8")


def _aggregate_metric_row(summary: pd.DataFrame, *, strict_only: bool) -> dict[str, object]:
    if summary.empty:
        return {
            "Accuracy": "",
            "BalancedAccuracy": "",
            "MacroF1": "",
            "WeightedF1": "",
            "Kappa": "",
            "DisagreementPct": "",
        }
    weight_col = "strict_valid_epoch_count" if strict_only else "aligned_epoch_count"
    prefix = "strict_" if strict_only else ""
    weights = pd.to_numeric(summary.get(weight_col, 0), errors="coerce").fillna(0)
    out: dict[str, object] = {}
    for label, column in [
        ("Accuracy", f"{prefix}accuracy"),
        ("BalancedAccuracy", f"{prefix}balanced_accuracy"),
        ("MacroF1", f"{prefix}macro_f1"),
        ("WeightedF1", f"{prefix}weighted_f1"),
        ("Kappa", f"{prefix}kappa"),
    ]:
        values = pd.to_numeric(summary.get(column, pd.Series(dtype=float)), errors="coerce")
        mask = values.notna() & (weights > 0)
        out[label] = float((values[mask] * weights[mask]).sum() / weights[mask].sum()) if mask.any() and weights[mask].sum() else ""
    if strict_only:
        out["DisagreementPct"] = ""
    else:
        disagreements = pd.to_numeric(summary.get("disagreement_count", 0), errors="coerce").fillna(0).sum()
        aligned = weights.sum()
        out["DisagreementPct"] = float(disagreements / aligned) if aligned else ""
    return out


def _write_lightweight_comparison_dashboard(
    summary: pd.DataFrame,
    differences: pd.DataFrame,
    output_path: Path,
    *,
    coverage: pd.DataFrame | None,
    title: str,
) -> None:
    s = summary.copy()
    c = coverage.copy() if coverage is not None else pd.DataFrame()
    pieces = []
    baseline_total = 0
    full_aligned = 0
    partial_aligned = 0
    if not c.empty:
        for col in ["BaselineEpochs", "AlignedEpochs", "MissingEpochs", "CoveragePercent"]:
            if col in c.columns:
                c[col] = pd.to_numeric(c[col], errors="coerce").fillna(0)
        baseline_total = int(c["BaselineEpochs"].max()) if "BaselineEpochs" in c.columns else 0
        if "Status" in c.columns:
            full_aligned = int(c.loc[c["Status"].astype(str) == "full_dataset", "AlignedEpochs"].sum())
            partial_aligned = int(c.loc[c["Status"].astype(str) == "partial_reference", "AlignedEpochs"].sum())
        pieces.append("<h2>Coverage</h2>" + _coverage_svg(c) + _mini_table(c.head(200)))
    if not s.empty:
        for col in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "kappa", "aligned_epoch_count", "disagreement_count", "disagreement_pct"]:
            if col in s.columns:
                s[col] = pd.to_numeric(s[col], errors="coerce")
        ok = s[s.get("status", pd.Series(dtype=str)).astype(str) == "ok"] if "status" in s.columns else s
        if not ok.empty:
            by_source = (
                ok.groupby("compared_source", dropna=False)
                .agg(
                    RecordingPairs=("recording_id", "count"),
                    AlignedEpochs=("aligned_epoch_count", "sum"),
                    MeanAccuracy=("accuracy", "mean"),
                    MeanDisagreementPct=("disagreement_pct", "mean"),
                )
                .reset_index()
            )
            pieces.append("<h2>Aligned Source Metrics</h2>" + _mini_table(by_source))
            top = ok.sort_values("disagreement_pct", ascending=False).head(50)
            pieces.append("<h2>Highest Disagreement Pairs</h2>" + _mini_table(top))
        invalid = s[s.get("status", pd.Series(dtype=str)).astype(str) != "ok"] if "status" in s.columns else pd.DataFrame()
        if not invalid.empty:
            pieces.append("<h2>Invalid Or Unaligned Pairs</h2>" + _mini_table(invalid.head(100)))
    if not differences.empty:
        diff_counts = (
            differences.groupby(["ComparedSource", "ComparedLabel"], dropna=False)
            .size()
            .rename("DisagreementEpochs")
            .reset_index()
            .sort_values("DisagreementEpochs", ascending=False)
        )
        pieces.append("<h2>Disagreement Summary</h2>" + _mini_table(diff_counts.head(100)))
    cards = _cards(
        [
            ("Baseline Epochs", f"{baseline_total:,}"),
            ("Full-Dataset Aligned", f"{full_aligned:,}"),
            ("Partial/Reference Aligned", f"{partial_aligned:,}"),
            ("Disagreement Rows", f"{len(differences):,}"),
        ]
    )
    message = (
        "<p class='note'>Full-dataset metrics require coverage of all baseline epochs. "
        "Partial reference comparisons are reported separately and should not be read as whole-dataset performance.</p>"
    )
    output_path.write_text(_page(title, [cards, message, *pieces]), encoding="utf-8")


def _coverage_svg(coverage: pd.DataFrame) -> str:
    if coverage.empty or "CoveragePercent" not in coverage.columns:
        return "<p class='note'>No coverage rows available.</p>"
    rows = coverage[coverage.get("Status", pd.Series(dtype=str)).astype(str) != "baseline"].copy()
    if rows.empty:
        return "<p class='note'>No compared sources available.</p>"
    rows = rows.head(30)
    height = 38 + 28 * len(rows)
    parts = [f'<svg viewBox="0 0 900 {height}" role="img" aria-label="Comparison coverage">']
    parts.append('<text x="18" y="24" font-size="14" font-weight="700" fill="#17213a">Coverage by source</text>')
    y = 42
    for _, row in rows.iterrows():
        label = f"{row.get('ComparedSource', '')} / {row.get('ComparedModel', '')}"
        pct = float(row.get("CoveragePercent", 0)) * 100
        status = str(row.get("Status", ""))
        color = "#2f6f4e" if status == "full_dataset" else "#b45309" if status == "partial_reference" else "#8a8f98"
        parts.append(f'<text x="18" y="{y + 14}" font-size="11" fill="#17213a">{html.escape(label[:42])}</text>')
        parts.append(f'<rect x="300" y="{y}" width="{max(0, min(100, pct)) * 5.2:.2f}" height="18" fill="{color}"/>')
        parts.append(f'<text x="832" y="{y + 14}" font-size="11" fill="#59657a" text-anchor="end">{pct:.1f}%</text>')
        y += 28
    parts.append("</svg>")
    return "".join(parts)


def _page(title: str, pieces: list[str]) -> str:
    return _style_block() + f"<main><h1>{html.escape(title)}</h1>" + "\n".join(pieces) + "</main></body></html>"


def _mini_table(frame: pd.DataFrame) -> str:
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    return out.to_html(index=False, escape=True)


def _cards(items: list[tuple[str, str]]) -> str:
    return "<div class='cards'>" + "".join(
        "<div class='card'>"
        f"<div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{html.escape(value)}</div>"
        "</div>"
        for label, value in items
    ) + "</div>"


def write_trace_envelope(
    envelope: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "Full Recording Trace Envelope",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from bokeh.embed import file_html
        from bokeh.models import Band, ColumnDataSource
        from bokeh.plotting import figure
        from bokeh.resources import INLINE

        source = ColumnDataSource(envelope)
        plot = figure(
            title=title,
            x_axis_label="Time (s)",
            y_axis_label="Amplitude (uV)",
            sizing_mode="stretch_width",
            height=360,
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        plot.line("time_seconds", "mean", source=source, line_width=1.5)
        band = Band(
            base="time_seconds",
            lower="min",
            upper="max",
            source=source,
            level="underlay",
            fill_alpha=0.25,
            line_alpha=0.0,
        )
        plot.add_layout(band)
        output_path.write_text(file_html(plot, INLINE, title), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        _write_fallback_html(output_path, title, envelope, exc)


def write_epoch_trace(
    times: pd.Series,
    values: pd.Series,
    output_path: str | Path,
    *,
    title: str = "Epoch Trace",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"time_seconds": times, "value": values})
    try:
        import plotly.express as px

        fig = px.line(frame, x="time_seconds", y="value", title=title)
        fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    except Exception as exc:  # noqa: BLE001
        _write_fallback_html(output_path, title, frame, exc)


def _write_fallback_html(
    output_path: Path,
    title: str,
    frame: pd.DataFrame,
    exc: Exception,
) -> None:
    table = frame.head(500).to_html(index=False, escape=True)
    output_path.write_text(
        f"{_style_block()}<main><h1>{title}</h1><p class='note'>Interactive plot unavailable: {exc}</p>{table}</main>",
        encoding="utf-8",
    )


def _empty_page(title: str, message: str) -> str:
    return f"{_style_block()}<main><h1>{title}</h1><p class='note'>{message}</p></main>"


def _style_block() -> str:
    return """
<html>
<head>
<style>
body { margin: 0; font-family: Arial, Helvetica, sans-serif; color: #111111; background: #f7f8fb; }
main { padding: 28px 34px; }
h1 { margin: 0 0 12px; font-size: 28px; font-weight: 700; }
h2 { margin: 24px 0 10px; font-size: 18px; }
.note { color: #59657a; font-size: 15px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 0 0 18px; }
.card { background: white; border: 1px solid #d8dee8; border-radius: 8px; padding: 13px 15px; }
.card .label { font-size: 12px; text-transform: uppercase; color: #59657a; margin-bottom: 6px; }
.card .value { font-size: 20px; font-weight: 700; color: #17213a; }
svg { background: white; border: 1px solid #d8dee8; border-radius: 8px; max-width: 100%; height: auto; }
table { border-collapse: collapse; width: 100%; background: white; border: 1px solid #d8dee8; }
th { background: #eef2f7; color: #17213a; text-align: left; padding: 9px 10px; border-bottom: 1px solid #d8dee8; }
td { padding: 8px 10px; border-bottom: 1px solid #edf0f4; }
</style>
</head>
<body>
"""

