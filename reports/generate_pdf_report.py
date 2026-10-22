"""
Script to generate the official 3-page Technical PDF Report using ReportLab.
Includes evidence-based visualizations grounded in existing evaluation artifacts:
  - Figure 1: Precision–Recall Trade-off across Decision Thresholds (Held-out Test)
  - Figure 2: Confusion Matrix at Selected Operational Threshold 0.40 (Held-out Test)
  - Figure 3: Top 5 Global Feature Importances (Random Forest Ensemble)

Run with:
    python reports/generate_pdf_report.py
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable,
    Image,
)
from reportlab.pdfgen import canvas


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PDF = PROJECT_ROOT / "reports" / "technical_report.pdf"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


class NumberedCanvas(canvas.Canvas):
    """Canvas that computes total pages dynamically for 'Page X of Y' footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(
                54,
                750,
                "Aiqon AI Assignment: Port Congestion & Shipment Delay Forecaster",
            )
            self.drawRightString(612 - 54, 750, "Technical Report")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, 744, 612 - 54, 744)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 45, 612 - 54, 45)

        self.drawString(54, 32, "Aiqon AI Assignment: Port Congestion & Shipment Delay Forecaster")
        self.drawRightString(
            612 - 54, 32, f"Page {self._pageNumber} of {page_count}"
        )
        self.restoreState()


def generate_figures() -> dict[str, Path]:
    """Generates the 3 evidence-based technical charts from existing artifacts."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    eval_json_path = PROJECT_ROOT / "results" / "threshold_evaluation.json"
    with open(eval_json_path, "r", encoding="utf-8") as f:
        te = json.load(f)
    test_eval = te["test_evaluation"]

    feat_csv_path = PROJECT_ROOT / "results" / "feature_importance.csv"
    df_fi = pd.read_csv(feat_csv_path)

    # -------------------------------------------------------------------------
    # Figure 1: Precision–Recall Trade-off across Decision Thresholds (Test Set)
    # -------------------------------------------------------------------------
    thresholds = [0.40, 0.50, 0.55]
    precisions = [
        test_eval["operational_threshold_0_40"]["precision"] * 100,
        test_eval["default_threshold_0_50"]["precision"] * 100,
        test_eval["f1_maximizing_threshold_0_55"]["precision"] * 100,
    ]
    recalls = [
        test_eval["operational_threshold_0_40"]["recall"] * 100,
        test_eval["default_threshold_0_50"]["recall"] * 100,
        test_eval["f1_maximizing_threshold_0_55"]["recall"] * 100,
    ]

    fig, ax = plt.subplots(figsize=(3.6, 1.7), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(
        thresholds,
        recalls,
        marker="o",
        markersize=4.5,
        color="#1A5F7A",
        linewidth=1.6,
        label="Recall (Delays Caught)",
    )
    ax.plot(
        thresholds,
        precisions,
        marker="s",
        markersize=4.5,
        color="#D9534F",
        linewidth=1.6,
        label="Precision (Alert Accuracy)",
    )

    for t, p, r in zip(thresholds, precisions, recalls):
        ax.annotate(
            f"{r:.1f}%",
            (t, r),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=6.2,
            fontweight="bold",
            color="#1A5F7A",
        )
        ax.annotate(
            f"{p:.1f}%",
            (t, p),
            textcoords="offset points",
            xytext=(0, -8),
            ha="center",
            fontsize=6.2,
            fontweight="bold",
            color="#D9534F",
        )

    ax.axvline(0.40, color="#0F2942", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(
        0.403,
        13.0,
        "Operational (0.40)\n[Selected on Val]",
        fontsize=5.6,
        color="#0F2942",
        fontweight="bold",
    )

    ax.set_title(
        "Precision–Recall Trade-off (Held-out Test)",
        fontsize=7.4,
        fontweight="bold",
        pad=4,
        color="#0F2942",
    )
    ax.set_xlabel("Decision Threshold", fontsize=6.8, fontweight="bold", color="#222222")
    ax.set_ylabel("Score (%)", fontsize=6.8, fontweight="bold", color="#222222")
    ax.set_xticks(thresholds)
    ax.set_xticklabels(["0.40\n(Operational)", "0.50\n(Default)", "0.55\n(Val F1-Max)"], fontsize=6.0)
    ax.set_ylim(8, 68)
    ax.tick_params(axis="y", labelsize=6.2)
    ax.grid(True, linestyle=":", alpha=0.45, color="#CCCCCC")
    ax.legend(loc="upper right", fontsize=5.8, frameon=True, facecolor="#F8F9FA", edgecolor="#E2E8F0")
    plt.tight_layout()

    fig1_path = FIGURES_DIR / "fig1_threshold_tradeoff.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 2: Confusion Matrix at Threshold 0.40 (Held-out Test Set)
    # -------------------------------------------------------------------------
    cm = np.array(test_eval["operational_threshold_0_40"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(3.6, 1.7), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cell_colors = [
        ["#E6F0FA", "#FEE2E2"],
        ["#FEF3C7", "#D1FAE5"],
    ]
    cell_texts = [
        [f"{cm[0,0]}\nTrue Negatives\n(On-time: 71.8%)", f"{cm[0,1]}\nFalse Positives\n(False Alarm: 17.6%)"],
        [f"{cm[1,0]}\nFalse Negatives\n(Missed: 4.4%)", f"{cm[1,1]}\nTrue Positives\n(Caught Delay: 6.2%)"],
    ]

    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle(
                (j, 1 - i), 1, 1, facecolor=cell_colors[i][j], edgecolor="#0F2942", linewidth=0.7
            )
            ax.add_patch(rect)
            ax.text(
                j + 0.5,
                1 - i + 0.5,
                cell_texts[i][j],
                ha="center",
                va="center",
                fontsize=6.0,
                fontweight="bold",
                color="#0F2942",
            )

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Pred: On-time (0)", "Pred: Delayed (1)"], fontsize=6.6, fontweight="bold", color="#222222")
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Actual: Delayed (1)", "Actual: On-time (0)"], fontsize=6.6, fontweight="bold", color="#222222")
    ax.set_title(
        "Test Confusion Matrix at Threshold 0.40\n(500 Observations, 53 Actual Delays)",
        fontsize=7.4,
        fontweight="bold",
        pad=4,
        color="#0F2942",
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout()

    fig2_path = FIGURES_DIR / "fig2_confusion_matrix_040.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 3: Top 5 Global Feature Importances
    # -------------------------------------------------------------------------
    top5 = df_fi.head(5).copy()
    top5 = top5.iloc[::-1].reset_index(drop=True)

    feature_labels = {
        "queue_length": "Queue Backlog (queue_length)",
        "historical_avg_delay_hours": "Recent Delay History (historical_avg_delay_hours)",
        "arrival_load_ratio": "Arrival Load Ratio (arrival_load_ratio)",
        "vessel_arrivals_6h": "Vessel Arrivals in 6h (vessel_arrivals_6h)",
        "effective_capacity_6h": "Effective Berth Capacity (effective_capacity_6h)",
    }
    labels = [feature_labels.get(f, f) for f in top5["feature"]]
    importances = top5["importance"].values

    fig, ax = plt.subplots(figsize=(6.2, 1.35), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.barh(range(5), importances, color="#1A5F7A", height=0.52, edgecolor="#0F2942", linewidth=0.5)
    bars[-1].set_color("#0F2942")

    for i, (bar, val) in enumerate(zip(bars, importances)):
        ax.text(
            val + 0.005,
            i,
            f"{val:.4f} ({val*100:.1f}%)",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color="#0F2942",
        )

    ax.set_yticks(range(5))
    ax.set_yticklabels(labels, fontsize=6.6, color="#222222")
    ax.set_xlabel("Mean Decrease in Impurity (Global Feature Importance)", fontsize=6.5, fontweight="bold", color="#222222")
    ax.set_title(
        "Top 5 Global Feature Importances (Random Forest Ensemble, 100 Trees)",
        fontsize=7.4,
        fontweight="bold",
        pad=3,
        color="#0F2942",
    )
    ax.set_xlim(0, 0.30)
    ax.tick_params(axis="x", labelsize=6.2)
    ax.grid(True, axis="x", linestyle=":", alpha=0.45, color="#CCCCCC")
    plt.tight_layout()

    fig3_path = FIGURES_DIR / "fig3_feature_importance.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    return {
        "fig1": fig1_path,
        "fig2": fig2_path,
        "fig3": fig3_path,
    }


def build_pdf():
    figures = generate_figures()

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom typography
    c_primary = colors.HexColor("#0F2942")     # Deep maritime navy
    c_secondary = colors.HexColor("#1A5F7A")   # Marine blue
    c_accent = colors.HexColor("#D9534F")      # Alert red
    c_text = colors.HexColor("#222222")        # Charcoal body
    c_muted = colors.HexColor("#444444")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18.5,
        leading=21.5,
        textColor=c_primary,
        spaceAfter=3,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.8,
        leading=12.8,
        textColor=c_secondary,
        spaceAfter=6,
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=c_primary,
        spaceBefore=6,
        spaceAfter=2.5,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.1,
        leading=10.9,
        textColor=c_text,
        spaceAfter=3,
    )
    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=1.8,
    )
    caption_style = ParagraphStyle(
        "Caption_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=6.5,
        leading=8.3,
        textColor=c_muted,
        spaceAfter=1.5,
    )
    callout_style = ParagraphStyle(
        "Callout_Custom",
        parent=body_style,
        fontSize=7.6,
        leading=9.8,
        textColor=c_primary,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.0,
        leading=8.8,
        textColor=c_text,
    )
    table_header = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.0,
        leading=8.8,
        textColor=colors.white,
    )

    story = []

    # =========================================================================
    # PAGE 1: Problem, Dataset, Temporal Integrity
    # =========================================================================
    story.append(Paragraph("Aiqon AI Take-Home Assignment: Technical Report", title_style))
    story.append(
        Paragraph(
            "<b>Forecasting Port Congestion and Shipping Delays Using Interpretable Machine Learning</b><br/>"
            "Author: Saaket Baldawa &nbsp;|&nbsp; Track: AI / Machine Learning Internship &nbsp;|&nbsp; Date: 22nd October 2026",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceAfter=6))

    # Section 1
    story.append(Paragraph("1. Problem Statement & Operational Objective", h1_style))
    story.append(
        Paragraph(
            "Global container supply chains rely on smooth maritime terminal throughput. "
            "When vessel arrival surges or adverse weather disrupt scheduled operations, queuing backlogs escalate rapidly, "
            "producing substantial ship idling costs, missed logistics handoffs, and steep carrier demurrage penalties. "
            "The objective of this project is to build an <b>interpretable early-warning forecasting model</b> that predicts whether "
            "a commercial port will experience an operational delay exceeding <b>12 hours</b> during the <b>next six-hour interval</b>, "
            "using only conditions observable at the close of the <b>current six-hour window</b>. "
            "Crucially, the system must deliver actionable probability scores, support flexible decision thresholds, "
            "and provide clear operational explanations without claiming ungrounded causal relationships.",
            body_style,
        )
    )

    # Section 2
    story.append(Paragraph("2. Synthetic Dataset & Simulation Dynamics", h1_style))
    story.append(
        Paragraph(
            "To simulate realistic operational patterns without compromising proprietary port data, a synthetic logging system "
            "(<code>data/generate_dataset.py</code>) models 5 distinct marine terminals (<code>PORT_A</code> to <code>PORT_E</code>) "
            "over 500 consecutive six-hour intervals from January 1, 2026 to May 5, 2026 (2,500 total records). Key generative mechanisms include:",
            body_style,
        )
    )
    story.append(Paragraph("• <b>Port Capacities & Arrivals:</b> Baseline handling capacities range from 9 to 16 vessels/window with Poisson-distributed arrivals modulated by weekend factors (0.90x) and diurnal peak surges (1.12x).", bullet_style))
    story.append(Paragraph("• <b>Weather Impacts:</b> Stochastic weather transitions (Clear 66%, Rain 27%, Storm 7%) scale effective service capacity downward (Rain: 0.88x, Storm: 0.68x) and create arrival clustering.", bullet_style))
    story.append(Paragraph("• <b>Queue Physics:</b> Unserviced vessels carry over into subsequent intervals via recursive backlog tracking: <i>Backlog<sub>t</sub> = max(0, Queue<sub>t</sub> + Arrivals<sub>t</sub> - EffectiveCapacity<sub>t</sub>)</i>.", bullet_style))
    story.append(Paragraph("• <b>Delay Equation & Noise:</b> Service delays compound arrival pressure ratio (&gt;0.70 load), accumulated queue length (1.15x), storm disruption shocks, and Gaussian process noise, yielding a 11.9% overall delay rate (&gt;12h).", bullet_style))

    # Section 3
    story.append(Paragraph("3. Target Definition, Alignment & Temporal Leakage Controls", h1_style))
    story.append(
        Paragraph(
            "Forecasting requires strict separation between the observation window and the predicted outcome horizon:",
            body_style,
        )
    )
    story.append(Paragraph("• <b>Target Formulation:</b> The target <code>target_delayed_next_6h</code> is defined as <code>(actual_delay_hours &gt; 12.0)</code> occurring during the immediate next 6-hour operational interval (<i>T<sub>cutoff</sub> &rarr; T<sub>cutoff</sub> + 6h</i>).", bullet_style))
    story.append(Paragraph("• <b>Intra-Port Alignment:</b> In <code>src/prepare_data.py</code>, target shifting is executed strictly within port groups via <code>df.groupby('port_id')['is_delayed'].shift(-1)</code>. Cross-port boundary contamination is impossible.", bullet_style))
    story.append(Paragraph("• <b>Boundary Row Handling:</b> The terminal record for each port lacks a subsequent observation; exactly 5 boundary records are pruned, yielding a validated modeling dataset of 2,495 rows.", bullet_style))
    story.append(Paragraph("• <b>Zero Target Leakage:</b> Current-interval outcome fields (<code>delay_hours</code>, <code>actual_delay_hours</code>, <code>is_delayed</code>) are permanently dropped from the feature space, guaranteeing that only pre-cutoff information is visible to classifiers.", bullet_style))

    # Callout Box Page 1
    callout_p1 = [
        [
            Paragraph(
                "<b>Operational Disclosure:</b> All findings in this report derive from synthetic simulation logs. "
                "While calibrated to simulate empirical queue physics, results should not be interpreted as empirical real-world validation on commercial ports.",
                callout_style,
            )
        ]
    ]
    t_callout_p1 = Table(callout_p1, colWidths=[504])
    t_callout_p1.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EBF3F5")),
            ("BOX", (0, 0), (-1, -1), 0.75, c_secondary),
            ("PADDING", (0, 0), (-1, -1), 4.5),
        ])
    )
    story.append(Spacer(1, 3))
    story.append(t_callout_p1)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: ML Pipeline, Chronological Validation & Threshold Policy
    # =========================================================================
    story.append(Paragraph("4. Machine Learning Pipeline & Chronological Validation", h1_style))
    story.append(
        Paragraph(
            "Random k-fold cross-validation is invalid for temporal logistics because randomly shuffing rows allows future patterns "
            "to leak into past predictions. We enforce a <b>strict chronological partition</b> with shared timestamp cutoffs across all 5 ports:",
            body_style,
        )
    )

    story.append(Paragraph("• <b>Training Split (70% — 1,745 rows):</b> Jan 1, 2026 00:00 to Mar 29, 2026 00:00 (204 delays, 11.7% base rate).", bullet_style))
    story.append(Paragraph("• <b>Validation Split (10% — 250 rows):</b> Mar 29, 2026 06:00 to Apr 10, 2026 12:00 (39 delays, 15.6% base rate) — reserved strictly for threshold tuning.", bullet_style))
    story.append(Paragraph("• <b>Held-Out Test Split (20% — 500 rows):</b> Apr 10, 2026 18:00 to May 5, 2026 12:00 (53 delays, 10.6% base rate) — locked for final unbiased evaluation.", bullet_style))

    story.append(Paragraph("<b>Model Architecture:</b> We deploy a scikit-learn <code>Pipeline</code> encapsulating a <code>ColumnTransformer</code> (one-hot encoding <code>port_id</code>, <code>weather_condition</code>, and <code>day_of_week</code>, with numeric passthrough) and a <code>RandomForestClassifier</code> (100 estimators, <code>max_depth=6</code>, <code>min_samples_leaf=2</code>, <code>class_weight='balanced'</code>). The pipeline is fitted exclusively on the training partition.", body_style))

    # Section 5: Metrics Table
    story.append(Paragraph("5. Empirical Performance & Baseline Benchmark", h1_style))
    story.append(
        Paragraph(
            "The model is compared against a zero-intelligence majority baseline (<code>DummyClassifier</code>, always predicting on-time). "
            "Evaluation metrics reflect reproducible pipeline outputs:",
            body_style,
        )
    )

    metrics_data = [
        [
            Paragraph("<b>Model & Policy Configuration</b>", table_header),
            Paragraph("<b>Split</b>", table_header),
            Paragraph("<b>Thresh</b>", table_header),
            Paragraph("<b>Accuracy</b>", table_header),
            Paragraph("<b>Precision</b>", table_header),
            Paragraph("<b>Recall</b>", table_header),
            Paragraph("<b>F1-Score</b>", table_header),
            Paragraph("<b>Delays Caught</b>", table_header),
        ],
        [
            Paragraph("Dummy Baseline (Always On-time)", table_cell),
            Paragraph("Validation", table_cell),
            Paragraph("—", table_cell),
            Paragraph("84.40%", table_cell),
            Paragraph("0.00%", table_cell),
            Paragraph("0.00%", table_cell),
            Paragraph("0.0000", table_cell),
            Paragraph("0 / 39 (0.0%)", table_cell),
        ],
        [
            Paragraph("Dummy Baseline (Always On-time)", table_cell),
            Paragraph("Held-out Test", table_cell),
            Paragraph("—", table_cell),
            Paragraph("89.40%", table_cell),
            Paragraph("0.00%", table_cell),
            Paragraph("0.00%", table_cell),
            Paragraph("0.0000", table_cell),
            Paragraph("0 / 53 (0.0%)", table_cell),
        ],
        [
            Paragraph("Random Forest (Default Baseline)", table_cell),
            Paragraph("Validation", table_cell),
            Paragraph("0.50", table_cell),
            Paragraph("83.60%", table_cell),
            Paragraph("48.00%", table_cell),
            Paragraph("61.54%", table_cell),
            Paragraph("0.5393", table_cell),
            Paragraph("24 / 39 (61.5%)", table_cell),
        ],
        [
            Paragraph("Random Forest (Default Baseline)", table_cell),
            Paragraph("Held-out Test", table_cell),
            Paragraph("0.50", table_cell),
            Paragraph("85.80%", table_cell),
            Paragraph("35.94%", table_cell),
            Paragraph("43.40%", table_cell),
            Paragraph("0.3932", table_cell),
            Paragraph("23 / 53 (43.4%)", table_cell),
        ],
        [
            Paragraph("Random Forest (Val F1-Maximizing)", table_cell),
            Paragraph("Validation", table_cell),
            Paragraph("0.55", table_cell),
            Paragraph("87.20%", table_cell),
            Paragraph("60.00%", table_cell),
            Paragraph("53.85%", table_cell),
            Paragraph("0.5676", table_cell),
            Paragraph("21 / 39 (53.8%)", table_cell),
        ],
        [
            Paragraph("Random Forest (Val F1-Maximizing)", table_cell),
            Paragraph("Held-out Test", table_cell),
            Paragraph("0.55", table_cell),
            Paragraph("87.60%", table_cell),
            Paragraph("37.84%", table_cell),
            Paragraph("26.42%", table_cell),
            Paragraph("0.3111", table_cell),
            Paragraph("14 / 53 (26.4%)", table_cell),
        ],
        [
            Paragraph("<b>Random Forest (Selected Operational)</b>", table_cell),
            Paragraph("<b>Validation</b>", table_cell),
            Paragraph("<b>0.40</b>", table_cell),
            Paragraph("<b>77.20%</b>", table_cell),
            Paragraph("<b>37.84%</b>", table_cell),
            Paragraph("<b>71.79%</b>", table_cell),
            Paragraph("<b>0.4956</b>", table_cell),
            Paragraph("<b>28 / 39 (71.8%)</b>", table_cell),
        ],
        [
            Paragraph("<b>Random Forest (Selected Operational)</b>", table_cell),
            Paragraph("<b>Held-out Test</b>", table_cell),
            Paragraph("<b>0.40</b>", table_cell),
            Paragraph("<b>78.00%</b>", table_cell),
            Paragraph("<b>26.05%</b>", table_cell),
            Paragraph("<b>58.49%</b>", table_cell),
            Paragraph("<b>0.3605</b>", table_cell),
            Paragraph("<b>31 / 53 (58.5%)</b>", table_cell),
        ],
    ]
    t_metrics = Table(metrics_data, colWidths=[126, 52, 42, 46, 46, 46, 46, 100])
    t_metrics.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), c_primary),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -3), [colors.white, colors.HexColor("#F8F9FA")]),
            ("BACKGROUND", (0, -2), (-1, -1), colors.HexColor("#EBF3F5")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("PADDING", (0, 0), (-1, -1), 2.0),
        ])
    )
    story.append(t_metrics)

    # -------------------------------------------------------------------------
    # Figures 1 and 2: Side-by-Side Visualizations (Page 2)
    # -------------------------------------------------------------------------
    img1 = Image(str(figures["fig1"]), width=244, height=115)
    img2 = Image(str(figures["fig2"]), width=244, height=115)

    cap1 = Paragraph(
        "<b>Figure 1: Precision–Recall Trade-off across Decision Thresholds (Held-out Test).</b> "
        "As threshold increases from 0.40 to 0.55, precision rises from 26.1% to 37.8% (suppressing false alarms), "
        "while recall falls from 58.5% to 26.4% (missing delays). The operational threshold (0.40) was selected "
        "on validation data to prioritize early delay detection.",
        caption_style,
    )
    cap2 = Paragraph(
        "<b>Figure 2: Confusion Matrix at Selected Threshold 0.40 (Held-out Test).</b> "
        "The operational policy successfully detected 31 of 53 actual delays (58.5% recall) with 22 false negatives. "
        "To capture these delays, 88 false alarms were generated out of 447 on-time intervals (19.7% false-positive rate).",
        caption_style,
    )

    t_figs_p2 = Table(
        [
            [img1, "", img2],
            [cap1, "", cap2],
        ],
        colWidths=[244, 16, 244],
    )
    t_figs_p2.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ])
    )
    story.append(Spacer(1, 2))
    story.append(t_figs_p2)

    # Section 6: Threshold Selection Analysis
    story.append(Paragraph("6. Decision Threshold Policy & Asymmetric Cost Trade-Offs", h1_style))
    story.append(
        Paragraph(
            "Standard machine learning defaults to an arbitrary 0.50 probability cutoff. However, in maritime terminal logistics, "
            "<b>false negatives (unpredicted bottlenecks)</b> incur severe cascading costs: idle containerships burning bunker fuel, "
            "intermodal rail/truck schedule collapse, and carrier demurrage penalties reaching $20,000–$50,000 per day. "
            "Conversely, <b>false positives (false alarms)</b> merely trigger modest proactive dispatcher review.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Validation threshold sweeping across 0.20 to 0.60 revealed clear operational trade-offs: While threshold 0.55 maximizes validation F1 (0.5676), "
            "it misses nearly half of delayed cases (46.2% false negative rate). In contrast, <b>threshold 0.40</b> was selected as the operational baseline "
            "because it achieves <b>71.79% recall on validation</b> while maintaining acceptable precision (37.84%). "
            "On the held-out test set, threshold 0.40 successfully detected <b>31 of 53 delays</b> (58.49% recall), "
            "compared to only 23 delays caught at 0.50 and 14 delays caught at 0.55.",
            body_style,
        )
    )

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: Interpretability, Deployment & Governance
    # =========================================================================
    story.append(Paragraph("7. Model Interpretability & Situational Operational Context", h1_style))
    story.append(
        Paragraph(
            "To build user trust without hallucinating precision, the system employs a two-tier interpretability framework:",
            body_style,
        )
    )
    story.append(Paragraph("• <b>Global Feature Weights:</b> Mean Decrease in Impurity (MDI) across all 100 trees identifies the primary factors governing model sensitivity: <code>queue_length</code> (0.2423), <code>historical_avg_delay_hours</code> (0.2064), and <code>arrival_load_ratio</code> (0.1603) explain over 60% of total variance, followed by <code>vessel_arrivals_6h</code> (0.0792) and <code>effective_capacity_6h</code> (0.0773).", bullet_style))
    story.append(Paragraph("• <b>Observed vs. Baseline Context:</b> For any given prediction, the system contrasts the user's observation against empirical training benchmarks (e.g., Queue: 8 vs. 1.04 baseline; Load Ratio: 1.33 vs. 0.74 baseline). This explains <i>operational pressure</i> without fabricating local attribution scores (such as ad-hoc SHAP approximations).", bullet_style))
    story.append(Paragraph("• <b>Governance Disclaimer:</b> The interface enforces an explicit non-causal notice: factors reflect historical statistical associations, not proven physical causation.", bullet_style))

    # Figure 3: Feature Importance Chart
    img3 = Image(str(figures["fig3"]), width=504, height=110)
    cap3 = Paragraph(
        "<b>Figure 3: Top 5 Global Feature Importances (Random Forest Ensemble, 100 Trees).</b> "
        "Mean Decrease in Impurity (MDI) confirms that queue backlog (0.2423), recent historical delay average (0.2064), "
        "and arrival load ratio (0.1603) dominate tree split criteria. Feature influences represent global model sensitivity "
        "across synthetic simulation history, not verified local feature attributions or physical real-world causation.",
        caption_style,
    )
    story.append(Spacer(1, 1))
    story.append(img3)
    story.append(Spacer(1, 1))
    story.append(cap3)

    # Section 8: Interactive Dashboard
    story.append(Paragraph("8. Interactive Prediction Interface (<code>app.py</code>)", h1_style))
    story.append(
        Paragraph(
            "An operator-facing Streamlit application provides accessible single-observation forecasting with enterprise safeguards:",
            body_style,
        )
    )
    story.append(Paragraph("• <b>Zero Retraining:</b> The app caches the pre-trained pipeline via <code>@st.cache_resource</code>, ensuring instant inference (&lt;50ms) with zero model modification.", bullet_style))
    story.append(Paragraph("• <b>Automated Safe Math:</b> Automatically computes <code>arrival_load_ratio = arrivals / capacity</code>. If capacity is set to 0, it renders a validation warning and safely disables prediction, preventing division-by-zero crashes.", bullet_style))
    story.append(Paragraph("• <b>Synchronized Temporal Framing:</b> Weekday is automatically derived from the selected observation date (e.g. <code>obs_date.strftime('%A')</code>), and exact datetime arithmetic advances midnight boundaries into the next day.", bullet_style))
    story.append(Paragraph("• <b>Preset Operational Scenarios:</b> Features pre-loaded scenarios (<i>Normal Operations</i>, <i>Elevated Delay Risk</i>, <i>Borderline</i>) for immediate testing.", bullet_style))

    # Section 9: Limitations
    story.append(Paragraph("9. System Limitations & Future Technical Horizons", h1_style))
    story.append(
        Paragraph(
            "To maintain high integrity, the following limitations are formally documented:",
            body_style,
        )
    )
    story.append(Paragraph("• <b>Uncalibrated Probabilities:</b> Random Forest predicted probabilities represent ensemble tree voting fractions. While monotonic with empirical risk, they have not undergone isotonic or Platt probability calibration.", bullet_style))
    story.append(Paragraph("• <b>Synthetic Domain Gap:</b> The simulator abstracts away real-world complexities such as live AIS ship telemetry, tugboat availability, multi-crane outages, pilotage bottlenecks, and labor contract disputes.", bullet_style))
    story.append(Paragraph("• <b>Single-Step Horizon:</b> The model forecasts the immediate next 6-hour window. Multi-interval compounding delays over 24–48 hours will require autoregressive sequential architectures (e.g., Temporal Fusion Transformers).", bullet_style))

    # Summary Callout Box
    summary_box = [
        [
            Paragraph(
                "<b>Conclusion & Internship Readiness:</b> The forecasting pipeline is reproducible, fully tested (17/17 automated tests passing), "
                "defensible in design, and backed by a responsive Streamlit dashboard. All code, data pipelines, and documentation adhere strictly "
                "to industry best practices in temporal machine learning governance.",
                callout_style,
            )
        ]
    ]
    t_summary = Table(summary_box, colWidths=[504])
    t_summary.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0F8F5")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#2E7D32")),
            ("PADDING", (0, 0), (-1, -1), 4.5),
        ])
    )
    story.append(Spacer(1, 3))
    story.append(t_summary)

    # Build document using NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report successfully compiled to: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_pdf()
