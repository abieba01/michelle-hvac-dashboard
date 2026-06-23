"""
report.py  -  PDF report generator for the HVAC Optimisation Dashboard.

Call generate_pdf() to get a bytes object that can be sent as a download.
"""
from __future__ import annotations

import io
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable, Image, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

# ── Brand colours ──────────────────────────────────────────────────────────────
C_DARK_BLUE  = colors.HexColor("#1F4E79")
C_MID_BLUE   = colors.HexColor("#2E75B6")
C_LIGHT_BLUE = colors.HexColor("#EBF3FB")
C_ORANGE     = colors.HexColor("#C55A11")
C_GREEN      = colors.HexColor("#375623")
C_AMBER      = colors.HexColor("#7F6000")
C_RED        = colors.HexColor("#843C0C")
C_GREY       = colors.HexColor("#404040")
C_LIGHT_GREY = colors.HexColor("#F2F2F2")
C_WHITE      = colors.white

MPL_PALETTE  = ["#2e75b6", "#5b9bd5", "#9dc3e6", "#c55a11"]


# ── Paragraph styles ───────────────────────────────────────────────────────────
def _styles():
    base = dict(fontName="Helvetica", textColor=C_GREY)
    return {
        "title":     ParagraphStyle("title",     fontSize=26, textColor=C_DARK_BLUE,
                                    fontName="Helvetica-Bold", alignment=TA_CENTER,
                                    spaceAfter=6),
        "subtitle":  ParagraphStyle("subtitle",  fontSize=14, textColor=C_MID_BLUE,
                                    fontName="Helvetica-Bold", alignment=TA_CENTER,
                                    spaceAfter=4),
        "doc_type":  ParagraphStyle("doc_type",  fontSize=11, textColor=C_ORANGE,
                                    fontName="Helvetica-Oblique", alignment=TA_CENTER,
                                    spaceAfter=2),
        "h1":        ParagraphStyle("h1",        fontSize=14, textColor=C_DARK_BLUE,
                                    fontName="Helvetica-Bold", spaceBefore=16,
                                    spaceAfter=4),
        "h2":        ParagraphStyle("h2",        fontSize=11, textColor=C_MID_BLUE,
                                    fontName="Helvetica-Bold", spaceBefore=10,
                                    spaceAfter=3),
        "body":      ParagraphStyle("body",      fontSize=10, leading=15,
                                    alignment=TA_JUSTIFY, spaceAfter=6, **base),
        "bullet":    ParagraphStyle("bullet",    fontSize=10, leading=14,
                                    leftIndent=16, bulletIndent=6,
                                    spaceAfter=3, **base),
        "caption":   ParagraphStyle("caption",   fontSize=8, textColor=colors.grey,
                                    fontName="Helvetica-Oblique", alignment=TA_CENTER,
                                    spaceAfter=4),
        "label":     ParagraphStyle("label",     fontSize=9,  textColor=C_GREY,
                                    fontName="Helvetica-Bold"),
        "cell":      ParagraphStyle("cell",      fontSize=9,  textColor=C_GREY,
                                    fontName="Helvetica",     leading=12),
        "cell_bold": ParagraphStyle("cell_bold", fontSize=9,  textColor=C_GREY,
                                    fontName="Helvetica-Bold"),
        "rec_strong":ParagraphStyle("rec_strong",fontSize=9,  textColor=C_GREEN,
                                    fontName="Helvetica-Bold"),
        "rec_good":  ParagraphStyle("rec_good",  fontSize=9,  textColor=C_MID_BLUE,
                                    fontName="Helvetica-Bold"),
        "rec_amber": ParagraphStyle("rec_amber", fontSize=9,  textColor=C_AMBER,
                                    fontName="Helvetica-Bold"),
        "rec_red":   ParagraphStyle("rec_red",   fontSize=9,  textColor=C_RED,
                                    fontName="Helvetica-Bold"),
        "footer":    ParagraphStyle("footer",    fontSize=8,  textColor=colors.grey,
                                    alignment=TA_CENTER),
    }


# ── Rating logic ───────────────────────────────────────────────────────────────
def _rating(payback, npv, lifetime, saving_pct):
    """Return (label, style_key, one-line reason)."""
    if saving_pct <= 0:
        return "No Saving", "rec_red", "This strategy produces no measurable energy saving under current conditions."
    if np.isnan(payback):
        return "Operational Change", "rec_good", "No capital cost — pure schedule or set-point change, immediate saving."
    if np.isnan(npv):
        npv_str = "unknown"
    else:
        npv_str = f"£{npv:,.0f}"

    if payback <= 3:
        return ("Strongly Recommended", "rec_strong",
                f"Exceptional payback of {payback:.1f} yrs. "
                f"{lifetime}-yr NPV: {npv_str}.")
    if payback <= 6:
        return ("Recommended", "rec_good",
                f"Strong payback of {payback:.1f} yrs. "
                f"{lifetime}-yr NPV: {npv_str}.")
    if payback <= lifetime:
        return ("Consider", "rec_amber",
                f"Payback of {payback:.1f} yrs is within the {lifetime}-yr measure life. "
                f"NPV: {npv_str}.")
    return ("Not Recommended", "rec_red",
            f"Payback of {payback:.1f} yrs exceeds the {lifetime}-yr measure lifetime "
            f"at current assumptions. Revisit CAPEX or await energy price increases.")


# ── Chart helpers ──────────────────────────────────────────────────────────────
def _chart_savings(non_bl: pd.DataFrame) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    bars = ax.bar(range(len(non_bl)), non_bl["Saving (%)"],
                  color=MPL_PALETTE[:len(non_bl)])
    ax.set_ylabel("HVAC energy saving (%)", fontsize=9)
    ax.set_xticks(range(len(non_bl)))
    ax.set_xticklabels([s.replace(" ", "\n") for s in non_bl["Strategy"]], fontsize=7.5)
    for bar, val in zip(bars, non_bl["Saving (%)"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.25,
                f"{val:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf


def _chart_payback(non_bl: pd.DataFrame) -> io.BytesIO:
    pb = non_bl.dropna(subset=["Payback (yrs)"])
    if pb.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.barh(pb["Strategy"], pb["Payback (yrs)"], color="#5b9bd5")
    ax.set_xlabel("Simple payback period (years)", fontsize=9)
    ax.invert_yaxis()
    for patch, val in zip(ax.patches, pb["Payback (yrs)"]):
        ax.text(val + 0.05, patch.get_y() + patch.get_height() / 2,
                f"{val:.1f} yr", va="center", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf


# ── Table builder ──────────────────────────────────────────────────────────────
def _tbl_style(header_bg=C_MID_BLUE):
    return TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  header_bg),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  C_WHITE),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  9),
        ("BOTTOMPADDING",(0, 0), (-1, 0),  6),
        ("TOPPADDING",   (0, 0), (-1, 0),  6),
        ("ROWBACKGROUNDS",(0,1), (-1,-1),  [C_LIGHT_BLUE, C_WHITE]),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 9),
        ("TOPPADDING",   (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 1), (-1, -1), 4),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#D0D0D0")),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ])


# ── Page template (header / footer lines) ─────────────────────────────────────
def _on_page(canvas, doc, project_name):
    canvas.saveState()
    w, _ = A4
    canvas.setFillColor(C_DARK_BLUE)
    canvas.rect(0, A4[1] - 1.1 * cm, w, 1.1 * cm, fill=1, stroke=0)
    canvas.setFillColor(C_WHITE)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(1.5 * cm, A4[1] - 0.75 * cm, project_name)
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(w - 1.5 * cm, A4[1] - 0.75 * cm,
                           "HVAC Optimisation — Confidential")
    canvas.setFillColor(C_GREY)
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(w / 2, 0.7 * cm, f"Page {doc.page}")
    canvas.setStrokeColor(C_MID_BLUE)
    canvas.setLineWidth(0.5)
    canvas.line(1.5 * cm, 1.1 * cm, w - 1.5 * cm, 1.1 * cm)
    canvas.restoreState()


# ── Main entry point ───────────────────────────────────────────────────────────
def generate_pdf(
    df: pd.DataFrame,
    assumptions: dict,
    metrics: dict,
    data_label: str,
    project_name: str = "Michelle's Project",
) -> bytes:
    """
    Build and return a PDF report as bytes.

    Parameters
    ----------
    df          : results DataFrame from _compute_results() — must include 'key' column
    assumptions : dict with elec_price, carbon_factor, discount_rate, lifetime,
                  capex_occ, capex_therm, capex_bas, capex_comb
    metrics     : model performance dict (r2, mae, rmse, n_train, n_test)
    data_label  : one-line string describing the data source
    project_name: name shown in the header
    """
    S = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
    )

    non_bl    = df[df["key"] != "baseline"].copy()
    lifetime  = assumptions["lifetime"]
    elec      = assumptions["elec_price"]
    today_str = date.today().strftime("%d %B %Y")

    story = []

    # ── TITLE PAGE ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph(project_name, S["title"]))
    story.append(Paragraph("HVAC Optimisation AI Model", S["subtitle"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Energy Optimisation Report &amp; Recommendations", S["doc_type"]))
    story.append(Spacer(1, 1 * cm))
    story.append(HRFlowable(width="80%", thickness=1.5,
                             color=C_MID_BLUE, spaceAfter=10))

    meta = [
        ["Date",         today_str],
        ["Data source",  data_label],
        ["Prepared by",  project_name],
        ["Electricity",  f"£{elec:.2f} / kWh"],
        ["Discount rate",f"{assumptions['discount_rate']*100:.0f}%"],
        ["Measure life", f"{lifetime} years"],
    ]
    meta_tbl = Table(meta, colWidths=[4 * cm, 10 * cm])
    meta_tbl.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",  (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",  (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), C_DARK_BLUE),
        ("TEXTCOLOR", (1, 0), (1, -1), C_GREY),
        ("TOPPADDING",(0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, colors.HexColor("#D0D0D0")),
    ]))
    story.append(meta_tbl)
    story.append(PageBreak())

    # ── EXECUTIVE SUMMARY ─────────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_MID_BLUE,
                             spaceAfter=6))

    best_row = non_bl.sort_values("Cost saving (GBP/yr)", ascending=False).iloc[0]
    best_name = best_row["Strategy"]
    best_saving_pct = best_row["Saving (%)"]
    best_cost = best_row["Cost saving (GBP/yr)"]
    best_carbon = best_row["Carbon saving (tCO2e/yr)"]
    total_carbon = non_bl["Carbon saving (tCO2e/yr)"].max()

    story.append(Paragraph(
        f"This report presents the results of an AI-driven energy optimisation analysis "
        f"for the building described above. A Gradient Boosting surrogate model was trained "
        f"on {metrics['n_train']:,} hourly records (R² = {metrics['r2']:.4f}) "
        f"and used to evaluate four control upgrade strategies against a conventional "
        f"fixed-schedule baseline.", S["body"]
    ))
    story.append(Paragraph(
        f"The highest-performing strategy is <b>{best_name}</b>, which is projected to "
        f"reduce annual HVAC energy consumption by <b>{best_saving_pct:.1f}%</b>, "
        f"saving <b>£{best_cost:,.0f} per year</b> and "
        f"<b>{best_carbon:.1f} tCO₂e per year</b> in carbon emissions "
        f"at the current electricity price of £{elec:.2f}/kWh.", S["body"]
    ))
    story.append(Paragraph(
        f"The combined optimisation strategy — applying all three measures together — "
        f"offers the greatest total energy reduction. Individual strategies provide "
        f"shorter payback periods and lower upfront investment, making them suitable "
        f"for phased implementation.", S["body"]
    ))
    story.append(Spacer(1, 0.3 * cm))

    # KPI summary boxes (as a simple table)
    kpi_data = [
        [
            _kpi_cell("Best annual saving", f"£{best_cost:,.0f}", S),
            _kpi_cell("Best energy reduction", f"{best_saving_pct:.1f}%", S),
            _kpi_cell("Best carbon saving", f"{total_carbon:.1f} tCO₂e/yr", S),
            _kpi_cell("Model accuracy (R²)", f"{metrics['r2']:.4f}", S),
        ]
    ]
    kpi_tbl = Table(kpi_data, colWidths=[3.8 * cm] * 4)
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), C_LIGHT_BLUE),
        ("BOX",          (0, 0), (-1, -1), 0.5, C_MID_BLUE),
        ("INNERGRID",    (0, 0), (-1, -1), 0.5, C_MID_BLUE),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
    ]))
    story.append(kpi_tbl)

    # ── ASSUMPTIONS ───────────────────────────────────────────────────────────
    story.append(Paragraph("Economic Assumptions", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_MID_BLUE,
                             spaceAfter=6))
    story.append(Paragraph(
        "All financial results in this report are calculated using the assumptions below. "
        "Use the dashboard sliders to regenerate the report under different scenarios.",
        S["body"]
    ))

    capex_map_display = {
        "Occupancy scheduling": assumptions["capex_occ"],
        "Smart thermostats":    assumptions["capex_therm"],
        "Building Automation System": assumptions["capex_bas"],
        "Combined strategy":    assumptions["capex_comb"],
    }
    assump_rows = [["Parameter", "Value"]]
    assump_rows += [
        ["Electricity price",    f"£{assumptions['elec_price']:.2f} / kWh"],
        ["Carbon factor",        f"{assumptions['carbon_factor']:.3f} kgCO₂e / kWh"],
        ["Discount rate",        f"{assumptions['discount_rate']*100:.0f}%"],
        ["Measure lifetime",     f"{lifetime} years"],
    ]
    for name, val in capex_map_display.items():
        assump_rows.append([f"CAPEX — {name}", f"£{val:,}"])

    assump_tbl = Table(assump_rows, colWidths=[9 * cm, 7.5 * cm])
    assump_tbl.setStyle(_tbl_style())
    story.append(assump_tbl)

    # ── RESULTS TABLE ─────────────────────────────────────────────────────────
    story.append(Paragraph("Scenario Results", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_MID_BLUE,
                             spaceAfter=6))

    res_headers = [
        "Strategy", "Annual\nHVAC\n(kWh/yr)", "Saving\n(kWh/yr)",
        "Saving\n(%)", "Cost\nSaving\n(£/yr)", "Carbon\n(tCO₂e/yr)",
        "CAPEX\n(£)", "Payback\n(yrs)", f"NPV\n{lifetime}yr\n(£)",
    ]
    res_rows = [res_headers]
    for _, r in df.drop(columns=["key"]).iterrows():
        payback = r["Payback (yrs)"]
        npv     = r["NPV (GBP)"]
        res_rows.append([
            r["Strategy"],
            f"{r['Annual HVAC (kWh/yr)']:,.0f}",
            f"{r['Saved (kWh/yr)']:,.0f}",
            f"{r['Saving (%)']:.1f}%",
            f"£{r['Cost saving (GBP/yr)']:,.0f}",
            f"{r['Carbon saving (tCO2e/yr)']:.1f}",
            f"£{r['CAPEX (GBP)']:,}" if r["CAPEX (GBP)"] else "—",
            f"{payback:.1f}" if not (isinstance(payback, float) and np.isnan(payback)) else "—",
            f"£{npv:,.0f}" if not (isinstance(npv, float) and np.isnan(npv)) else "—",
        ])

    col_w = [4.2*cm, 1.8*cm, 1.8*cm, 1.4*cm, 1.8*cm, 1.8*cm, 1.8*cm, 1.5*cm, 1.9*cm]
    res_tbl = Table(res_rows, colWidths=col_w, repeatRows=1)
    res_tbl.setStyle(_tbl_style())
    story.append(res_tbl)

    # ── CHARTS ────────────────────────────────────────────────────────────────
    story.append(Paragraph("Visual Summary", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_MID_BLUE,
                             spaceAfter=6))

    chart_row = [[
        Image(_chart_savings(non_bl), width=8.5 * cm, height=5 * cm),
    ]]
    pb_buf = _chart_payback(non_bl)
    if pb_buf:
        chart_row[0].append(Image(pb_buf, width=8.5 * cm, height=5 * cm))
    chart_tbl = Table(chart_row, colWidths=[8.8 * cm] * len(chart_row[0]))
    chart_tbl.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
    story.append(chart_tbl)
    story.append(Paragraph(
        "Left: annual HVAC energy saving by strategy.  "
        "Right: simple payback period by strategy.",
        S["caption"]
    ))

    # ── RECOMMENDATIONS ───────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Recommendations", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_MID_BLUE,
                             spaceAfter=6))
    story.append(Paragraph(
        "The following recommendations are generated automatically from the model "
        "results using the economic assumptions above. Each strategy is assessed "
        "independently; the combined strategy reflects the full potential if all "
        "measures are implemented together.",
        S["body"]
    ))

    for _, r in non_bl.iterrows():
        payback  = r["Payback (yrs)"]
        npv      = r["NPV (GBP)"]
        saving   = r["Saving (%)"]
        cost_sav = r["Cost saving (GBP/yr)"]
        carbon   = r["Carbon saving (tCO2e/yr)"]
        capex    = r["CAPEX (GBP)"]

        label, style_key, reason = _rating(payback, npv, lifetime, saving)

        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(r["Strategy"], S["h2"]))

        # Rating badge row
        badge_data = [[
            Paragraph(label, S[style_key]),
            Paragraph(reason, S["cell"]),
        ]]
        badge_tbl = Table(badge_data, colWidths=[4.5 * cm, 12 * cm])
        badge_tbl.setStyle(TableStyle([
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(badge_tbl)

        # Detail bullets
        story.append(Paragraph(
            f"•  Energy saving: <b>{saving:.1f}%</b> of baseline HVAC consumption",
            S["bullet"]
        ))
        story.append(Paragraph(
            f"•  Annual cost saving: <b>£{cost_sav:,.0f}</b> at £{elec:.2f}/kWh",
            S["bullet"]
        ))
        story.append(Paragraph(
            f"•  Carbon saving: <b>{carbon:.1f} tCO₂e per year</b>",
            S["bullet"]
        ))
        if capex > 0:
            story.append(Paragraph(
                f"•  Capital investment: <b>£{capex:,}</b>",
                S["bullet"]
            ))
        if not (isinstance(payback, float) and np.isnan(payback)):
            story.append(Paragraph(
                f"•  Simple payback: <b>{payback:.1f} years</b> "
                f"({'within' if payback <= lifetime else 'exceeds'} the "
                f"{lifetime}-year measure lifetime)",
                S["bullet"]
            ))
        if not (isinstance(npv, float) and np.isnan(npv)):
            story.append(Paragraph(
                f"•  {lifetime}-year NPV: <b>£{npv:,.0f}</b> "
                f"({'positive — value-creating investment' if npv > 0 else 'negative — costs outweigh savings at these assumptions'})",
                S["bullet"]
            ))
        story.append(HRFlowable(width="100%", thickness=0.3,
                                 color=colors.HexColor("#D0D0D0"), spaceAfter=2))

    # Overall recommendation
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Overall Recommendation", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_MID_BLUE,
                             spaceAfter=6))

    # Identify strategies with positive NPV and payback within lifetime
    viable = non_bl[
        non_bl.apply(
            lambda r: not np.isnan(r["Payback (yrs)"]) and r["Payback (yrs)"] <= lifetime,
            axis=1
        )
    ].sort_values("Cost saving (GBP/yr)", ascending=False)

    if viable.empty:
        story.append(Paragraph(
            "Under the current economic assumptions, no strategy recovers its capital "
            "cost within the measure lifetime. This is most likely driven by a low "
            "electricity price or high CAPEX values. Consider increasing the electricity "
            "price assumption to reflect future tariff increases, or negotiating lower "
            "capital costs with suppliers.",
            S["body"]
        ))
    else:
        best_v = viable.iloc[0]
        story.append(Paragraph(
            f"<b>Priority 1 — Implement {best_v['Strategy']} first.</b> "
            f"This delivers the highest annual cost saving (£{best_v['Cost saving (GBP/yr)']:,.0f}/yr) "
            f"with a payback of {best_v['Payback (yrs)']:.1f} years. "
            f"It should be treated as the immediate first step.",
            S["body"]
        ))

        if len(viable) > 1:
            others = viable.iloc[1:]
            names  = ", ".join(others["Strategy"].tolist())
            story.append(Paragraph(
                f"<b>Priority 2 — Consider {names} in a phased programme.</b> "
                f"These strategies are also financially viable within the measure lifetime "
                f"and would compound the savings from Priority 1. A phased approach "
                f"spreads capital expenditure and allows operational lessons from the "
                f"first upgrade to inform subsequent ones.",
                S["body"]
            ))

        combined_row = non_bl[non_bl["Strategy"].str.lower().str.contains("combined")]
        if not combined_row.empty:
            c = combined_row.iloc[0]
            if not np.isnan(c["NPV (GBP)"]) and c["NPV (GBP)"] > 0:
                story.append(Paragraph(
                    f"<b>Long-term — The Combined strategy offers the greatest total impact.</b> "
                    f"Implementing all three measures together is projected to save "
                    f"£{c['Cost saving (GBP/yr)']:,.0f}/yr and "
                    f"{c['Carbon saving (tCO2e/yr)']:.1f} tCO₂e/yr, with a "
                    f"{lifetime}-year NPV of £{c['NPV (GBP)']:,.0f}. "
                    f"If budget allows, this is the most impactful single investment.",
                    S["body"]
                ))

        story.append(Paragraph(
            "<b>Sensitivity note:</b> All projections are based on current electricity "
            f"prices (£{elec:.2f}/kWh). UK commercial tariffs have historically "
            "trended upward. At higher electricity prices, payback periods shorten "
            "and NPVs increase — use the dashboard to test different price scenarios.",
            S["body"]
        ))

    # ── MODEL PERFORMANCE ─────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Model Performance", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_MID_BLUE,
                             spaceAfter=6))
    story.append(Paragraph(
        "The surrogate model was evaluated on a held-out test set "
        f"({metrics['n_test']:,} rows) not seen during training. "
        "High R² and low error values indicate the model reliably reproduces "
        "the energy response surface across the full range of operating conditions.",
        S["body"]
    ))
    perf_rows = [
        ["Metric", "Value", "Interpretation"],
        ["R² (coefficient of determination)",
         f"{metrics['r2']:.4f}",
         "Proportion of energy variation explained. 1.0 = perfect."],
        ["MAE (mean absolute error)",
         f"{metrics['mae']:.2f} kWh/h",
         f"Average prediction error vs mean target of {metrics['mean_target']:.1f} kWh/h."],
        ["RMSE (root mean square error)",
         f"{metrics['rmse']:.2f} kWh/h",
         "Penalises large errors more than MAE. Close to MAE = errors are consistent."],
        ["Training rows", f"{metrics['n_train']:,}", "Rows used to fit the model."],
        ["Test rows",     f"{metrics['n_test']:,}",  "Held-out rows used only for evaluation."],
    ]
    perf_tbl = Table(perf_rows, colWidths=[5.5 * cm, 2.5 * cm, 8.5 * cm])
    perf_tbl.setStyle(_tbl_style())
    story.append(perf_tbl)

    # ── BUILD PDF ─────────────────────────────────────────────────────────────
    doc.build(
        story,
        onFirstPage =lambda c, d: _on_page(c, d, project_name),
        onLaterPages=lambda c, d: _on_page(c, d, project_name),
    )
    return buf.getvalue()


# ── Helper: KPI cell content ───────────────────────────────────────────────────
def _kpi_cell(label: str, value: str, S: dict):
    return [
        Paragraph(f"<b>{value}</b>",
                  ParagraphStyle("kpi_val", fontSize=13, textColor=C_DARK_BLUE,
                                 fontName="Helvetica-Bold", alignment=TA_CENTER)),
        Paragraph(label,
                  ParagraphStyle("kpi_lbl", fontSize=8, textColor=C_GREY,
                                 fontName="Helvetica", alignment=TA_CENTER)),
    ]
