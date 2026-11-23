"""
AutoVision AI - PDF Report Generation Utility
Generates executive showroom analytics and research summary reports using ReportLab.
"""

import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

import config


def generate_dealership_pdf(kpis, top_models_df, forecast_df, model_metrics_df):
    """
    Generates a PDF bytes buffer containing the executive dealership report.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0B132B")
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B")
    )

    h2_style = ParagraphStyle(
        "DocH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1C2541"),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E293B")
    )

    elements = []

    # Title & Header
    elements.append(Paragraph(f"<b>{config.APP_NAME}</b> - Executive Dealership Analytics Report", title_style))
    elements.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | Dealership Intelligence & ML Sales Forecasting",
        subtitle_style
    ))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#00B4D8"), spaceBefore=5, spaceAfter=15))

    # 1. Executive KPIs Summary Table
    elements.append(Paragraph("1. Executive Dealership KPIs", h2_style))
    kpi_data = [
        ["Total Vehicles Sold", str(kpis.get("total_sold", "N/A")), "Available Inventory", str(kpis.get("available_stock", "N/A"))],
        ["Total Sales Revenue", str(kpis.get("total_revenue", "N/A")), "Active Bookings", str(kpis.get("active_bookings", "N/A"))],
        ["Estimated Gross Profit", str(kpis.get("gross_profit", "N/A")), "Enquiry Conversion Rate", str(kpis.get("conversion_rate", "N/A"))]
    ]
    kpi_table = Table(kpi_data, colWidths=[2.2 * inch, 1.8 * inch, 2.0 * inch, 1.5 * inch])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0F172A")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(kpi_table)
    elements.append(Spacer(1, 12))

    # 2. Top-Selling Vehicle Models
    elements.append(Paragraph("2. Top-Selling Vehicle Models", h2_style))
    if top_models_df is not None and not top_models_df.empty:
        top_data = [["Rank", "Brand", "Model", "Fuel Type", "Units Sold", "Total Revenue"]]
        for idx, row in top_models_df.head(5).iterrows():
            top_data.append([
                str(idx + 1),
                str(row.get("brand", "")),
                str(row.get("model", "")),
                str(row.get("fuel_type", "")),
                str(row.get("units_sold", "")),
                str(row.get("formatted_revenue", ""))
            ])
        top_table = Table(top_data, colWidths=[0.6 * inch, 1.3 * inch, 1.6 * inch, 1.2 * inch, 1.0 * inch, 1.8 * inch])
        top_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(top_table)
    elements.append(Spacer(1, 12))

    # 3. Machine Learning Forecasting Horizon
    elements.append(Paragraph("3. ML Sales Forecast Horizon (Next 6 Months)", h2_style))
    if forecast_df is not None and not forecast_df.empty:
        fc_data = [["Forecast Period", "Predicted Units", "Lower Bound (95%)", "Upper Bound (95%)"]]
        for _, r in forecast_df.iterrows():
            fc_data.append([
                str(r.get("month_label", "")),
                f"{r.get('predicted_units', 0):.1f} units",
                f"{r.get('lower_bound', 0):.1f} units",
                f"{r.get('upper_bound', 0):.1f} units"
            ])
        fc_table = Table(fc_data, colWidths=[2.0 * inch, 1.8 * inch, 1.8 * inch, 1.9 * inch])
        fc_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284C7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F9FF")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BAE6FD")),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(fc_table)
    elements.append(Spacer(1, 12))

    # 4. Research Model Evaluation Benchmark
    elements.append(Paragraph("4. Research Machine Learning Model Evaluation", h2_style))
    if model_metrics_df is not None and not model_metrics_df.empty:
        eval_data = [["Algorithm", "MAE", "RMSE", "R² Score", "MAPE (%)"]]
        for _, r in model_metrics_df.iterrows():
            eval_data.append([
                str(r.get("Model", "")),
                f"{r.get('MAE', 0):.2f}",
                f"{r.get('RMSE', 0):.2f}",
                f"{r.get('R2_Score', 0):.4f}",
                f"{r.get('MAPE (%)', 0):.2f}%"
            ])
        eval_table = Table(eval_data, colWidths=[2.5 * inch, 1.2 * inch, 1.2 * inch, 1.3 * inch, 1.3 * inch])
        eval_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(eval_table)

    # Footer note
    elements.append(Spacer(1, 18))
    elements.append(Paragraph(
        "<b>Note:</b> This report was generated autonomously by AutoVision AI. Evaluation metrics reflect actual model validation scores on chronologically partitioned test sets.",
        body_style
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
