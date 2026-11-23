"""
AutoVision AI - Multi-Sheet Excel Workbook Generator
Compiles comprehensive dealership business reports into structured Excel (.xlsx) workbooks
featuring Executive Summary, Sales Register, Inventory Audit, ML Forecasts, and Evaluation Benchmarks.
"""

import io
import pandas as pd
from datetime import datetime
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


def generate_dealership_excel(kpis_df, sales_df, inv_df, forecast_df, metrics_df):
    """
    Generates a professionally formatted multi-sheet Excel (.xlsx) workbook buffer.
    """
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        # Write individual sheets
        kpis_df.to_excel(writer, sheet_name="Executive_Summary", index=False)
        sales_df.to_excel(writer, sheet_name="Sales_Register", index=False)
        inv_df.to_excel(writer, sheet_name="Inventory_Audit", index=False)
        forecast_df.to_excel(writer, sheet_name="ML_Forecasts", index=False)
        metrics_df.to_excel(writer, sheet_name="Model_Evaluation", index=False)

        # Style each sheet with dark header themes
        header_fill = PatternFill(start_color="131C38", end_color="131C38", fill_type="solid")
        header_font = Font(name="Segoe UI", size=11, bold=True, color="00F5D4")
        thin_border = Border(
            left=Side(style='thin', color='CBD5E1'),
            right=Side(style='thin', color='CBD5E1'),
            top=Side(style='thin', color='CBD5E1'),
            bottom=Side(style='thin', color='CBD5E1')
        )

        for sheetname in writer.sheets:
            ws = writer.sheets[sheetname]
            
            # Style header row
            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center")
                
            # Auto-adjust column widths
            for col in ws.columns:
                max_len = max(len(str(cell.value or '')) for cell in col)
                col_letter = get_column_letter(col[0].column)
                ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

    buffer.seek(0)
    return buffer.getvalue()
