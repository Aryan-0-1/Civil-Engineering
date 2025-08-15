import os
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from django.conf import settings
import numpy as np
from .models import RectangularNotchReading

# def extract_manual_pages():
#     input_path = os.path.join(settings.BASE_DIR, "OCF Manual.pdf")
#     output_path = os.path.join(settings.STATIC_ROOT, 'rect/static/manual_pages_25-33.pdf')
    
#     if os.path.exists(output_path):
#         return
    
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
#     reader = PdfReader(input_path)
#     writer = PdfWriter()
    
#     # Extract pages 25-33 (zero-indexed: 24-32)
#     for page_num in range(24, 33):
#         writer.add_page(reader.pages[page_num])
    
#     with open(output_path, "wb") as output_pdf:
#         writer.write(output_pdf)

def generate_results_pdf():
    # Fetch data
    readings = RectangularNotchReading.objects.all().order_by('created_at')[:7]
    H_3_2 = np.array([r.H_3_2 for r in readings])
    Q = np.array([r.Q for r in readings])
    
    # Create PDF
    pdf_path = os.path.join(settings.STATIC_ROOT, 'rect/static/results.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Rectangular Notch Experiment Results", styles['Title']))

    # Observation Table
    data = [["No.", "H (m)", "Q (m³/s)", "H³/²", "Exp. Cd", "Theo. Cd", "% Error"]]
    for i, r in enumerate(readings, 1):
        data.append([
            str(i),
            f"{r.H:.3f}",
            f"{r.Q:.5f}",
            f"{r.H_3_2:.5f}",
            f"{r.experimental_Cd:.3f}",
            "0.62",
            f"{r.percent_error:.2f}%"
        ])
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER')
    ]))
    elements.append(table)
    elements.append(Paragraph("<br/><br/>", styles['Normal']))

    # Regression Analysis
    if len(readings) >= 7:
        coeffs = np.polyfit(H_3_2, Q, 1)
        elements.append(Paragraph(f"Regression Equation: Q = {coeffs[0]:.4f}H³/² + {coeffs[1]:.4f}", styles['Heading2']))
        elements.append(Paragraph(f"Theoretical Equation: Q = {(2/3)*0.03*(2*9.81)**0.5*0.62:.4f}H³/²", styles['Heading3']))

    # Build PDF
    doc.build(elements)
    return pdf_path