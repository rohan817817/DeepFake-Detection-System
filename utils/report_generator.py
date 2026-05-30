from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer)
from reportlab.lib.styles import getSampleStyleSheet

def generate_report(prediction, confidence, risk):
    pdf_path = "reports/deepfake_report.pdf"

    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph(
        "DeepFake Detection Report",
        styles["Title"]
    ))

    content.append(Spacer(1, 20))

    content.append(Paragraph(
        f"Prediction: {prediction}",
        styles["Normal"]
    ))

    doc.build(content)

    return pdf_path