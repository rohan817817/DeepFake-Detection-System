from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

from datetime import datetime

import os


def generate_report(
    prediction,
    confidence,
    risk,
    video_confidence,
    audio_confidence,
    gradcam_path="outputs/graphs/gradcam.jpg"
):

    os.makedirs(
        "reports",
        exist_ok=True
    )

    pdf_path = "reports/deepfake_report.pdf"

    doc = SimpleDocTemplate(pdf_path)

    styles = getSampleStyleSheet()

    content = []

    # Title
    content.append(
        Paragraph(
            "DeepFake Detection Report",
            styles["Title"]
        )
    )

    content.append(
        Spacer(1, 20)
    )

    # Prediction Section
    content.append(
        Paragraph(
            f"<b>Prediction:</b> {prediction}",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Confidence Score:</b> {confidence:.2f}%",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"<b>Risk Level:</b> {risk}",
            styles["Normal"]
        )
    )

    content.append(
        Spacer(1, 20)
    )

    # Multimodal Analysis
    content.append(
        Paragraph(
            "<b>Multimodal Analysis</b>",
            styles["Heading1"]
        )
    )

    content.append(
        Paragraph(
            f"Video Confidence: {video_confidence:.2f}%",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Audio Confidence: {audio_confidence:.2f}%",
            styles["Normal"]
        )
    )

    content.append(
        Paragraph(
            f"Manipulation Score: {confidence:.2f}%",
            styles["Normal"]
        )
    )

    content.append(
        Spacer(1, 20)
    )

    # GradCAM
    if os.path.exists(gradcam_path):

        content.append(
            Paragraph(
                "<b>Grad-CAM Visualization</b>",
                styles["Heading1"]
            )
        )

        content.append(
            Spacer(1, 10)
        )

        img = Image(
            gradcam_path,
            width=300,
            height=300
        )

        content.append(img)

        content.append(
            Spacer(1, 20)
        )

    # Analysis Summary
    content.append(
        Paragraph(
            "<b>Analysis Summary</b>",
            styles["Heading1"]
        )
    )

    content.append(
        Paragraph(
            """
            This media file was analyzed using a multimodal deepfake
            detection framework. The system evaluates both visual and
            audio characteristics and combines the results to estimate
            the likelihood of manipulation.
            """,
            styles["BodyText"]
        )
    )

    content.append(
        Spacer(1, 15)
    )

    # Model Information
    content.append(
        Paragraph(
            "<b>Models Used</b>",
            styles["Heading1"]
        )
    )

    content.append(
        Paragraph(
            """
            • ResNet18 + Attention Mechanism (Video Analysis)<br/>
            • MFCC Feature Extraction (Audio Analysis)<br/>
            • Multimodal Fusion Strategy<br/>
            • Grad-CAM Explainable AI Visualization
            """,
            styles["BodyText"]
        )
    )

    content.append(
        Spacer(1, 15)
    )

    # Timestamp
    content.append(
        Paragraph(
            f"Generated On: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
            styles["Normal"]
        )
    )

    doc.build(content)

    return pdf_path