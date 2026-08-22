from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from io import BytesIO
from datetime import datetime 


def create_pdf_report(
    user_name,
    prediction,
    confidence,
    risk,
    probabilities=None
):
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    # ================= STYLES =================

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=28,
        textColor=colors.HexColor("#2563EB"),
        alignment=TA_CENTER,
        spaceAfter=25
    )

    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        fontSize=12,
        textColor=colors.HexColor("#64748B"),
        alignment=TA_CENTER,
        spaceAfter=20
    )

    section_style = ParagraphStyle(
        "SectionStyle",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#2563EB"),
        spaceAfter=10
    )

    info_style = ParagraphStyle(
        "InfoStyle",
        parent=styles["Normal"],
        fontSize=14,
        leading=22
    )

    content = []

    # ================= HEADER =================

    content.append(
        Paragraph(
            "XRAY AI REPORT",
            title_style
        )
    )

    content.append(
        Paragraph(
            "Deep Learning Chest Disease Detection System",
            subtitle_style
        )
    )

    content.append(Spacer(1, 20))

    # ================= REPORT INFO =================

    report_id = f"XR-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    generated_time = datetime.now().strftime("%d %b %Y, %I:%M %p")

    content.append(
        Paragraph(
            f"<b>Report ID:</b> {report_id}",
            info_style
        )
    )

    content.append(
        Paragraph(
            f"<b>Generated:</b> {generated_time}",
            info_style
        )
    )

    content.append(Spacer(1, 20))

    # ================= PATIENT INFO =================

    content.append(
        Paragraph(
            "<b>Patient Information</b>",
            section_style
        )
    )

    content.append(
        Paragraph(
            f"<b>Patient Name:</b> {user_name}",
            info_style
        )
    )

    content.append(Spacer(1, 10))

    # ================= PREDICTION =================

    if prediction.upper() == "NORMAL":
        prediction_color = "#16A34A"
    elif prediction.upper() == "PNEUMONIA":
        prediction_color = "#DC2626"
    elif prediction.upper() == "COVID":
        prediction_color = "#EA580C"
    else:
        prediction_color = "#2563EB"

    content.append(
        Paragraph(
            f"""
            <font color="{prediction_color}">
            <b>Prediction:</b> {prediction}
            </font>
            """,
            info_style
        )
    )

    # ================= CONFIDENCE =================

    if confidence >= 85:
        confidence_color = "#16A34A"
    elif confidence >= 60:
        confidence_color = "#EAB308"
    else:
        confidence_color = "#DC2626"

    content.append(
        Paragraph(
            f"""
            <font color="{confidence_color}">
            <b>Confidence Score:</b> {confidence:.2f}%
            </font>
            """,
            info_style
        )
    )

    # ================= RISK LEVEL =================

    if risk.upper() == "LOW":
        risk_color = "#16A34A"
    elif risk.upper() == "MEDIUM":
        risk_color = "#EAB308"
    elif risk.upper() == "HIGH":
        risk_color = "#DC2626"
    else:
        risk_color = "#2563EB"

    content.append(
        Paragraph(
            f"""
            <font color="{risk_color}">
            <b>Risk Level:</b> {risk}
            </font>
            """,
            info_style
        )
    )

    content.append(Spacer(1, 25))

    # ================= AI PROBABILITY GRAPH =================

    content.append(
        Paragraph(
            "<b>AI Probability Analysis</b>",
            section_style
        )
    )

    drawing = Drawing(400, 250)

    bar_chart = VerticalBarChart()
    bar_chart.x = 50
    bar_chart.y = 50
    bar_chart.height = 150
    bar_chart.width = 300

    # Use actual model probabilities
    if probabilities:
        covid_prob = probabilities.get("COVID", 0)
        pneumonia_prob = probabilities.get("PNEUMONIA", 0)
        normal_prob = probabilities.get("NORMAL", 0)
    else:
        covid_prob = 0
        pneumonia_prob = 0
        normal_prob = 0

    bar_chart.data = [[
        covid_prob,
        pneumonia_prob,
        normal_prob
    ]]

    bar_chart.categoryAxis.categoryNames = [
        "COVID",
        "PNEUMONIA",
        "NORMAL"
    ]

    bar_chart.valueAxis.valueMin = 0
    bar_chart.valueAxis.valueMax = 100
    bar_chart.valueAxis.valueStep = 20

    # Labels above bars
    bar_chart.barLabelFormat = '%0.0f%%'
    bar_chart.barLabels.nudge = 10
    bar_chart.barLabels.fontName = 'Helvetica-Bold'
    bar_chart.barLabels.fontSize = 10
    bar_chart.barLabels.fillColor = colors.black
    bar_chart.barLabels.boxAnchor = 's'
    bar_chart.barLabels.dx = 0
    bar_chart.barLabels.dy = 5
    bar_chart.barLabels.visible = True

    # Bar colors
    bar_chart.bars[0].fillColor = colors.HexColor("#EF4444")  # COVID
    bar_chart.bars[1].fillColor = colors.HexColor("#F59E0B")  # Pneumonia
    bar_chart.bars[2].fillColor = colors.HexColor("#22C55E")  # Normal

    drawing.add(bar_chart)

    content.append(Spacer(1, 10))
    content.append(drawing)
    content.append(Spacer(1, 20))

    # ================= DISCLAIMER =================

    content.append(
        Paragraph(
            "This report was generated automatically by XrayAI using a deep learning model and is intended for educational purposes only. This report should not be considered a medical diagnosis or medical advice.",
            styles["Normal"]
        )
    )

    # ================= BUILD PDF =================

    doc.build(content)

    buffer.seek(0)

    return buffer