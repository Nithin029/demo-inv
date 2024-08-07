from typing import Any, List, Tuple
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib import colors
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.colors import HexColor
from bs4 import BeautifulSoup

# Register Poppins font
pdfmetrics.registerFont(TTFont("Poppins", "Poppins-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Poppins-Bold", "Poppins-Bold.ttf"))


def header_footer(canvas, doc):
    canvas.saveState()
    styles = getSampleStyleSheet()

    # Header
    header = Paragraph(f"Report By iResearcher", styles["Italic"])
    header_2 = Paragraph(f"Elevatics", styles["Heading3"])

    w, h = header.wrap(doc.width, doc.topMargin)
    w_, h_ = header_2.wrap(doc.width, doc.topMargin)
    header.drawOn(canvas, doc.leftMargin, doc.height + doc.topMargin)
    header_2.drawOn(canvas, doc.width - doc.rightMargin, doc.height + doc.topMargin)

    canvas.setStrokeColor(colors.HexColor("#1766e6"))
    canvas.setLineWidth(2)
    canvas.line(
        doc.leftMargin,
        doc.bottomMargin - 1,
        doc.width + doc.leftMargin,
        doc.bottomMargin - 1,
    )
    canvas.line(
        doc.leftMargin,
        doc.height + doc.topMargin - 5,
        doc.width + doc.leftMargin,
        doc.height + doc.topMargin - 5,
    )

    # Footer
    footer = Paragraph(
        "LLM may generate wrong and inaccurate results please verify the information.",
        styles["Italic"],
    )
    w, h = footer.wrap(doc.width, doc.bottomMargin)
    footer.drawOn(canvas, doc.leftMargin, h)

    canvas.restoreState()


class ConditionalSpacer(Spacer):
    def wrap(self, availWidth, availHeight):
        if availHeight < self.height:
            self.height = availHeight
        return Spacer.wrap(self, availWidth, availHeight)


def ensure_space(story, needed_space, doc):
    # Adding a spacer to move to next page if not enough space
    story.append(ConditionalSpacer(1, needed_space))


def create_pdf_iresearch(buffer: Any, user_query: str, htmls: List[Tuple[str, str]]):
    # Initialize document with smaller margins for better visual appeal
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=30,
        rightMargin=30,
        topMargin=40,
        bottomMargin=30,
    )

    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontName="Poppins-Bold",
        fontSize=24,
        leading=22,
        textColor=HexColor("#141414"),
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Heading2"],
        fontName="Poppins-Bold",
        alignment=TA_JUSTIFY,
        fontSize=16,
        leading=16,
        textColor=HexColor("#141414"),
    )
    subtitle_style_small = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Heading2"],
        fontName="Poppins-Bold",
        fontSize=14,
        leading=14,
        textColor=HexColor("#141414"),
    )
    h2_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Heading2"],
        fontName="Poppins-Bold",
        fontSize=14,
        leading=14,
        textColor=HexColor("#141414"),
    )
    h3_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Heading2"],
        fontName="Poppins-Bold",
        fontSize=12,
        leading=12,
        textColor=HexColor("#141414"),
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontName="Poppins",
        alignment=1,
        fontSize=10,
        leading=14,
    )
    html_style = ParagraphStyle(
        "HTMLStyle",
        parent=body_style,
        fontName="Poppins",
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
    )
    html_style_bold = ParagraphStyle(
        "HTMLStyle",
        parent=body_style,
        fontName="Poppins-Bold",
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        textColor=HexColor("#141414"),
    )
    url_style = ParagraphStyle(
        "URLStyle",
        parent=body_style,
        fontName="Poppins",
        fontSize=8,
        leading=14,
        alignment=TA_JUSTIFY,
        textColor=colors.blue,
    )

    table_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),  # Header color
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),  # Header text color
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),  # Center align all cells
            ("FONTNAME", (0, 0), (-1, 0), "Poppins-Bold"),  # Header font
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),  # Header padding
            ("GRID", (0, 0), (-1, -1), 1, colors.black),  # Grid lines
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
    )

    # Story container
    story = []

    # Title
    title = Paragraph("iResearcher Report", title_style)
    story.append(title)
    story.append(Spacer(1, 6))
    uq = Paragraph(f"<b>Asked Query</b>: {user_query}", style=styles["Normal"])
    story.append(uq)
    story.append(Spacer(1, 16))
    for data in htmls:
        subtopic = data[0]
        html = data[1]
        soup = BeautifulSoup(html, "html.parser")

        if len(subtopic) > 0:
            sub = Paragraph(subtopic, style=subtitle_style)
            story.append(sub)
            story.append(Spacer(1, 8))
        # Parse and add paragraphs
        for element in soup:

            if element.name == "h1":
                h1 = Paragraph(str(element), style=subtitle_style)
                story.append(h1)
                story.append(Spacer(1, 6))

            if element.name == "h2":
                h1 = Paragraph(str(element), style=h2_style)
                story.append(h1)
                story.append(Spacer(1, 6))

            if element.name == "h3":
                h1 = Paragraph(str(element), style=h3_style)
                story.append(h1)
                story.append(Spacer(1, 6))

            if element.name == "p":
                if (
                    str(element).find("<strong>") >= 0
                    and len(str(element).split()) < 20
                ):
                    paragraph_content = Paragraph(str(element), html_style_bold)
                else:
                    paragraph_content = Paragraph(str(element), html_style)
                story.append(paragraph_content)
                story.append(Spacer(1, 6))

            elif element.name == "ul":
                for li in element.find_all("li"):
                    list_item_content = Paragraph(f"• {li.text}", html_style)
                    story.append(list_item_content)
                    story.append(Spacer(1, 0))
                story.append(Spacer(1, 6))
            elif element.name == "table":
                # Process table
                data = []
                for row in element.find_all("tr"):
                    cells = row.find_all(["td", "th"])
                    data.append(
                        [
                            Paragraph(cell.get_text(strip=True), body_style)
                            for cell in cells
                        ]
                    )
                table = Table(data)
                table.setStyle(table_style)
                story.append(table)
                story.append(Spacer(1, 6))

        # New page for Other Information Section
        # story.append(PageBreak())

        urls = soup.find_all("a", href=True)
        if urls:
            references_title = Paragraph("References:", subtitle_style)
            story.append(references_title)
            story.append(Spacer(1, 6))

            for url in urls:
                url_paragraph = Paragraph(
                    f"<a href='{url['href']}'>{url['href']}</a>", url_style
                )
                story.append(url_paragraph)

            # story.append(Spacer(1, 12))

        # New page for Grading Results Section
        story.append(PageBreak())
    # Build PDF
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    buffer.seek(0)
    return buffer
