from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.units import inch
import markdown
import pandas as pd
from bs4 import BeautifulSoup

# Register Poppins font
pdfmetrics.registerFont(TTFont('Poppins', 'Poppins-Regular.ttf'))
pdfmetrics.registerFont(TTFont('Poppins-Bold', 'Poppins-Bold.ttf'))

class ConditionalSpacer(Spacer):
    def wrap(self, availWidth, availHeight):
        if availHeight < self.height:
            self.height = availHeight
        return Spacer.wrap(self, availWidth, availHeight)

def ensure_space(story, needed_space, doc):
    # Adding a spacer to move to next page if not enough space
    story.append(ConditionalSpacer(1, needed_space))

def create_pdf(filename, queries, query_results, other_info_results, grading_results):
    # Initialize document with smaller margins for better visual appeal
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=30,
        rightMargin=30,
        topMargin=30,
        bottomMargin=30,
    )

    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        fontName='Poppins-Bold',
        fontSize=18,
        leading=22
    )
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Heading2'],
        fontName='Poppins-Bold',
        fontSize=14,
        leading=18
    )
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['BodyText'],
        fontName='Poppins',
        fontSize=10,
        leading=14
    )
    html_style = ParagraphStyle(
        "HTMLStyle", parent=body_style, fontName="Poppins", fontSize=10, leading=14
    )
    url_style = ParagraphStyle(
        "URLStyle", parent=body_style, fontName="Poppins", fontSize=10, leading=14, textColor=colors.blue
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
    title = Paragraph("Pitch Deck Report", title_style)
    story.append(title)
    story.append(Spacer(1, 12))

    # Frequently Asked Questions Section
    faq = Paragraph("Frequently Asked Questions", subtitle_style)
    story.append(faq)
    story.append(Spacer(1, 4))

    for idx in range(len(queries)):
        ensure_space(story, 40, doc)  # Ensure space before adding a new section
        query = Paragraph(queries[idx], subtitle_style)
        story.append(query)
        story.append(Spacer(1, 4))

        query_html = markdown.markdown(query_results[idx])
        query_content = Paragraph(query_html, html_style)
        story.append(query_content)
        story.append(Spacer(1, 4))

    # New page for Other Information Section
    story.append(PageBreak())

    other_info_title = Paragraph("Information related the Market ", subtitle_style)
    story.append(other_info_title)
    story.append(Spacer(1, 4))

    for key, content in other_info_results.items():
        ensure_space(story, 40, doc)  # Ensure space before adding a new section
        # Add the key as a bold title
        info_paragraph = Paragraph(f"<b>{key}</b>:", subtitle_style)
        story.append(info_paragraph)
        story.append(Spacer(1, 12))

        # Convert markdown content to HTML
        content_html = markdown.markdown(content, extensions=["tables"])
        soup = BeautifulSoup(content_html, 'html.parser')

        # Parse and add paragraphs
        for element in soup:
            if element.name == 'p':
                paragraph_content = Paragraph(str(element), html_style)
                story.append(paragraph_content)
                story.append(Spacer(1, 6))
            elif element.name == 'ul':
                for li in element.find_all('li'):
                    list_item_content = Paragraph(f'• {li.text}', html_style)
                    story.append(list_item_content)
                    story.append(Spacer(1, 6))
            elif element.name == 'table':
                # Process table
                data = []
                for row in element.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    data.append([Paragraph(cell.get_text(strip=True), html_style) for cell in cells])
                table = Table(data)
                table.setStyle(table_style)
                story.append(table)
                story.append(Spacer(1, 12))

        # Extract URLs from <a> tags and add to story
        urls = soup.find_all('a', href=True)
        if urls:
            references_title = Paragraph("References:", subtitle_style)
            story.append(references_title)
            story.append(Spacer(1, 6))

            for url in urls:
                url_paragraph = Paragraph(f"<a href='{url['href']}'>{url['href']}</a>", url_style)
                story.append(url_paragraph)

            story.append(Spacer(1, 12))

    # New page for Grading Results Section
    story.append(PageBreak())

    grading_title = Paragraph("Grading", subtitle_style)
    story.append(grading_title)
    story.append(Spacer(1, 4))

    grading_dis = Paragraph("The scoring is done on the basis of the provided information from the pitch deck and is an estimate.", html_style)
    story.append(grading_dis)
    story.append(Spacer(1, 12))

    # Convert grading results to table format
    df = {"Area/Section": [], "Score": [], "Weightage": [], "Reasoning": []}
    for datapoint in grading_results["sections"]:
        df["Area/Section"].append(datapoint["section"])
        df["Score"].append(datapoint["score"])
        df["Weightage"].append(datapoint["weight"])
        df["Reasoning"].append(datapoint["reasoning"])
    df = pd.DataFrame(df)
    table_data = [df.columns.values.tolist()] + df.values.tolist()

    for idx in range(1, len(table_data)):
        for jdx in range(len(table_data[idx])):
            table_data[idx][jdx] = Paragraph(str(table_data[idx][jdx]), style=body_style)

    table = Table(
        table_data,
        colWidths=[doc.width * 0.2, doc.width * 0.1, doc.width * 0.1, doc.width * 0.6],
    )
    table.setStyle(table_style)
    story.append(table)

    # Final Score Section
    ensure_space(story, 20, doc)  # Ensure space before adding a new section
    final_score_title = Paragraph("Final Estimated Score:", subtitle_style)
    story.append(final_score_title)
    story.append(Spacer(1, 12))

    custom_style = ParagraphStyle(
        name="BigItalicBold",
        fontSize=24,
        leading=30,
        fontName="Poppins-Bold",
        textColor=colors.red,
    )
    final_score = Paragraph(
        f"{grading_results['overall_score']}", custom_style
    )
    story.append(final_score)

    # Build PDF
    doc.build(story)
