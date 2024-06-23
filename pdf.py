from fpdf import FPDF
class PDF(FPDF):
    def __init__(self):
        super().__init__()

        # Add built-in fonts or already embedded TrueType fonts
        self.add_font('Poppins-Regular', '', 'Poppins-Regular.ttf', uni=True)  # Example with Arial font
        self.add_font('Poppins-Bold', '', 'Poppins-Bold.ttf', uni=True)  # Example with Arial Bold font

    def header(self):
        self.set_font('Poppins-Bold', '', 12)
        self.cell(0, 10, 'Query Responses', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Poppins-Bold', '', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Poppins-Regular', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_table(self, data, col_widths):
        self.set_font('Poppins-Regular', '', 12)
        for row in data:
            for datum, width in zip(row, col_widths):
                self.cell(width, 10, str(datum), 1)
            self.ln()

    def add_section_divider(self):
        self.set_draw_color(0, 128, 0)  # Green color for the divider
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(10)


def save_to_pdf(queries, query_results, grading_results, other_info_results, output_file):
    pdf = PDF()
    pdf.add_page()

    # Queries and Answers
    pdf.set_font('Poppins-Bold', '', 12)
    pdf.cell(0, 10, 'Queries and Answers', 0, 1, 'C')
    pdf.ln(10)

    for query, response in zip(queries, query_results):
        pdf.chapter_title(f'Query: {query}')
        pdf.chapter_body(response)
        pdf.add_section_divider()

    # Other Info
    for category, response in other_info_results.items():
        pdf.chapter_title(f'{category}')
        pdf.chapter_body(response)
        pdf.add_section_divider()

    # Grading Results
    pdf.chapter_title('Grading Results')

    table_data = [['Section', 'Grade', 'Weightage','Reasoning']]
    col_widths = [50, 30, 80, 30]  # Adjust these as per your requirement

    for section in grading_results['sections']:
        table_data.append([section['section'], section['score'], section['reasoning'], section['weight']])

    pdf.add_table(table_data, col_widths)
    pdf.add_section_divider()

    pdf.output(output_file)