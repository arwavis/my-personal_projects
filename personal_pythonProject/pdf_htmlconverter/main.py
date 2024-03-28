import pdfplumber


def pdf_to_html(pdf_file_path, html_file_path):
    with pdfplumber.open(pdf_file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

    with open(html_file_path, 'w', encoding='utf-8') as html_file:
        html_file.write("<html><body>")
        html_file.write("<pre>{}</pre>".format(text))
        html_file.write("</body></html>")


# Example usage:
pdf_to_html('/Users/aravindv/Documents/code/github/my-personal_projects/my-personal_projects/personal_pythonProject'
            '/pdf_htmlconverter/test.pdf', 'output.html')
