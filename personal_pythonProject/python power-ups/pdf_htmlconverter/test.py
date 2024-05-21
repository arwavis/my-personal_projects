import subprocess


def pdf_to_html(pdf_file_path, html_file_path):
    subprocess.run(["pdf2htmlEX", pdf_file_path, html_file_path])


# Example usage:
pdf_to_html('input.pdf', 'output.html')
