import PyPDF2
import requests
from io import BytesIO

pdf_url = "https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1239453"

# Fetch the PDF content from the URL
response = requests.get(pdf_url)
pdf_content = BytesIO(response.content)

# Read the PDF content
reader = PyPDF2.PdfReader(pdf_content)

# Extract text from each page and combine it into a single string
text = ""
for page_num in range(reader.numPages):
    page = reader.getPage(page_num)
    text += page.extract_text()

print(text)
