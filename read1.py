import requests
from io import BytesIO
from pdfminer.high_level import extract_text

def get_text_from_pdf_link(url):
  """
  Extracts text from a PDF provided by a link.

  Args:
      url: The URL of the PDF document.

  Returns:
      A string containing all the extracted text from the PDF.
  """
  response = requests.get(url)

  if response.status_code == 200:
    # Read the PDF content into a BytesIO object
    pdf_data = BytesIO(response.content)
    # Extract text using pdfminer
    text = extract_text(pdf_data)
    return text
  else:
    raise Exception(f"Error downloading PDF: {response.status_code}")

# Example usage
pdf_url = "http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1201173"
text = get_text_from_pdf_link(pdf_url)

# Now you can use the 'text' variable which contains all the extracted text
print(text)
