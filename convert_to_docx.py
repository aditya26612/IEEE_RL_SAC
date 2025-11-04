from python_docx import Document
from python_docx.shared import Inches
import markdown
import os

def md_to_docx(md_file, docx_file):
    # Read markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Create document with a title
    doc = Document()
    
    # Add title
    doc.add_heading('Comparison: Q-Learning vs SAC', 0)
    
    # Add subtitle
    doc.add_heading('Multi-Microgrid Energy Management', 1)
    
    # Convert markdown to HTML (for basic formatting)
    html = markdown.markdown(md_content)
    
    # Split into sections and add to document
    sections = html.split('\n\n')
    for section in sections:
        if section.startswith('&lt;h'):  # Handle headers
            level = int(section[3])
            text = section.split('>')[-2].split('<')[0]
            doc.add_heading(text, level)
        else:
            # Add regular paragraph
            doc.add_paragraph(section)
            
    # Check for images in artifacts/Comparison
    img_dir = 'artifacts/Comparison'
    if os.path.exists(img_dir):
        for img in os.listdir(img_dir):
            if img.endswith('.png'):
                doc.add_picture(os.path.join(img_dir, img), width=Inches(6))
                # Add caption
                caption = img.replace('.png', '').replace('_', ' ').title()
                doc.add_paragraph(f'Figure: {caption}', style='Caption')
    
    # Save the document
    doc.save(docx_file)

if __name__ == '__main__':
    md_to_docx('comparison_Qlearning_vs_SAC.md', 'comparison_Qlearning_vs_SAC.docx')