📚 Core E-Book Formats & Requirements
Format
Platforms
Key Requirements
EPUB
All (Kindle, Apple, Kobo, Google)
- Validated XML structure<br>- Semantic HTML/CSS<br>- Accessible metadata (title/author)<br>- Responsive design (reflows on devices)
MOBI/AZW3
Amazon Kindle
- Converted via KindleGen<br>- Limited styling (no advanced CSS)<br>- Mandatory cover image (600×800 px min)
PDF
Print-on-demand, some retailers
- Fixed layout (300 DPI for print)<br>- Consistent margins/fonts<br>- Embedded fonts<br>- High-res cover (≥2000×3000 px)
🎯 Critical Requirements for Self-Publishers
Metadata:
ISBN (optional but recommended)
Author name, title, subtitle
Category, keywords, blurb
Publication date
Cover Design:
Dimensions: 1200×1800 px (minimum) for digital; 300 DPI for print
File Type: JPG/PNG (no transparency)
Simplicity: Avoid cluttered designs
Interior Formatting:
Fonts: Use standard serif/sans-serif (e.g., Times New Roman, Arial)
Margins: 0.5–1 inch (prevents cropping)
Chapter Headings: Consistent hierarchy (H1/H2 tags in EPUB)
Images: Optimized for screen (72–150 DPI)
💧 Watermarking in E-Books
Your watermarking feature integrates seamlessly with these formats:

EPUB/MOBI: Apply watermarks to the source text (HTML/CSS) before conversion. Use subtle positioning (e.g., footer with low opacity).
PDF: Ideal for watermarks—embed them as backgrounds or overlays (your Stegano integration shines here).
Best Practice:
Visible Watermarks: Add faint text/logos (e.g., "© [Author Name]").
Invisible Watermarks: Use Stegano to embed metadata (e.g., author ID) without affecting readability.
🚀 Workflow for Your Application
User Creates Text → Adds watermark via your tool.
Export Options:
EPUB: For broad retailer compatibility.
PDF: For print-on-demand or fixed-layout needs.
Automate Compliance:
Validate metadata/cover specs pre-export.
Offer templates matching platform guidelines (e.g., KDP’s cover template).