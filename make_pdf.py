"""Convert a markdown file to PDF.
Usage: python make_pdf.py <input.md> <output.pdf>
"""
import re
import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa


CSS = """
@page { size: A4; margin: 1.5cm 1.5cm 1.8cm 1.5cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.45; color: #222; }
h1 { font-size: 20pt; color: #111; margin-top: 0; padding-bottom: 6px; border-bottom: 2px solid #333; }
h2 { font-size: 14pt; color: #111; margin-top: 18pt; padding-bottom: 3px; border-bottom: 1px solid #999; }
h3 { font-size: 12pt; color: #222; margin-top: 12pt; }
h4 { font-size: 11pt; color: #444; margin-top: 10pt; }
p, li { font-size: 10pt; }
code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-family: Consolas, monospace; font-size: 9pt; color: #c7254e; }
pre { background: #f4f4f4; padding: 8px; border-radius: 4px; font-family: Consolas, monospace; font-size: 8.5pt; line-height: 1.3; white-space: pre-wrap; word-wrap: break-word; }
pre code { background: none; padding: 0; color: #222; }
blockquote { border-left: 3px solid #888; padding-left: 10px; color: #555; margin-left: 0; font-style: italic; }
table { width: 100%; border-collapse: collapse; margin: 8pt 0; }
th, td { border: 1px solid #aaa; padding: 4px 6px; font-size: 9pt; text-align: left; vertical-align: top; }
th { background: #eee; }
hr { border: none; border-top: 1px solid #bbb; margin: 14pt 0; }
a { color: #1a5fb4; text-decoration: none; }
"""

HTML_SHELL = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{css}</style></head>
<body>{body}</body></html>"""


_PRE_RE = re.compile(r"<pre>(.*?)</pre>", re.DOTALL)


def _preserve_newlines_in_pre(html: str) -> str:
    """xhtml2pdf collapses whitespace inside <pre> tags. Convert newlines to
    explicit <br/> so code blocks keep their line breaks in the PDF."""
    def repl(m: re.Match) -> str:
        content = m.group(1)
        #only convert newlines outside of any nested tags
        content = content.replace("\n", "<br/>")
        return f"<pre>{content}</pre>"
    return _PRE_RE.sub(repl, html)


def convert(md_path: Path, pdf_path: Path) -> None:
    md_text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
    )
    body = _preserve_newlines_in_pre(body)
    html = HTML_SHELL.format(css=CSS, body=body)
    with open(pdf_path, "wb") as f:
        result = pisa.CreatePDF(html, dest=f, encoding="utf-8")
    if result.err:
        raise SystemExit(f"PDF generation failed with {result.err} errors")
    print(f"wrote {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python make_pdf.py <input.md> <output.pdf>")
        sys.exit(1)
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
