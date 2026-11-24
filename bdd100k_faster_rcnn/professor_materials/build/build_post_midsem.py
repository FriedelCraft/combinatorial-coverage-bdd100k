"""Generate a Beamer source, speaker notes and a matching presentation PDF."""

import html
import io
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out"
STEM = "BDD100K_Post_Midsem_Presentation"
SLIDES = json.loads((HERE / "post_midsem_slides.json").read_text(encoding="utf-8"))
PALETTE = {"ink": "16273C", "text": "24394A", "accent": "126B65", "muted": "4E6473",
           "paper": "FCFDFB", "line": "CFDCD9", "tint": "E5F2ED"}
WIDTHS = {
    2: [.48, .19, .33], 4: [.23, .55, .22], 5: [.25, .75],
    6: [.30, .25, .18, .27], 7: [.25, *([.125] * 6)],
    8: [.38, .34, .28], 9: [.27, .11, .205, .205, .21],
    12: [.16, .25, .34, .25],
}
ESCAPES = {
    "&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$",
    "−": r"\ensuremath{-}", "–": "--", "—": "---", "·": r"\textperiodcentered{}",
    "’": "'", "§": r"\S{}", "Δ": r"\ensuremath{\Delta}", "⊆": r"\ensuremath{\subseteq}",
    "≥": r"\ensuremath{\geq}",
}


def tex(text):
    inline_math = {
        "nₜ(S)": r"$n_t(S)$", "wₜ = fₜ⁻¹⁄⁴": r"$w_t=f_t^{-1/4}$",
        "fₜ": r"$f_t$", "Δ(i | A) ≥ Δ(i | B)": r"$\Delta(i\mid A)\geq\Delta(i\mid B)$",
        "√(n+1)−√n": r"$\sqrt{n+1}-\sqrt n$",
    }
    parts = re.split("(" + "|".join(re.escape(s) for s in inline_math) + ")", text)
    return "".join(inline_math.get(part, "".join(ESCAPES.get(c, c) for c in part))
                   for part in parts)


def tex_table(table, number):
    widths = WIDTHS[number]
    specs = []
    for i, fraction in enumerate(widths):
        align = r"\raggedright" if i == 0 or (number in (4, 5) and i == 1) else r"\centering"
        specs.append(r">{\hsize=" + f"{len(widths) * fraction:.4f}" +
                     r"\hsize\linewidth=\hsize" + align + r"\arraybackslash}X")
    lines = [r"\begin{tabularx}{\linewidth}{" + "".join(specs) + "}", r"\toprule",
             " & ".join(r"\textcolor{accent}{\textbf{" + tex(c) + "}}" for c in table["heads"]) + r" \\",
             r"\midrule"]
    for i, row in enumerate(table["rows"]):
        cells = [tex(c) for c in row]
        if table.get("emphasis") == i:
            cells = [r"\textbf{" + c + "}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabularx}"])


def write_beamer():
    lines = [r"""\documentclass[aspectratio=169,11pt]{beamer}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,booktabs,tabularx,array}
% Uncomment to print the presenter notes after each slide.
% \setbeameroption{show notes}
\setbeamersize{text margin left=8mm,text margin right=8mm}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{headline}{}"""]
    lines += [rf"\definecolor{{{name}}}{{HTML}}{{{code}}}" for name, code in PALETTE.items()]
    lines += [r"""
\setbeamercolor{normal text}{fg=text,bg=paper}
\setbeamercolor{frametitle}{fg=ink,bg=paper}
\setbeamerfont{frametitle}{size=\Large,series=\bfseries}
\setbeamercolor{structure}{fg=accent}
\setbeamertemplate{footline}{\hbox to \paperwidth{%
  \hspace{8mm}\color{muted}\scriptsize MAC F266 \enspace / \enspace BDD100K\hfill
  \insertframenumber\hspace{8mm}}\vskip3mm}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.28}
\newcommand{\credit}[1]{\vfill{\scriptsize\color{muted}#1\par}}
\newcommand{\takeaway}[1]{\medskip{\color{accent}\bfseries #1\par}}
\newcommand{\talkpoint}[2]{%
  \smallskip{\color{accent}\bfseries #1}\enspace #2\par}
\begin{document}
\begin{frame}[plain]
\vfill
{\color{accent}\bfseries MAC F266\par}
\vspace{4mm}
{\fontsize{25}{29}\selectfont\bfseries\color{ink}
Target-aware image selection\\for object detection\par}
\vspace{3mm}
{\large Post-midsem experiments on BDD100K\par}
\vspace{7mm}
Yash Saxena \enspace / \enspace November 2026
\vfill
""", r"\note{" + tex(SLIDES[0]["notes"]) + "}", r"\end{frame}"]
    for number, slide in enumerate(SLIDES[1:], 2):
        if number == 11:
            lines.append(r"\appendix")
        lines.extend(["", r"\begin{frame}[t]{" + tex(slide["title"]) + "}",
                      r"{\scriptsize\bfseries\color{accent}" + tex(slide["section"]) + r"\par}",
                      r"\smallskip"])
        if "lead" in slide:
            lines.append(tex(slide["lead"]) + r"\par\smallskip")
        if "math" in slide and number != 12:
            lines.append(r"\[" + slide["math"] + r"\]")
        if "table" in slide:
            lines.extend([r"\smallskip", r"{\small", tex_table(slide["table"], number), r"\par}"])
        if number == 12:
            lines.append(r"\[" + slide["math"] + r"\]")
        for label, point in slide.get("bullets", []):
            lines.append(r"\talkpoint{" + tex(label) + "}{" + tex(point) + "}")
        if "callout" in slide:
            lines.append(r"\takeaway{" + tex(slide["callout"]) + "}")
        lines.extend([r"\credit{" + tex(slide["source"]) + "}",
                      r"\note{" + tex(slide["notes"]) + "}", r"\end{frame}"])
    (OUT / f"{STEM}.tex").write_text("\n".join(lines + ["", r"\end{document}", ""]),
                                    encoding="utf-8")


def write_notes():
    lines = ["# BDD100K post-midsem speaker notes", "",
             "Slides 1–10 are the main talk. Slides 11–13 are for questions.", ""]
    for number, slide in enumerate(SLIDES, 1):
        lines.extend([f"## {number}. {slide['title']}", "", slide["notes"], ""])
    (OUT / "BDD100K_Post_Midsem_Speaker_Notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_pdf():
    from matplotlib.mathtext import math_to_image
    from matplotlib.font_manager import FontProperties
    from matplotlib import rc_context
    from PIL import Image
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph, Table, TableStyle

    fonts = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("Segoe", str(fonts / "segoeui.ttf")))
    pdfmetrics.registerFont(TTFont("Segoe Bold", str(fonts / "segoeuib.ttf")))
    pdfmetrics.registerFontFamily("Segoe", normal="Segoe", bold="Segoe Bold")
    colors = {name: HexColor("#" + value) for name, value in PALETTE.items()}
    page_w, page_h, left = 960, 540, 58
    content_w = page_w - 2 * left
    staging = HERE / f"{STEM}.pdf"
    document = canvas.Canvas(str(staging), pagesize=(page_w, page_h), pageCompression=1)
    document.setTitle(SLIDES[0]["title"])
    document.setAuthor("Yash Saxena")
    document.setSubject("MAC F266 · BDD100K post-midsem experiments")

    def text(value, top, size=24, *, lead=None, bold=False, color="text",
             width=content_w, markup=False):
        style = ParagraphStyle("slide", fontName="Segoe Bold" if bold else "Segoe",
                               fontSize=size, leading=lead or size * 1.25,
                               textColor=colors[color])
        block = Paragraph(value if markup else html.escape(value), style)
        _, height = block.wrap(width, page_h)
        block.drawOn(document, left, top - height)
        return top - height

    def table(spec, number, top):
        align_left = {0, 1} if number in (4, 5) else {0}
        cells = []
        size = 17 if number in (4, 6, 7, 9, 12) else 19
        for i, row in enumerate([spec["heads"], *spec["rows"]]):
            line = []
            for j, cell in enumerate(row):
                emphasize = i == 0 or spec.get("emphasis", -2) == i - 1
                style = ParagraphStyle("cell", fontName="Segoe Bold" if emphasize else "Segoe",
                                       fontSize=size, leading=size * 1.18,
                                       textColor=colors["accent" if i == 0 else "ink"])
                if j not in align_left:
                    style.alignment = TA_CENTER
                line.append(Paragraph(html.escape(cell), style))
            cells.append(line)
        styles = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                  ("LEFTPADDING", (0, 0), (-1, -1), 11),
                  ("RIGHTPADDING", (0, 0), (-1, -1), 11),
                  ("TOPPADDING", (0, 0), (-1, -1), 8),
                  ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                  ("LINEABOVE", (0, 0), (-1, 0), 1.1, colors["accent"]),
                  ("LINEBELOW", (0, 0), (-1, -1), 0.65, colors["line"])]
        if "emphasis" in spec:
            row = spec["emphasis"] + 1
            styles.append(("BACKGROUND", (0, row), (-1, row), colors["tint"]))
        block = Table(cells, colWidths=[content_w * f for f in WIDTHS[number]],
                      style=TableStyle(styles))
        _, height = block.wrap(content_w, page_h)
        block.drawOn(document, left, top - height)
        return top - height

    def equation(expression, top):
        buffer = io.BytesIO()
        with rc_context({"mathtext.fontset": "stix", "savefig.transparent": True}):
            math_to_image("$" + expression + "$", buffer, prop=FontProperties(size=27),
                          dpi=280, format="png", color="#" + PALETTE["ink"])
        buffer.seek(0)
        image = Image.open(buffer).convert("RGBA")
        w, h = (pixels * 72 / 280 for pixels in image.size)
        factor = min(1, (content_w - 9) / w, 62 / h)
        w, h = w * factor, h * factor
        document.drawImage(ImageReader(image), left + (content_w - w) / 2, top - h - 2,
                           width=w, height=h, mask="auto")
        return top - h - 6

    for number, slide in enumerate(SLIDES, 1):
        document.setFillColor(colors["paper"])
        document.rect(0, 0, page_w, page_h, fill=1, stroke=0)
        document.bookmarkPage(f"slide{number}")
        document.addOutlineEntry(slide["title"], f"slide{number}")
        if number == 1:
            text("MAC F266", 399, 18, bold=True, color="accent")
            text("Target-aware image selection<br/>for object detection", 354, 52,
                 lead=59, bold=True, color="ink", markup=True)
            text(slide["subtitle"], 209, 26)
            text("Yash Saxena  /  November 2026", 131, 19, color="muted")
            document.showPage()
            continue

        text(slide["section"], 504, 14, bold=True, color="accent")
        y = text(slide["title"], 473, 37, lead=42, bold=True, color="ink") - 22
        if "lead" in slide:
            y = text(slide["lead"], y, 22, lead=27) - 13
        if "math" in slide and number != 12:
            y = equation(slide["math"], y) - 11
        if "table" in slide:
            y = table(slide["table"], number, y) - 15
        if number == 12:
            y = equation(slide["math"], y) - 11
        for label, point in slide.get("bullets", []):
            prefix = f'<font color="#{PALETTE["accent"]}"><b>{html.escape(label)}</b></font>'
            y = text(prefix + " &nbsp; " + html.escape(point), y, 22, lead=28, markup=True) - 9
        if "callout" in slide:
            y = text(slide["callout"], y, 22, lead=27, bold=True, color="accent")
        if y < 67:
            raise ValueError(f"Slide {number} runs into the footer (y={y:.1f})")
        text(slide["source"], 53, 13, color="muted", width=content_w - 58)
        document.setFillColor(colors["muted"])
        document.setFont("Segoe", 13)
        document.drawRightString(page_w - left, 38, f"{number:02d} / {len(SLIDES)}")
        document.showPage()
    document.save()
    staging.replace(OUT / f"{STEM}.pdf")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    write_beamer()
    write_notes()
    write_pdf()
