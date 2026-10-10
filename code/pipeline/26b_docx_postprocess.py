#!/usr/bin/env python3
"""26b_docx_postprocess.py — Submission DOCX post-processing: line numbering + document metadata.

Using python-docx on manuscript.docx:
  1. Add continuous line numbering to every section (sectPr): w:lnNumType, countBy=1, restart=continuous
  2. Write core properties: title / subject / keywords / creator (placeholder)
Output: paper/manuscript.docx (updated in place)
"""
import os
import re
from copy import deepcopy
from pathlib import Path
from docx import Document
from docx.shared import RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

BASE = Path(os.environ.get("REPRO_BASE", str(Path(__file__).resolve().parents[2])))
F = BASE / "paper" / "manuscript.docx"

TITLE = ("Reservoir regulation rotates river thermal response geometry "
         "across a contiguous-U.S. benchmark")
KEYWORDS = ("river thermal regime; water temperature; reservoir operation; "
            "air-water temperature coupling; heatwave; environmental flow")

doc = Document(str(F))


def set_paragraph_text(paragraph, text):
    """Replace paragraph text while retaining its paragraph style and first-run formatting."""
    first = paragraph.runs[0] if paragraph.runs else paragraph.add_run()
    first.text = text
    for run in paragraph.runs[1:]:
        run.text = ""


def math_run(parent, text, italic=False):
    """Add a Word equation run with an explicit Cambria Math font."""
    r = OxmlElement("m:r")
    rpr = OxmlElement("m:rPr")
    fonts = OxmlElement("m:rFonts")
    for attr in ("m:ascii", "m:hAnsi", "m:cs", "m:eastAsia"):
        fonts.set(qn(attr), "Cambria Math")
    rpr.append(fonts)
    if italic:
        style = OxmlElement("m:sty")
        style.set(qn("m:val"), "i")
        rpr.append(style)
    r.append(rpr)
    t = OxmlElement("m:t")
    t.text = text
    r.append(t)
    parent.append(r)


def math_script(parent, base, sub=None, sup=None, italic=True):
    kind = "m:sSubSup" if sub is not None and sup is not None else (
        "m:sSub" if sub is not None else "m:sSup")
    script = OxmlElement(kind)
    elem = OxmlElement("m:e")
    math_run(elem, base, italic=italic)
    script.append(elem)
    if sub is not None:
        sub_elem = OxmlElement("m:sub")
        math_run(sub_elem, sub, italic=True)
        script.append(sub_elem)
    if sup is not None:
        sup_elem = OxmlElement("m:sup")
        math_run(sup_elem, sup, italic=False)
        script.append(sup_elem)
    parent.append(script)


def math_script_with_substyle(parent, base, sub, base_italic=True, sub_italic=False):
    """Add an inline subscript while keeping text subscripts upright when appropriate."""
    script = OxmlElement("m:sSub")
    elem = OxmlElement("m:e")
    math_run(elem, base, italic=base_italic)
    script.append(elem)
    sub_elem = OxmlElement("m:sub")
    math_run(sub_elem, sub, italic=sub_italic)
    script.append(sub_elem)
    parent.append(script)


def inline_math(paragraph, token):
    """Replace one plain-text variable token in a paragraph with editable inline OMML."""
    p = paragraph._p
    for run in list(p.findall(qn("w:r"))):
        text_nodes = run.findall(qn("w:t"))
        if not text_nodes:
            continue
        text = "".join(node.text or "" for node in text_nodes)
        if token not in text:
            continue
        before, after = text.split(token, 1)
        original_rpr = run.find(qn("w:rPr"))

        def append_text(value):
            if not value:
                return
            new_run = OxmlElement("w:r")
            if original_rpr is not None:
                new_run.append(deepcopy(original_rpr))
            t = OxmlElement("w:t")
            t.set(qn("xml:space"), "preserve")
            t.text = value
            new_run.append(t)
            p.insert(p.index(run), new_run)

        def append_formula(kind):
            omath = OxmlElement("m:oMath")
            if kind == "Q_peak7":
                math_script_with_substyle(omath, "Q", "peak7", base_italic=True, sub_italic=False)
            elif kind == "log10":
                math_script_with_substyle(omath, "log", "10", base_italic=False, sub_italic=False)
            p.insert(p.index(run), omath)

        append_text(before)
        append_formula(token)
        append_text(after)
        p.remove(run)
        return True
    return False


def add_model_equation(paragraph):
    """Insert an editable Word-native equation with real scripts."""
    p = paragraph._p
    for child in list(p):
        if child.tag != qn("w:pPr"):
            p.remove(child)
    math_para = OxmlElement("m:oMathPara")
    math = OxmlElement("m:oMath")
    math_script(math, "T", sub="g,t", sup="peak")
    math_run(math, " = ", italic=False)
    math_script(math, "α", sub="g")
    math_run(math, " + ", italic=False)
    math_script(math, "β", sub="g")
    math_run(math, "·", italic=False)
    math_script(math, "A", sub="g,t", sup="peak")
    math_run(math, " + ", italic=False)
    math_run(math, "γ", italic=True)
    math_run(math, "·", italic=False)
    math_script(math, "S", sub="d,t")
    math_run(math, " + ", italic=False)
    math_run(math, "δ", italic=True)
    math_run(math, "·", italic=False)
    math_script(math, "Q", sub="g,t")
    math_run(math, " + ", italic=False)
    math_run(math, "η", italic=True)
    math_run(math, "·", italic=False)
    math_script(math, "A", sub="g,t", sup="peak")
    math_run(math, "·", italic=False)
    math_script(math, "S", sub="d,t")
    math_run(math, " + ", italic=False)
    math_script(math, "ε", sub="g,t")
    math_para.append(math)
    p.append(math_para)


def all_paragraphs(document):
    yield from document.paragraphs
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                yield from cell.paragraphs
    for section in document.sections:
        yield from section.header.paragraphs
        yield from section.footer.paragraphs


for paragraph in doc.paragraphs:
    text = paragraph.text
    if "Q_peak7" in text:
        inline_math(paragraph, "Q_peak7")
        text = paragraph.text
    if "log10" in text:
        inline_math(paragraph, "log10")
        text = paragraph.text
    if "regional stream-temperature trends can diverge from air-temperature trends" in text:
        text = text.replace(
            "(Arismendi et al., 2012, 2014; Isaak et al., 2012; Rice & Jastram, 2015), and atmospheric",
            "(Arismendi et al., 2012, 2014; Isaak et al., 2012; Rice & Jastram, 2015), "
            "snowmelt-fed rivers can show heightened thermal vulnerability to warming "
            "(Yan et al., 2021), and atmospheric")
        set_paragraph_text(paragraph, text)
    if text.startswith("Reservoirs are a natural test of this paradox"):
        text = text.replace(
            "pathway by which atmospheric heat reaches downstream rivers. Seasonal storage",
            "pathway by which atmospheric heat reaches downstream rivers. Across the "
            "southeastern United States, reservoirs also modify river-temperature "
            "sensitivity to climate change (Cheng et al., 2020). Seasonal storage")
        set_paragraph_text(paragraph, text)
    if re.search(r"Philippus, D\., Corona, C\.R\., Hogue, T\.S\., et al\.\s*\(2025\)", text):
        set_paragraph_text(
            paragraph,
            re.sub(r"Philippus, D\., Corona, C\.R\., Hogue, T\.S\., et al\.\s*\(2025\)",
                   "Philippus, D., Corona, C.R., Schneider, K., Rust, A., Hogue, T.S. (2025)",
                   text))
    if ("[ T^{}{g,t}=g+g A^{}{g,t}+S{d,t}+Q{g,t}+,A^{}{g,t}S{d,t}+_{g,t}, ]" in text
            or "T^{{peak}}_{g,t}" in text
            or "T^{peak}_{g,t}" in text
            or text.startswith("T_peak(g,t) =")):
        add_model_equation(paragraph)
    elif "((_g))" in text and "state term (())" in text:
        text = text.replace("((_g))", "(β_g)")
        text = text.replace("state term (())", "state term (γ)")
        text = text.replace("interaction term (())", "interaction term (η)")
        set_paragraph_text(paragraph, text)

# 1) Continuous line numbering
n_sec = 0
for sec in doc.sections:
    sectPr = sec._sectPr
    # Remove any existing element first
    for el in sectPr.findall(qn("w:lnNumType")):
        sectPr.remove(el)
    ln = OxmlElement("w:lnNumType")
    ln.set(qn("w:countBy"), "1")
    ln.set(qn("w:restart"), "continuous")
    sectPr.append(ln)
    n_sec += 1

# 2) Metadata
cp = doc.core_properties
cp.title = TITLE
cp.subject = "A three-tier benchmark of river thermal response geometry below reservoirs"
cp.keywords = KEYWORDS
cp.author = "Zijie Deng; Yu Xia; Jiyang Yu; Zhenyu Liao"
cp.comments = ("Submission file with continuous line numbering and embedded figures. "
               "All statistics regenerate from the archived pipeline (reproduce.py).")

# Normalize template colors and typography, including heading and hyperlink styles.
for style in doc.styles:
    try:
        style.font.name = "Times New Roman"
        if style.font.color is not None and style.font.color.rgb is not None:
            style.font.color.rgb = RGBColor(0, 0, 0)
    except (AttributeError, TypeError):
        continue
for paragraph in all_paragraphs(doc):
    for run in paragraph.runs:
        run.font.name = "Times New Roman"
        if run.font.color is not None and run.font.color.rgb is not None:
            run.font.color.rgb = RGBColor(0, 0, 0)

doc.save(str(F))

# 3) Rebuild docProps/app.xml with statistics from the final document.
import zipfile, os, re
tmp = str(F) + ".tmp"
fig4 = BASE / "figures" / "fig4_paired.png"
fig4_rel = doc.part.rels[
    doc.inline_shapes[3]._inline.xpath(".//a:blip")[0].get(qn("r:embed"))]
fig4_media = str(fig4_rel.target_part.partname).lstrip("/")
with zipfile.ZipFile(str(F)) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    n_words = sum(len(p_.text.split()) for p_ in doc.paragraphs)
    n_paras = len(doc.paragraphs)
    app_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        f"<Application>Microsoft Office Word</Application><AppVersion>16.0000</AppVersion>"
        f"<Words>{n_words}</Words><Paragraphs>{n_paras}</Paragraphs>"
        "<Company></Company></Properties>")
    for item in zin.infolist():
        if item.filename == "docProps/app.xml":
            zout.writestr(item, app_xml)
        elif item.filename == fig4_media:
            zout.writestr(item, fig4.read_bytes())
        else:
            zout.writestr(item, zin.read(item.filename))
os.replace(tmp, str(F))

print(f"OK: {n_sec} sections line-numbered; metadata set; app.xml rebuilt ({n_words} words); Fig. 4 refreshed -> {F.name}")
