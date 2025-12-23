#!/bin/bash

mkdir -p static/pdf

echo "Generating conchordal_manifesto.pdf..."
sed '/^+++$/,/^+++$/d' content/manifesto.md | sed '/^\[Download PDF\]/d' | \
  pandoc -o static/pdf/conchordal_manifesto.pdf \
    --pdf-engine=xelatex \
    --variable geometry:margin=1in \
    --variable fontsize=11pt \
    --variable linestretch=1.3 \
    --metadata title="Manifesto Conchordal"

echo "Generating conchordal_manifesto_ja.pdf..."
sed '/^+++$/,/^+++$/d' content/manifesto.ja.md | sed '/^\[PDFをダウンロード\]/d' | python3 add_zwsp.py | \
  pandoc -o static/pdf/conchordal_manifesto_ja.pdf \
    --pdf-engine=xelatex \
    --variable geometry:margin=1in \
    --variable fontsize=11pt \
    --variable linestretch=1.3 \
    --variable mainfont="Noto Serif CJK JP" \
    --variable sansfont="Noto Sans CJK JP" \
    --metadata title="Manifesto Conchordal"

echo "Generating conchordal_technote.pdf..."
sed '/^+++$/,/^+++$/d' content/technote.md | sed '/^\[Download PDF\]/d' | \
  pandoc -o static/pdf/conchordal_technote.pdf \
    --pdf-engine=xelatex \
    --variable geometry:margin=1in \
    --variable fontsize=10pt \
    --variable linestretch=1.2 \
    --toc \
    --toc-depth=2 \
    --metadata title="Technical Note: The Physics of Conchordal"

echo ""
echo "✓ conchordal_manifesto.pdf"
echo "✓ conchordal_manifesto_ja.pdf"
echo "✓ conchordal_technote.pdf"
