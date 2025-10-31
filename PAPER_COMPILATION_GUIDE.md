# Research Paper Compilation Guide

## Files Created

1. **BENCHMARK_PAPER.md** - Markdown format (easiest to edit/share)
2. **benchmark_paper.tex** - IEEE conference format (LaTeX)
3. **benchmark_paper_simple.tex** - Standard article format (LaTeX)

## Quick Start

### Option 1: Use Markdown (Recommended for Quick Sharing)

The `BENCHMARK_PAPER.md` file is ready to use:
- View directly on GitHub/GitLab
- Convert to PDF using Pandoc: `pandoc BENCHMARK_PAPER.md -o paper.pdf`
- Or use online converters like Markdown to PDF

### Option 2: Compile LaTeX to PDF

#### For IEEE format:
```bash
pdflatex benchmark_paper.tex
bibtex benchmark_paper  # if you add citations
pdflatex benchmark_paper.tex
pdflatex benchmark_paper.tex
```

#### For simple article format:
```bash
pdflatex benchmark_paper_simple.tex
pdflatex benchmark_paper_simple.tex
```

## Required Files for Compilation

Make sure these chart files are in the same directory:
- `benchmark_full_pipeline_results_charts_inference_speed.png`
- `benchmark_full_pipeline_results_charts_translation_error.png`
- `benchmark_full_pipeline_results_charts_radar.png`
- `benchmark_full_pipeline_results_charts_improvements.png`

## Customization Needed

Before final submission, update:

1. **Author Information**: Replace "Your Name" and "Your Institution"
2. **Citations**: Add proper references in the bibliography section
3. **Abstract**: Adjust if needed based on your conference/journal requirements
4. **Figures**: Ensure chart paths are correct if compiling LaTeX
5. **Related Work**: Add more detailed citations to relevant papers

## Paper Structure

- Abstract
- Introduction
- Related Work
- Methodology
- Results (with tables and figures)
- Discussion
- Conclusion
- References

## Key Results Highlighted

- UAAS: 36.4% accuracy improvement
- Semantic: 7.5% improvement with zero overhead
- Probabilistic: 5.4% improvement + uncertainty quantification

## Tips

- For conferences: Use `benchmark_paper.tex` (IEEE format)
- For journals: Use `benchmark_paper_simple.tex` and customize
- For quick sharing: Use `BENCHMARK_PAPER.md`
- Check page limits and adjust sections accordingly

