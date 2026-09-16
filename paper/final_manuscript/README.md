# Final Manuscript Bundle

This directory is the paper-facing archive for the audited historical
ProtoHedge study. It contains the full LaTeX draft, compiled PDF, generated
tables and figures, compact checksummed inputs, and a locked results ledger.

## Main Files

- `protohedge_empirical_validation_ieee.tex`: anonymous empirical IEEE
  conference manuscript for double-anonymous review.
- `protohedge_empirical_validation_ieee.txt`: plain-text copy of the complete
  LaTeX source for direct viewing or pasting into Overleaf.
- `protohedge_empirical_validation_ieee.pdf`: compiled anonymous submission
  PDF.
- `protohedge_empirical_validation_ieee_identified.tex`: identified author
  version retained for post-acceptance preparation; do not submit it for
  double-anonymous review.
- `protohedge_empirical_validation_ieee_identified.txt`: plain-text copy of the
  identified source.
- `overleaf_submission/`: minimal self-contained source package with the
  anonymous manuscript, bibliography, referenced figure, compiled PDF, and
  hashes.
- `protohedge_overleaf_submission.zip`: upload-ready archive of that package.
- `references_empirical.bib`: focused bibliography for the concise manuscript.
- `protohedge_historical_validation.tex`: full IEEEtran manuscript draft.
- `protohedge_historical_validation.pdf`: compiled 19-page draft.
- `references.bib`: manuscript bibliography.
- `RESULTS_LEDGER.md`: concise numerical claim boundary.
- `generate_assets.py`: deterministic table, figure, and archive generator.
- `tables/`: manuscript-ready LaTeX tables and their CSV counterparts.
- `figures/`: vector PDF figures and PNG previews.
- `source_data/`: compact copies of every result input used by the generator.
- `generation_environment.json`: Git and package provenance at generation.

The raw historical result tree is not duplicated here. Its embedded manifest
records SHA-256 hashes for all 4,025 raw files. Separate embedded manifests
cover the 53 panel inputs and 73 archived source files.

Only the anonymous PDF or `overleaf_submission/` bundle is suitable for review.
The broader research archive contains identifying provenance, including local
filesystem paths, and must not be uploaded during double-anonymous review.
After acceptance, restore the author block from the identified source and
replace the Introduction's release pledge with the permanent artifact link.

## Rebuild

From the repository root:

```bash
cd paper/final_manuscript
python generate_assets.py
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  protohedge_historical_validation.tex

# Build the six-page empirical submission, including references.
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  protohedge_empirical_validation_ieee.tex
```

## Verify

```bash
cd paper/final_manuscript/source_data
shasum -a 256 -c source_checksums.sha256
cd ..
shasum -a 256 -c generated_asset_checksums.sha256
shasum -a 256 -c release_checksums.sha256
```

## Headline Interpretation

The defensible result is synthetic near parity, historical European near
parity after the targeted checkpoint sensitivity, and a positive Asian
tail-selected panel result for the validation-selected expanded ProtoHedge
family. The evidence does not establish universal ProtoHedge superiority, an
exact full-panel numerical port, or superiority over an equally tuned Deep
Hedging family.

The historical checkpoint sensitivity is hybrid and targeted: it replaces the
12 affected initialized Deep Hedging fits and their matched selected
ProtoHedge comparisons under pure validation OCE, while retaining unaffected
completed-run outputs. It is not a complete rerun of every model candidate.

## Editing Rule

Change numerical inputs or `generate_assets.py`, then regenerate. Do not edit
generated tables or figures by hand. The manuscript can be shortened for a
venue later without changing this evidence archive.
