# Exploratory WAR calculations

These notebooks and outputs predate the player-value pipeline. They are retained
as research artifacts, not validated ratings or reproducible production releases.

`war/scripts/` contains the five original notebooks; `war/output/` contains their
saved CSVs; `war/readme/` contains the original figures. The original explanation
is in [war/methodology.md](war/methodology.md). Its claims describe the exploratory
implementation and do not establish the validity of its wins scale.

The ODI batting notebook fits expected runs using over, inferred batting position,
and year, scales expected runs by 0.85 to represent replacement, and divides the
residual by 10. It does not explicitly value the future cost of dismissals.
The runs-per-win notebook computes the mean victory margin among wins by runs;
this is not an estimate of runs per additional expected win. Test batting and
bowling notebooks use other assumptions and should not be combined as one scale.

The notebooks retain their original working-directory assumptions and dependencies.
They are excluded from production checks and automatic execution. Their outputs
must not feed current ratings without a new specification and validation.

[relocations.csv](relocations.csv) records each original path, its new location,
and SHA-256 hash. The move preserves every notebook, output, and figure byte for
byte. Existing ignored match downloads remain in the local `data/` directories.
