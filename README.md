# Player value

Cricket player strength and contributions: ratings, histories, uncertainty, and
comparisons across methods. This is the player-level counterpart to
[Outside Edge's team ratings](https://outside-edge.github.io/side_order/).

[Rankings website](https://outside-edge.github.io/player_value/) ·
[Test](https://outside-edge.github.io/player_value/docs/test.html) ·
[ODI](https://outside-edge.github.io/player_value/docs/odi.html) ·
[T20I](https://outside-edge.github.io/player_value/docs/t20i.html) ·
[Data and downloads](https://outside-edge.github.io/player_value/docs/data.html)

The project distinguishes **strength**, an estimate of a player's ability at a
given date, from **contribution**, the value of their performances in matches
already played. Models retain their own units and assumptions. WAR is one
possible contribution measure, not the definition of every player rating.

## Current status

[Player rankings](docs/ratings.qmd) are available for **Tests, ODIs, and T20
internationals**, separately for men and women, through **26 October 2026**.
The player-only models use five years of match outcomes; Tests include draws.
Every table includes appearances and downloadable coefficients. Women's Tests
have very little data and use the specified untuned regularization setting.
These are exploratory lineup associations, not WAR or instantaneous form.

The separate [ODI experiment](docs/models.qmd) compares team-only, player-only,
and combined models on chronological holdouts. Contribution estimates and
uncertainty intervals for individual ranks remain future work.

The earlier WAR notebooks, results, and figures are preserved in
[the exploratory collection](https://github.com/outside-edge/player_value/tree/main/experiments).
Their runs-to-wins conversion and replacement assumptions are not production
defaults. See the [modeling agenda](docs/models.qmd) for the research sequence and
acceptance criteria.

## Run locally

Install Python 3.12 or later and [uv](https://docs.astral.sh/uv/), then run:

```sh
make sync
make check
make fetch
make prepare AS_OF=2026-10-26
```

`make fetch` downloads the official Cricsheet ODI JSON archive and records its
checksum and retrieval time. `make prepare` validates identities and outcomes,
retains matches through the specified date, and writes match, participant, and
player tables to `data/build/odis/`. It does not estimate player ability.

An existing Cricsheet JSON or YAML directory can be prepared without downloading:

```sh
make prepare DATA_DIR=data/odis AS_OF=2026-10-26
```

Run the specified 2023–2025 forecast comparison and assemble its saved results:

```sh
make compare
make publish
```

`make compare` tunes each model using past matches, then evaluates annual holdouts
with three- and five-year training windows. `make publish` verifies the saved
outputs and creates the website's tables and downloads. The saved publication
can be rendered without downloading data or rerunning models. See the
[experiment specification](docs/lineup_experiment.qmd).

To build the additional format tables after preparing ODIs:

```sh
make fetch FORMAT=Test
make prepare FORMAT=Test AS_OF=2026-10-26
make fetch FORMAT=T20I
make prepare FORMAT=T20I AS_OF=2026-10-26 PREPARE_ARGS="--since 2021-10-27"
make rankings AS_OF=2026-10-26
```

`make rankings` tunes and fits six player-only models, then writes verified CSVs,
model parameters, and provenance to `out/release/current/`. The T20 preparation
start matches the five-year window; adjust it when changing the ranking cutoff.
See the [ranking specification](docs/format_rankings.qmd) and
[pipeline guide](docs/pipeline.qmd).

Install [Quarto](https://quarto.org/docs/get-started/) to render the website:

```sh
make site
```

Rendering uses saved documentation and does not fetch data or fit models.
`make ci-docker` runs the Python checks in a standard Python container when Docker
is running. The [pipeline guide](docs/pipeline.qmd) describes commands, outputs,
and provenance.

## Publication

- [Rankings](docs/ratings.qmd): Test, ODI, and T20I player tables and exposure flags.
- [Contributions](docs/contributions.qmd): match, season, and career value.
- [Player histories](docs/histories.qmd): dated estimates and retrospective histories.
- [Model comparisons](docs/models.qmd): research priorities and forecast evaluation.
- [Methods and data](docs/methods.qmd): definitions, coverage, and limitations.

Each publication will identify its model, cutoff, competition, units, and
uncertainty method. Rate measures and cumulative contributions will be reported
separately. Team ratings remain in `side_order`; any comparison will use dated
exports rather than a dependency on a sibling checkout.

