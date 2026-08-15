Here is a corrected and polished version of your introduction text, keeping it precise but clear. I fixed grammar, typos, style consistency, and made some sentences flow better.

---

# Project Information

## Goal

The primary goal of this project is to create a catalog of typical problems found in legacy systems, along with their symptoms and root causes.

This catalog is intended to be especially valuable for software maintainers and architects who need to perform legacy system analysis and modernization tasks.

# Scope
The scope of the problem area ends at the requirements and management level (for example, political causes like tariff wars are out of scope) and does not include very detailed technological issues (for example, CPU-level problems are out of scope).

## Layout

* The `_problems` directory contains the relevant content, including the problems and their descriptions.
* The `_solutions` directory contains solutions, each linking to one or more problems.
* The `problem-pattern-template.md` file defines the format for each problem pattern.

## Linking

* When documenting a problem, link its symptoms and root causes to other existing problems whenever possible. Do this by directly linking the title of those problems within the text, not by adding links at the end.
* Use simple Markdown links with relative paths.

## Solutions

* Solutions live in the `_solutions/` directory, one file per solution.
* Each problem links to its solutions via a `solutions:` list of solution slugs in its front matter.
* Solutions that correspond to a tactic in the [Quality Tactics](https://qualitytactics.de/en/) book carry a `quality_tactics_url` field. Their `title` and front-matter `description` are automatically synced from the Quality Tactics tactic files by running `scripts/sync_quality_tactics.py`. The body content (Description, How to Apply, Tradeoffs, Examples) is never overwritten by the sync.
* A standalone reference of all 539 Quality Tactics (title, short description, URL, category) is available in `quality-tactics-reference.md`. Use this file to look up or pick tactics without needing the `qualitaetstaktiken` repo. The sync script also falls back to this file automatically when the sibling repo is absent.
* Solutions without a Quality Tactics equivalent omit `quality_tactics_url`.
* Every solution, QT-backed or not, includes a `## Description` section in the body — a fuller paragraph beyond the one-line front-matter `description`.
* The `solution-pattern-template.md` file defines the format for solution files.

## Guiding Principle

The core task is to continuously expand the catalog by analyzing additional problems related to legacy systems and interconnecting existing problems when they are linked by cause or effect.

If you identify a symptom or root cause in an existing pattern, create a separate problem entry for it. Each root cause and symptom should have a brief, descriptive title and an accompanying explanation so that the work can be continued in the future.

Follow title case rules for titles, where nouns, pronouns, verbs, adjectives, and adverbs are capitalized (to be more precise: use New York Times Manual of Style).

Markdown file names should be in lowercase and use hyphens as separators.

When linking one problem to another, the linked title must match exactly (including upper and lower case) the title of the problem it leads to and the complete title only is the link text (no additions to the title). However, it is fine to include a context-specific description of the linked problem to explain why the symptom or cause is connected to it.

## Tech Stack

* The main idea is to publish the content as a Jekyll-based website on GitHub Pages.
* A prototype exists for a graph-based view using a Python script to generate a D3-based network of problems.

## Developing Locally

### Jekyll Site Prototype

To build the site and run it locally, use the following command:

`bundle exec jekyll serve`

The site will be available at [http://localhost:4000](http://localhost:4000).

### Fast Build for Styles and Scripts (Asset-Only)

When making changes only to styling (SCSS/CSS) or scripts (JS) without modifying Markdown problem or solution content, use the `--incremental` build flag:

`bundle exec jekyll build --incremental`

Or for fast local serving with incremental updates:

`bundle exec jekyll serve --incremental`

### Landscape View

Run `scripts/create_landscape.py` with Python to regenerate `assets/js/landscape-data.js`, the data file behind the `/landscape/` page. It reduces each problem's/solution's cached embedding (from `embeddings/problems/` / `embeddings/solutions/`, produced by `calculate_related_problems.py` / `calculate_related_solutions.py`) to a 2D position via UMAP, runs k-means on the original embeddings to find real cluster groups (UMAP alone tends to blur into one haze) and pushes those groups further apart, then nudges apart labels that would otherwise overlap. Re-run it whenever problems/solutions are added or their embeddings change. Use `--separation` to control how far apart cluster groups are pushed and `--min-label-distance` for how much room individual labels get (see `--help`).

### Helper Scripts

The `scripts/` directory contains more utility scripts for maintaining the catalog:

* `calculate_related_problems.py`: Generates semantic similarity scores for related_problems sections using sentence-transformers. Updates all problem files with automatically calculated relationships based on content similarity.
* `calculate_related_solutions.py`: Generates semantic similarity scores for related_solutions sections using the same embedding mechanism. Updates all solution files with automatically calculated similar solutions.
* `create_landscape.py`: Generates the UMAP-based clustered layout data (`assets/js/landscape-data.js`) for the `/landscape/` page, for both problems and solutions.
* `backlog_refinement.py`: Takes ideas withing the file `scripts/backlog/candidates.md` and sorts them into different files depending on already existing or similar problems.
* `sync_quality_tactics.py`: Syncs `title` and `description` from the Quality Tactics tactic files into `_solutions/` front matter for solutions that have a `quality_tactics_url`. Uses the `qualitaetstaktiken` sibling repo when available, otherwise falls back to `quality-tactics-reference.md`. Use `--dry-run` to preview.
* `sync_problem_solution_links.py`: Keeps the problem <-> solution links consistent in both directions. Builds the union of the `solutions:` lists in `_problems/` and the `problems:` lists in `_solutions/`, writes it back to both sides, and reports links pointing to missing files. Run it after adding or changing links on either side. Use `--dry-run` to preview.
* `validate_causal_links.py`: Checks the causal graph formed by the `Symptoms` and `Causes` sections. An entry under `Symptoms` of one problem and under `Causes` of another express the same directed claim, so a claim should appear in both files. Reports contradictions, self links, dangling targets, claims recorded on only one side, and how much of the graph collapses into a single strongly connected component. This is a structural consistency check; reciprocal entries do not prove that a causal relationship is true. Use `--detail` for every finding and `--asymmetry-report FILE` for the work list.
* `check_links.py`: Checks for broken markdown links in the `_problems` directory. Use `--fix` flag to automatically remove broken links while preserving the title and description text.
* `convert_titles.py`: Converts titles to proper title case using New York Times Manual of Style rules. Works on YAML front matter titles, H1 headers, and markdown link text. Use `--fix` flag to actually modify files.
* `consolidate_categories.py`: Consolidates problem categories from ~200+ categories down to 15 core categories to improve organization and navigation.

## Categories

The catalog uses 15 core categories to organize problems:

1. **Process** - workflow, planning, development process
2. **Architecture** - design, system structure, coupling issues
3. **Code** - maintainability, technical debt, code issues
4. **Performance** - speed, scalability, resource usage
5. **Team** - team coordination, collaboration
6. **Communication** - knowledge sharing, documentation
7. **Management** - leadership, project management
8. **Security** - vulnerabilities, compliance
9. **Business** - strategy, product, business impact
10. **Operations** - deployment, infrastructure, configuration
11. **Testing** - quality assurance, integration tests
12. **Database** - data management, queries
13. **Dependencies** - vendor management, integration, API issues
14. **Requirements** - user experience, planning, stakeholder needs
15. **Culture** - individual issues, workplace health, organizational problems

New categories may be added only if really needed and cannot be reasonably mapped to one of the existing 15 categories.

## Development Workflow Rules

* Do not automatically trigger `bundle exec jekyll build` / `jekyll serve` after intermediate code edits during active iteration steps. Only run build commands when explicitly asked by the user or when all task changes are completely finished.
* When a build is needed, prefer `bundle exec jekyll build --incremental` over a plain full build — it is much faster. A full (non-incremental) build is only needed when Markdown problem/solution content changed alongside the assets, since Jekyll's incremental mode does not reliably pick up every kind of content-side dependency. See "Fast Build for Styles and Scripts (Asset-Only)" above for the asset-only case this covers best.
* `--incremental` recompiles changed JS/CSS files themselves, but does NOT regenerate the problem/solution HTML pages that merely reference them via the `?v={{ site.time | date: '%s' }}` cache-busting query string — Jekyll's incremental dependency tracking does not see that link. Those pages keep the stale `?v=...` value, so a browser that already cached the old asset under that exact URL keeps using it no matter how many times the asset is rebuilt. Before checking a JS/CSS change live in a browser, run one plain `bundle exec jekyll build` (no `--incremental`) to refresh that timestamp everywhere — `--incremental` alone is only safe for a quick syntax/compile check.
* Before editing site layout/CSS/JS (`_layouts/`, `_includes/`, `assets/main.scss`, `assets/js/`), read `MISTAKES.md` — a log of regressions from past sessions (layout changes that broke on a page type nobody re-checked, CSS specificity/`hidden`-attribute traps, fixed-position elements silently eating clicks, scope bugs invisible without an AST trace, stale-cache confusion). Add a new entry there whenever a change like this turns out to have broken something, instead of only fixing it silently.
