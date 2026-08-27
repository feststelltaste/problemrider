#!/usr/bin/env python3
"""
List `_problems/` and `_solutions/` entries that have no German counterpart yet.

Phase 3 of the German translation effort (see plans/german-translation-plan.md)
gave every existing `_problems/*.md` and `_solutions/*.md` file a German
translation in `_problems_de/` / `_solutions_de/`, linked back via the
`en_slug` front-matter key. New English entries added afterwards (by
`/pr:generate_new_problems`, manual additions, etc.) don't automatically get
a German counterpart, and there is no other marker anywhere that flags this —
so this script IS the backlog: run it periodically (Phase 5 "Nachzieh-Lauf")
to see what still needs translating, instead of maintaining a separate
tracking file that could itself go stale.

Usage:
    python scripts/list_untranslated_de.py

Exit code is 0 if every English entry has a German counterpart, 1 otherwise
(so this can be used as a CI-style check, e.g. right after a batch of new
problem/solution files is merged).
"""

import glob
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def slug_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def split_frontmatter(content):
    match = re.match(r'^---\n(.*?\n)---\n', content, re.DOTALL)
    return match.group(1) if match else None


def read_en_slug(frontmatter):
    match = re.search(r'^en_slug:\s*(\S+)\s*$', frontmatter, re.MULTILINE)
    return match.group(1) if match else None


def find_untranslated(en_dir, de_dir):
    """Return sorted EN slugs under `en_dir` with no matching `en_slug` in `de_dir`."""
    en_slugs = {slug_of(p) for p in glob.glob(os.path.join(en_dir, '*.md'))}

    de_en_slugs = set()
    for path in glob.glob(os.path.join(de_dir, '*.md')):
        with open(path, encoding='utf-8') as f:
            frontmatter = split_frontmatter(f.read())
        if frontmatter is None:
            continue
        en_slug = read_en_slug(frontmatter)
        if en_slug:
            de_en_slugs.add(en_slug)

    return sorted(en_slugs - de_en_slugs)


def main():
    problems_missing = find_untranslated(
        os.path.join(REPO_ROOT, '_problems'),
        os.path.join(REPO_ROOT, '_problems_de'),
    )
    solutions_missing = find_untranslated(
        os.path.join(REPO_ROOT, '_solutions'),
        os.path.join(REPO_ROOT, '_solutions_de'),
    )

    if not problems_missing and not solutions_missing:
        print("Nothing to translate: every _problems/ and _solutions/ entry "
              "has a German counterpart.")
        return 0

    if problems_missing:
        print(f"Problems missing a German translation ({len(problems_missing)}):")
        for slug in problems_missing:
            print(f"  - {slug}")
        print()

    if solutions_missing:
        print(f"Solutions missing a German translation ({len(solutions_missing)}):")
        for slug in solutions_missing:
            print(f"  - {slug}")
        print()

    return 1


if __name__ == '__main__':
    sys.exit(main())
