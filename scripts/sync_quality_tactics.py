#!/usr/bin/env python3
"""
Sync title and short description from Quality Tactics tactic files into _solutions/ front matter.

For each _solutions/ file that has a quality_tactics_url field, this script finds the
matching tactic file in ../qualitaetstaktiken/tactics/, extracts its title and short
description, and updates those two fields in the solution's front matter. The body
(How to Apply, Tradeoffs, Examples) is never touched.

When the sibling qualitaetstaktiken repo is not available, falls back to reading from
quality-tactics-reference.md in the repo root.

Usage:
    python scripts/sync_quality_tactics.py           # update files
    python scripts/sync_quality_tactics.py --dry-run # preview only
"""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TACTICS_DIR = os.path.join(REPO_ROOT, '..', 'qualitaetstaktiken', 'tactics')
SOLUTIONS_DIR = os.path.join(REPO_ROOT, '_solutions')
REFERENCE_FILE = os.path.join(REPO_ROOT, 'quality-tactics-reference.md')


def parse_tactic(filepath):
    """Extract title and short description (bold line) from a tactic file."""
    with open(filepath, encoding='utf-8') as f:
        content = f.read()

    title_match = re.search(r'^## (.+)$', content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else None

    desc_match = re.search(r'^\*\*(.+)\*\*$', content, re.MULTILINE)
    description = desc_match.group(1).strip() if desc_match else None

    return title, description


def build_tactic_index():
    """
    Walk ../qualitaetstaktiken/tactics/ and build a mapping:
        url -> {title, description}
    where url is the canonical qualitytactics.de URL for that tactic.
    Falls back to quality-tactics-reference.md if the sibling repo is absent.
    """
    if os.path.isdir(TACTICS_DIR):
        return _build_index_from_repo()
    elif os.path.isfile(REFERENCE_FILE):
        print(f'  qualitaetstaktiken repo not found — falling back to {os.path.basename(REFERENCE_FILE)}')
        return _build_index_from_reference()
    else:
        return {}


def _build_index_from_repo():
    index = {}

    for chapter_dir in sorted(os.listdir(TACTICS_DIR)):
        chapter_path = os.path.join(TACTICS_DIR, chapter_dir)
        if not os.path.isdir(chapter_path):
            continue

        en_path = os.path.join(chapter_path, 'en')
        if not os.path.isdir(en_path):
            continue

        # "07_maintainability" -> "maintainability", "05_performance_efficiency" -> "performance-efficiency"
        quality_char = re.sub(r'^\d+_', '', chapter_dir).replace('_', '-')

        for fname in sorted(os.listdir(en_path)):
            if not fname.endswith('.md') or fname.startswith('_'):
                continue

            slug = fname[:-3].replace('_', '-')
            url = f'https://qualitytactics.de/en/{quality_char}/{slug}/'

            title, description = parse_tactic(os.path.join(en_path, fname))
            if title and description:
                index[url] = {'title': title, 'description': description}

    return index


def _build_index_from_reference():
    """Parse quality-tactics-reference.md table rows into a url->tactic mapping."""
    index = {}
    with open(REFERENCE_FILE, encoding='utf-8') as f:
        for line in f:
            # Match table rows: | Title | Short Description | URL |
            m = re.match(r'^\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(https://\S+?)\s*\|', line)
            if m:
                title, description, url = m.group(1), m.group(2), m.group(3)
                if title != 'Title':  # skip header row
                    index[url] = {'title': title, 'description': description}
    return index


def remove_quality_tactics_url(filepath):
    """Remove the quality_tactics_url field from a solution file's front matter."""
    with open(filepath, encoding='utf-8') as f:
        content = f.read()

    if not content.startswith('---'):
        return

    second_dashes = content.index('---', 3)
    fm = content[3:second_dashes]
    body = content[second_dashes + 3:]

    fm = re.sub(r'^quality_tactics_url:.*\n', '', fm, flags=re.MULTILINE)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f'---{fm}---{body}')


def update_frontmatter(filepath, title, description, dry_run):
    """Update title and description in a solution file's YAML front matter."""
    with open(filepath, encoding='utf-8') as f:
        content = f.read()

    if not content.startswith('---'):
        print(f'  SKIP {os.path.basename(filepath)}: no front matter')
        return False

    # Split into front matter and body
    second_dashes = content.index('---', 3)
    fm = content[3:second_dashes]
    body = content[second_dashes + 3:]

    # Update or insert title
    if re.search(r'^title:', fm, re.MULTILINE):
        fm = re.sub(r'^title:.*$', f'title: {title}', fm, flags=re.MULTILINE)
    else:
        fm = f'title: {title}\n' + fm

    # Update or insert description (single-line only)
    if re.search(r'^description:', fm, re.MULTILINE):
        fm = re.sub(r'^description:.*$', f'description: {description}', fm, flags=re.MULTILINE)
    else:
        fm = fm.rstrip('\n') + f'\ndescription: {description}\n'

    new_content = f'---{fm}---{body}'

    if not dry_run:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

    return True


def main():
    dry_run = '--dry-run' in sys.argv

    if not os.path.isdir(TACTICS_DIR) and not os.path.isfile(REFERENCE_FILE):
        print(f'ERROR: qualitaetstaktiken not found at {TACTICS_DIR}')
        print(f'       and no fallback at {REFERENCE_FILE}')
        print('Clone qualitaetstaktiken as a sibling directory, or ensure quality-tactics-reference.md exists.')
        sys.exit(1)

    if not os.path.isdir(SOLUTIONS_DIR):
        print('No _solutions/ directory — nothing to sync.')
        sys.exit(0)

    print('Building Quality Tactics index...')
    index = build_tactic_index()
    print(f'  {len(index)} tactics indexed.')

    updated = skipped = warned = 0

    for fname in sorted(os.listdir(SOLUTIONS_DIR)):
        if not fname.endswith('.md'):
            continue

        filepath = os.path.join(SOLUTIONS_DIR, fname)
        with open(filepath, encoding='utf-8') as f:
            content = f.read()

        url_match = re.search(r'^quality_tactics_url:\s*(.+)$', content, re.MULTILINE)
        if not url_match:
            skipped += 1
            continue

        url = url_match.group(1).strip()
        if url not in index:
            verb = 'Would remove QT link from' if dry_run else 'Removing QT link from'
            print(f'  {verb}: {fname} (URL not found: {url})')
            if not dry_run:
                remove_quality_tactics_url(filepath)
            warned += 1
            continue

        tactic = index[url]
        verb = 'Would update' if dry_run else 'Updating'
        print(f'  {verb}: {fname} → "{tactic["title"]}"')
        update_frontmatter(filepath, tactic['title'], tactic['description'], dry_run)
        updated += 1

    print(f'\nDone: {updated} updated, {skipped} skipped (no QT url), {warned} QT links removed (tactic deleted/renamed).')
    if dry_run:
        print('(dry run — no files written)')


if __name__ == '__main__':
    main()
