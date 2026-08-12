#!/usr/bin/env python3
"""
Keep the problem <-> solution links consistent in both directions.

Each problem lists its solutions in the `solutions:` front matter key, and each
solution lists the problems it addresses in its `problems:` key. Both lists are
maintained by hand and drift apart over time: a solution gets a new problem slug
without the problem file learning about it, or the other way round.

This script builds the union of both directions and writes it back, so every
link shows up on both ends. Slugs that point to a non-existent file are reported
and dropped.

Usage:
    python scripts/sync_problem_solution_links.py           # update files
    python scripts/sync_problem_solution_links.py --dry-run # preview only
"""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBLEMS_DIR = os.path.join(REPO_ROOT, '_problems')
SOLUTIONS_DIR = os.path.join(REPO_ROOT, '_solutions')


def split_frontmatter(content):
    """Return (frontmatter, body) or (None, None) if the file has no front matter."""
    match = re.match(r'^---\n(.*?\n)---\n(.*)$', content, re.DOTALL)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def read_slug_list(frontmatter, key):
    """Read a simple block sequence of slugs under `key` from the front matter."""
    match = re.search(
        rf'^{key}:[ \t]*\n((?:[ \t]*-[ \t]+\S+[ \t]*\n)*)',
        frontmatter,
        re.MULTILINE,
    )
    if not match:
        return []
    return [line.strip()[2:].strip() for line in match.group(1).splitlines() if line.strip()]


def write_slug_list(frontmatter, key, slugs):
    """Replace (or insert) the block sequence under `key` with `slugs`."""
    block = f'{key}:\n' + ''.join(f'- {slug}\n' for slug in slugs)

    existing = re.search(
        rf'^{key}:[ \t]*\n(?:[ \t]*-[ \t]+\S+[ \t]*\n)*',
        frontmatter,
        re.MULTILINE,
    )
    if existing:
        return frontmatter[:existing.start()] + block + frontmatter[existing.end():]

    # Not present yet: put it directly above `layout:`, which always comes last.
    layout = re.search(r'^layout:.*\n', frontmatter, re.MULTILINE)
    if layout:
        return frontmatter[:layout.start()] + block + frontmatter[layout.start():]
    return frontmatter + block


def collect(directory, key):
    """Map slug -> list of linked slugs for every markdown file in `directory`."""
    links = {}
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith('.md'):
            continue
        with open(os.path.join(directory, filename), encoding='utf-8') as f:
            frontmatter, _ = split_frontmatter(f.read())
        if frontmatter is None:
            continue
        links[filename[:-3]] = read_slug_list(frontmatter, key)
    return links


def merge_preserving_order(existing, wanted):
    """
    Keep the curated order of `existing` (most relevant first) and append the
    slugs that only the other side knew about, alphabetically.
    """
    kept = [slug for slug in dict.fromkeys(existing) if slug in wanted]
    added = sorted(wanted - set(kept))
    return kept + added


def apply_links(directory, key, wanted, dry_run):
    """Write the merged slug lists back into the files of `directory`."""
    changed = 0
    for slug, slugs in sorted(wanted.items()):
        path = os.path.join(directory, f'{slug}.md')
        with open(path, encoding='utf-8') as f:
            content = f.read()
        frontmatter, body = split_frontmatter(content)
        if frontmatter is None:
            continue

        if read_slug_list(frontmatter, key) == slugs:
            continue

        changed += 1
        if dry_run:
            continue
        updated = write_slug_list(frontmatter, key, slugs)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f'---\n{updated}---\n{body}')
    return changed


def main():
    dry_run = '--dry-run' in sys.argv

    problem_to_solutions = collect(PROBLEMS_DIR, 'solutions')
    solution_to_problems = collect(SOLUTIONS_DIR, 'problems')

    problems = set(problem_to_solutions)
    solutions = set(solution_to_problems)

    edges = set()
    dangling = []
    for problem, slugs in problem_to_solutions.items():
        for solution in slugs:
            if solution in solutions:
                edges.add((problem, solution))
            else:
                dangling.append(f'_problems/{problem}.md -> {solution}')
    for solution, slugs in solution_to_problems.items():
        for problem in slugs:
            if problem in problems:
                edges.add((problem, solution))
            else:
                dangling.append(f'_solutions/{solution}.md -> {problem}')

    merged_problems = {
        p: merge_preserving_order(problem_to_solutions[p], {s for q, s in edges if q == p})
        for p in problems
    }
    merged_solutions = {
        s: merge_preserving_order(solution_to_problems[s], {p for p, q in edges if q == s})
        for s in solutions
    }

    before = sum(len(v) for v in problem_to_solutions.values())
    print(f'Problems: {len(problems)}, solutions: {len(solutions)}')
    print(f'Links before: {before} (problem side), {sum(len(v) for v in solution_to_problems.values())} (solution side)')
    print(f'Links after:  {len(edges)} on both sides')

    if dangling:
        print(f'\nDropped {len(dangling)} link(s) to missing files:')
        for entry in dangling:
            print(f'  {entry}')

    changed_problems = apply_links(PROBLEMS_DIR, 'solutions', merged_problems, dry_run)
    changed_solutions = apply_links(SOLUTIONS_DIR, 'problems', merged_solutions, dry_run)

    verb = 'Would update' if dry_run else 'Updated'
    print(f'\n{verb} {changed_problems} problem file(s) and {changed_solutions} solution file(s)')


if __name__ == '__main__':
    main()
