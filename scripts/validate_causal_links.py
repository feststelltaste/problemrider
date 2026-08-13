#!/usr/bin/env python3
"""
Check the causal graph formed by the Symptoms and Causes sections for structural defects.

Every link is a directed causal claim. A entry under `## Symptoms ▲` of problem A
claims A causes B. An entry under `## Causes ▼` of problem A claims C causes A.
Both notations express the same relation seen from opposite ends, so a claim
recorded on one side should also appear on the other.

The checks are structural and need no model:

  contradiction  the same problem appears under both Symptoms and Causes of one
                 problem, which asserts a cause and its own effect at once
  self           a problem lists itself
  asymmetry      an edge recorded on only one side, so the two problem files
                 disagree about whether the relation exists
  cycle          a causal loop, which may be a real reinforcing loop but is more
                 often an artifact of edges added independently
  hub            a problem far above the usual degree, which tends to indicate
                 links added because they were plausible rather than specific

Usage:
    python scripts/validate_causal_links.py            # summary
    python scripts/validate_causal_links.py --detail   # list every finding
    python scripts/validate_causal_links.py --asymmetry-report FILE
"""

import os
import re
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBLEMS_DIR = os.path.join(REPO_ROOT, '_problems')

SYMPTOMS_HEADING = 'Symptoms ▲'
CAUSES_HEADING = 'Causes ▼'

# A problem this far above the median out- or in-degree is reported as a hub.
HUB_FACTOR = 6


def read_section_links(body, heading):
    """Return the problem slugs linked in one section of a problem file."""
    match = re.search(
        rf'^## {re.escape(heading)}\n(.*?)(?=^## |\Z)',
        body,
        re.DOTALL | re.MULTILINE,
    )
    if not match:
        return []
    return re.findall(r'\[[^\]]+\]\(([a-z0-9-]+)\.md\)', match.group(1))


def load_problems():
    """Map each problem slug to its title and its two link sections."""
    problems = {}
    for filename in sorted(os.listdir(PROBLEMS_DIR)):
        if not filename.endswith('.md'):
            continue
        with open(os.path.join(PROBLEMS_DIR, filename), encoding='utf-8') as f:
            content = f.read()
        title = re.search(r'^title: (.*)$', content, re.MULTILINE)
        problems[filename[:-3]] = {
            'title': title.group(1) if title else filename[:-3],
            'symptoms': read_section_links(content, SYMPTOMS_HEADING),
            'causes': read_section_links(content, CAUSES_HEADING),
        }
    return problems


def find_cycles(edges, limit=25):
    """Return up to `limit` simple cycles found by depth-first search."""
    successors = defaultdict(list)
    for source, target in edges:
        successors[source].append(target)

    cycles = []
    colour = {}

    def walk(node, path):
        if len(cycles) >= limit:
            return
        colour[node] = 'open'
        path.append(node)
        for nxt in successors[node]:
            if colour.get(nxt) == 'open':
                cycles.append(path[path.index(nxt):] + [nxt])
                if len(cycles) >= limit:
                    break
            elif nxt not in colour:
                walk(nxt, path)
        path.pop()
        colour[node] = 'done'

    for node in list(successors):
        if node not in colour:
            walk(node, [])
    return cycles


def main():
    detail = '--detail' in sys.argv
    report_path = None
    if '--asymmetry-report' in sys.argv:
        report_path = sys.argv[sys.argv.index('--asymmetry-report') + 1]

    problems = load_problems()
    known = set(problems)

    forward = set()   # from Symptoms: this problem causes the linked one
    backward = set()  # from Causes: the linked one causes this problem
    dangling = []
    contradictions = []
    self_links = []

    for slug, data in problems.items():
        symptoms, causes = set(data['symptoms']), set(data['causes'])

        for target in symptoms | causes:
            if target not in known:
                dangling.append((slug, target))
        if slug in symptoms | causes:
            self_links.append(slug)
        for target in sorted(symptoms & causes):
            contradictions.append((slug, target))

        forward |= {(slug, t) for t in symptoms if t in known}
        backward |= {(t, slug) for t in causes if t in known}

    edges = forward | backward
    both_sides = forward & backward
    symptom_only = forward - backward
    cause_only = backward - forward

    out_degree = Counter(a for a, _ in edges)
    in_degree = Counter(b for _, b in edges)
    cycles = find_cycles(edges)

    print(f'Problems: {len(problems)}')
    print(f'Causal claims: {len(edges)}')
    print(f'  recorded on both sides: {len(both_sides)} '
          f'({len(both_sides) / len(edges) * 100:.1f}%)')
    print(f'  only under Symptoms of the cause: {len(symptom_only)}')
    print(f'  only under Causes of the effect:  {len(cause_only)}')
    print()
    print(f'Contradictions (cause and effect at once): {len(contradictions)}')
    print(f'Self links: {len(self_links)}')
    print(f'Links to missing problems: {len(dangling)}')
    print(f'Cycles found (up to {25}): {len(cycles)}')

    def median(counter):
        values = sorted(counter[s] for s in problems)
        return values[len(values) // 2] if values else 0

    out_hubs = [(s, c) for s, c in out_degree.most_common()
                if c > max(median(out_degree) * HUB_FACTOR, 1)]
    in_hubs = [(s, c) for s, c in in_degree.most_common()
               if c > max(median(in_degree) * HUB_FACTOR, 1)]
    print(f'Hubs above {HUB_FACTOR}x the median degree: '
          f'{len(out_hubs)} outgoing, {len(in_hubs)} incoming')

    if detail:
        for label, items in (
            ('Contradictions', [f'{a} <-> {b}' for a, b in contradictions]),
            ('Self links', self_links),
            ('Links to missing problems', [f'{a} -> {b}' for a, b in dangling]),
            ('Cycles', [' -> '.join(c) for c in cycles]),
            ('Outgoing hubs', [f'{problems[s]["title"]}: {c}' for s, c in out_hubs]),
            ('Incoming hubs', [f'{problems[s]["title"]}: {c}' for s, c in in_hubs]),
        ):
            if items:
                print(f'\n{label}:')
                for item in items:
                    print(f'  {item}')

    if report_path:
        lines = ['# Asymmetric causal claims', '',
                 'Each line is a claim recorded in one problem file but not the other.',
                 'Either the missing side should be added, or the claim is wrong and',
                 'should be removed from the side that has it.', '']
        lines.append(f'## Recorded only under Symptoms ({len(symptom_only)})')
        lines.append('')
        for cause, effect in sorted(symptom_only):
            lines.append(f'- `{cause}` claims `{effect}` as a symptom; '
                         f'`{effect}` does not list it as a cause')
        lines.append('')
        lines.append(f'## Recorded only under Causes ({len(cause_only)})')
        lines.append('')
        for cause, effect in sorted(cause_only):
            lines.append(f'- `{effect}` claims `{cause}` as a cause; '
                         f'`{cause}` does not list it as a symptom')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'\nWrote asymmetry report to {report_path}')

    return 1 if contradictions or self_links or dangling else 0


if __name__ == '__main__':
    sys.exit(main())
