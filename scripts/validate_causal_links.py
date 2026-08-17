#!/usr/bin/env python3
"""
Check the causal graph formed by the Symptoms and Causes sections for structural defects.

This script checks representation and graph consistency only. Reciprocal entries are
two representations of one assertion, not independent evidence that it is causally
true.

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
  structure      how much of the graph collapses into one strongly connected
                 component. Small feedback loops are a real property of the
                 domain and are expected. A single component spanning the whole
                 catalog is not a loop, it means every problem both causes and
                 is caused by every other, so no problem can be a root cause
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

# Sections are matched by their trailing glyph, not the English heading word,
# so this also works on translated (e.g. German) problem files that keep the
# same glyphs but translate the heading text itself (see
# plans/german-translation-plan.md, Decision 2).
SYMPTOMS_GLYPH = '▲'
CAUSES_GLYPH = '▼'

# A problem this far above the median out- or in-degree is reported as a hub.
HUB_FACTOR = 6


def read_section_links(body, glyph):
    """Return the problem slugs linked in one section of a problem file."""
    match = re.search(
        rf'^## .*{re.escape(glyph)}\s*\n(.*?)(?=^## |\Z)',
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
            'symptoms': read_section_links(content, SYMPTOMS_GLYPH),
            'causes': read_section_links(content, CAUSES_GLYPH),
        }
    return problems


def strongly_connected_components(edges):
    """
    Return the strongly connected components of the causal graph, largest first.

    Counting cycles is not informative here: a depth-first search in a dense
    graph reports arbitrarily long ones and the count depends on traversal
    order. Components answer the question that matters — whether the graph
    still has a direction, or whether everything reaches everything.
    """
    successors = defaultdict(list)
    predecessors = defaultdict(list)
    nodes = set()
    for source, target in edges:
        successors[source].append(target)
        predecessors[target].append(source)
        nodes |= {source, target}

    visited = set()
    order = []
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        stack = [(start, iter(successors[start]))]
        while stack:
            node, following = stack[-1]
            for nxt in following:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append((nxt, iter(successors[nxt])))
                    break
            else:
                order.append(node)
                stack.pop()

    assigned = {}
    components = []
    for node in reversed(order):
        if node in assigned:
            continue
        component = []
        stack = [node]
        assigned[node] = len(components)
        while stack:
            current = stack.pop()
            component.append(current)
            for previous in predecessors[current]:
                if previous not in assigned:
                    assigned[previous] = len(components)
                    stack.append(previous)
        components.append(component)

    return sorted(components, key=len, reverse=True)


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

    def structure(edge_set):
        """Component count, largest component, and the roots and leaves left."""
        components = strongly_connected_components(edge_set)
        nodes = {n for edge in edge_set for n in edge}
        sources = {n for n in nodes if not any(b == n for _, b in edge_set)}
        sinks = {n for n in nodes if not any(a == n for a, _ in edge_set)}
        return components, nodes, sources, sinks

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

    for label, edge_set in (('all claims', edges),
                            ('reciprocal claims', both_sides)):
        components, nodes, sources, sinks = structure(edge_set)
        largest = len(components[0]) if components else 0
        share = largest / len(nodes) * 100 if nodes else 0
        print(f'Structure of {label}: {len(edge_set)} claims over {len(nodes)} problems')
        print(f'  components: {len(components)}, largest holds {largest} '
              f'({share:.0f}% of the problems involved)')
        print(f'  root causes (nothing causes them): {len(sources)}, '
              f'terminal symptoms (they cause nothing): {len(sinks)}')

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
            ('Components larger than one problem',
             [f'{len(c)}: ' + ', '.join(problems[n]['title'] for n in sorted(c)[:6])
              + ('...' if len(c) > 6 else '')
              for c in strongly_connected_components(edges) if len(c) > 1]),
            ('Outgoing hubs', [f'{problems[s]["title"]}: {c}' for s, c in out_hubs]),
            ('Incoming hubs', [f'{problems[s]["title"]}: {c}' for s, c in in_hubs]),
        ):
            if items:
                print(f'\n{label}:')
                for item in items:
                    print(f'  {item}')

    if report_path:
        lines = ['# Structurally One-Sided Causal Claims', '',
                 'Each line is a claim recorded in one problem file but not the other.',
                 'This report does not establish whether any claim is causally true.',
                 'Review the claim before adding the missing side or removing it.', '']
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
