Inspect the **Symptoms** and **Causes** sections of `_problems/*.md` and connect
mentions to existing problem files. This command maintains the representation of the
causal graph; it does not establish that a causal relationship is true.

Treat these as separate operations:

1. **Link an existing claim**: When an existing Symptoms or Causes entry clearly
   names an existing problem, convert the complete canonical problem title into a
   relative Markdown link. This preserves a claim already present in the prose; it
   does not validate the claim.
2. **Add or make a claim reciprocal**: Adding a new relationship, or adding its
   missing representation in the counterpart file, changes the causal graph. Do this
   only after reviewing the claim using the causal checks below.
3. **Repair a broken link**: Correct the target when the intended existing problem is
   unambiguous. Do not silently redirect an ambiguous link to the nearest-sounding
   problem.

For every relationship that would be newly added or made reciprocal, assess:

- whether cause and effect are distinct and the direction is correct;
- whether the cause normally precedes the effect;
- the mechanism by which the cause can contribute to the effect;
- whether a shared cause, correlation, or feedback loop is a better explanation;
- the conditions under which the relationship applies; and
- whether evidence independent of this catalog supports it.

Use `/pr:review_causal_links` for a full evidence-based review. A reciprocal entry,
AI-generated prose, semantic similarity, or the fact that two problems commonly
co-occur is not evidence of causality.

When wording differs slightly, change only the mention in the Symptoms or Causes
entry to the existing problem's complete canonical title. The linked title must match
the target problem's title exactly, including capitalization. Do not rename a problem
or its file merely to force a match.

If you discover a potentially relevant relationship that is not already stated,
propose it with its direction, mechanism, qualification, and evidence. Do not add it
until the user approves it. Likewise, do not automatically add the counterpart of a
one-sided claim: report it as structurally one-sided and review it first.

If a `lack of <something>` link points to a nonexistent file, remove the link while
preserving useful prose. Leave other unresolved nonexistent targets in place and
report them, because a problem entry may still need to be created.

Choose a random problem file to start with unless the user supplies a scope in
`$ARGUMENTS`. Do not create a script for the semantic review; inspect the surrounding
text and both endpoint problem files.

After making approved changes, run:

- `python scripts/check_links.py`
- `python scripts/validate_causal_links.py --detail`

Report broken-link and structural results separately from causal-review results.
Passing either script does not prove that the relationships are causally valid.
