Review causal relationships in `_problems/` for validity. Treat every relationship as
a hypothesis, including relationships recorded reciprocally in both problem files.
Reciprocity establishes structural consistency only; it is not independent evidence
that the relationship is true.

Review the scope specified in `$ARGUMENTS`. The scope may be a problem slug, a list
of slugs, a category, or the whole catalog. If no scope is supplied, start with a
small batch of high-degree problems reported by
`python scripts/validate_causal_links.py --detail`, because incorrect hub links have
the greatest effect on the graph. State the selected scope before reviewing it.

For each unique directed claim `A -> B`, inspect both files and record whether its
representation is:

- **Reciprocal**: A lists B under Symptoms and B lists A under Causes.
- **One-sided**: only one of those entries exists.

Keep that structural state separate from causal validity. Assess the claim itself by
checking:

1. **Meaning**: Do A and B describe sufficiently specific, distinct problems?
2. **Direction**: Would A normally precede B, or is the direction reversed?
3. **Mechanism**: Is there a clear explanation of how A can contribute to B?
4. **Alternatives**: Could correlation, a shared cause, selection bias, or a feedback
   loop explain the apparent relationship?
5. **Scope**: Under which technical, organizational, or process conditions should
   the relationship hold? Do not imply that a context-dependent claim is universal.
6. **Evidence**: Is there applicable support independent of this catalog, such as an
   empirical study, an authoritative industry source, a documented incident or
   postmortem, or observations supplied by a domain expert? Cite the exact source.
   AI-generated prose, semantic similarity, and repetition in two catalog files are
   not evidence.

Use one of these causal verdicts:

- **Supported**: credible evidence and a plausible mechanism support the qualified
  claim.
- **Plausible but unverified**: the mechanism is credible, but adequate independent
  evidence was not found.
- **Context-dependent**: defensible only when explicitly stated conditions apply.
- **Unsupported**: no adequate mechanism or evidence supports keeping the claim.
- **Contradicted or wrong direction**: evidence opposes the claim or supports a
  materially different direction.

For every claim, report: source problem, target problem, structural state, causal
verdict, mechanism, scope/qualifications, evidence with links, alternatives
considered, and a proposed action (`retain`, `qualify`, `reverse`, or `remove`). Do
not use missing evidence alone as proof that a claim is false.

Present the review and proposed changes before editing any problem files. After the
user approves the changes:

- Apply each accepted relationship consistently to both problem files.
- Preserve a concise mechanism and any essential qualification in the relationship
  descriptions.
- Remove or reverse both representations of rejected claims as applicable.
- Do not add relationships merely because two problems are semantically similar.
- Run `python scripts/validate_causal_links.py --detail` and
  `python scripts/check_links.py` after editing.
- Report structural validation separately from the evidence-based review result.
