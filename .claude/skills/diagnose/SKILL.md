---
name: diagnose
description: Use when a user describes a real problem, symptom, or messy situation in their own legacy software system and wants help finding the root cause and possible fixes — e.g. "our deployments take forever and nobody knows why", "why is this system so hard to change", "our onboarding takes months, what's actually wrong". Runs a guided root-cause diagnosis against this repo's _problems/ and _solutions/ catalog using Walter Schönwandt's "Komplexe Probleme lösen" method, checking for problem back-/forward-shifting and common thinking errors before recommending solutions. Do not use for maintaining the catalog itself (creating/linking/categorizing problem entries) — that's pr:add_tech_debt, pr:link_problems, pr:refine_categories, pr:generate_new_problems, pr:deduplicated.
---

**Status: Experimental.** This skill is new and its process is still being refined — treat the structure below as a working draft, not a finished methodology.

Help the user diagnose the root cause(s) of a problem in their own (real, external) legacy system and point them to solutions from this catalog.

This follows Walter Schönwandt's method from *Komplexe Probleme lösen* (German: "Solving Complex Problems") — the "Key Seven" ("sieben Themen"), as explained in his interview on the *Abenteuer Problemlösen* podcast, episode 21. Act as a diagnostic partner working through these themes with the user, not a search engine that matches keywords from a single sentence.

**This method is for genuinely tricky problems** — the kind where no ready-made routine solution applies — if the user's situation already has an obvious, known fix, just point them at it directly instead of running the full process.

## Which causal links you can rely on

The **Symptoms** and **Causes** sections of `_problems/*.md` are the causal graph you
will be walking. They are not uniformly trustworthy, and the difference matters for
this method specifically.

Both sections express the same kind of directed claim from opposite ends: an entry
under **Symptoms** of problem A claims *A causes B*, and an entry under **Causes** of
problem B claims the same thing. So a claim should appear in both files. Currently
only about one in seven does. The rest were written into one file without the other
file ever being updated.

The consequence is structural, and it breaks root-cause analysis if you ignore it:

- Taking **all** links at face value, every one of the catalog's problems both causes
  and is caused by every other. There is not a single problem that nothing causes.
  Backward shifting on this graph never terminates — you can walk from any problem to
  any other and back.
- Taking only the links **both files record**, the graph falls apart into many small
  pieces with real root causes, real terminal symptoms, and small feedback loops.

So when you traverse:

- A link recorded in **both** files is confirmed. Treat it as evidence.
- A link recorded in **only one** file is a lead, not evidence. Follow it if it looks
  relevant, but say so, and do not let an unconfirmed link carry a conclusion on its
  own — especially not the claim that you have reached a root cause.
- If a chain you want to present rests mainly on unconfirmed links, say that plainly
  in the report rather than presenting it with the same confidence as a confirmed one.

To check a specific link, open both problem files and look for the counterpart entry.
`python scripts/validate_causal_links.py --detail` reports the overall state, and
`--asymmetry-report FILE` writes out every unconfirmed claim.

Small feedback loops in the confirmed graph are expected and are not a defect —
reinforcing loops are a real feature of legacy systems. Do not treat a loop as a
reason to distrust a chain; treat it as something to name.

## The core discipline: iterate, don't pipeline

The seven themes below are not a linear checklist. Schönwandt is explicit that you go through them repeatedly, in whatever order makes sense, until they're mutually consistent (reflective equilibrium). You can start from a suspected cause, a proposed measure, or the original complaint — it doesn't matter, as long as you end up circling back to check the others. In particular:

- A cause-analysis that reveals a new understanding of the problem should send you back to **theme 1**.
- A **thinking error** (theme 5) found late should send you back to whichever theme it contaminated.
- **Theme 6** ("is this even the right problem?") isn't a final rubber stamp — run it as soon as you have a testable claim, and be ready to loop all the way back if it fails.

Never present a diagnosis as a single confirmed pass through the steps. Say when you're looping back and why.

## 1. Problem Understanding — what is the problem?

Get the user to state the unsatisfactory state of affairs itself — not a solution wearing a problem's clothes. Ask what they actually observe (symptoms, effects, complaints), not their own diagnosis of why it's happening.

Clarify terms as part of this, not as an afterthought: if the user uses a loaded or ambiguous term ("legacy", "technical debt", "slow", "outdated"), ask what they specifically mean by it — definitions steer the entire solution space, and two people can think they agree while meaning different things.

Only once the problem is stated in observable terms, look for a matching `_problems/*.md` entry via front matter (`title`, `description`, `category`) and the **Symptoms** / **Indicators** sections. Treat matches as hypotheses, not conclusions yet.

## 2. Problem Back-/Forward-Shifting — check for solution-space shifts

Before settling on the problem framing, deliberately try shifting it in both directions — this is a reframing technique to widen the options, not an error to avoid:

- **Backward shift**: "Where does this come from?" — step back toward the root. Would solving something further upstream make the named problem moot? (Schönwandt's example: not "we lack landfill sites" but "we produce too much waste" → a recycling law removed the original problem entirely.)
- **Forward shift**: "What does this lead to?" — accept the named cause as a given, and ask what it leads to instead. Can the downstream effect be handled directly? (Example: not "we emit too much CO2" but "we don't use the CO2 we already emit" → capture, storage, or reuse.)

When you shift backward, walk the **Causes** links; when you shift forward, walk the
**Symptoms** links. Prefer confirmed links in both directions. A backward walk that
only continues because of unconfirmed links has probably reached the end of what the
catalog actually supports — stop there and say so, rather than continuing until the
scope boundary stops you.

In this catalog, the scope section of `CLAUDE.md` already fixes how far each direction is allowed to go: not further back than the requirements/management level (political or economic root causes like tariff wars are out of scope), and not further forward than fine-grained technical detail (CPU-level issues are out of scope). If the user's stated problem sits outside that band, that's exactly this theme at work — help them find the backward- or forward-shifted framing that lands back inside it, and note explicitly which direction you shifted and why.

## 3. Problem Causes (plural)

Ask "What else could it be?" and keep generating candidate causes past the first plausible one — settling for a single cause too early is the specific thinking error Schönwandt calls "monocausalitis" (assuming one plausible cause is already good work). Match candidates against the **Causes** sections of `_problems/*.md`, and check whether something the user named as "the problem" is actually a **Symptom** of a different, deeper entry.

Check each candidate cause against the counterpart file before relying on it. If the
deeper entry does not list the user's problem among its **Symptoms**, the link is
unconfirmed — which is common and does not make it wrong, but it does mean the
catalog is not evidence for it. Say which of your candidate causes are confirmed and
which rest on a one-sided link, and do not let the strongest conclusion rest on the
weakest link.

Also generate candidates the catalog does not offer. A graph in which everything links
to everything invites the hammer-and-nail bias of theme 5: whatever you look at will
have plenty of plausible neighbours.

## 4. Fitting Measures

For each named cause, the measure should target that cause specifically enough that the correspondence is checkable — not a vague "fix it". Schönwandt's fare-dodging example: catch-probability too low → more inspectors; fines too lenient → raise fines; tickets unaffordable → subsidize/lower prices; too little social stigma → awareness campaign; ticket machines unusable → redesign the machines. Different assumed causes imply different, non-interchangeable measures.

Resolve confirmed causes to solutions via the problem's `solutions:` front matter into `_solutions/*.md`. Summarize each one's `description`, key points from **How to Apply**, and relevant **Tradeoffs** — check that the solution's actual mechanism targets the cause you settled on, not just a plausible-sounding but mismatched one.

## 5. Thinking Errors — including your own professional lens

Before treating the analysis as settled, check it against:

- **Monocausalitis** — did we actually consider more than one cause, or stop at the first one that sounded right?
- **Solution disguised as problem** — does the problem statement already smuggle in a particular fix?
- **The hammer-and-nail bias** ("if my only tool is a hammer, every problem looks like a nail") — is this diagnosis reaching for the categories/solutions most familiar or available in this catalog, rather than what actually fits the user's situation?
- Anything else the user's own checklist includes — ask if they have additional thinking-error checks they want applied.

If you find one, name it plainly and re-run whichever theme it affected — don't quietly patch the conclusion.

## 6. Is This the Right Problem? — does the thesis actually hold?

Explicitly sanity-check the problem statement against evidence before finalizing anything, as its own step, not just implicitly. Schönwandt's example: a city assumed poorer districts had the highest crime rate; on closer inspection, wealthier districts scored higher once "crime" was defined to include things like tax evasion — the original thesis didn't survive scrutiny, which invalidated the work built on top of it. Ask the user directly: does this framing hold up against what you actually know, or only against what seemed obvious at first? If it fails, loop back to theme 1 with the corrected framing — that loop-back is the process working, not a setback.

## Causal chains are black boxes — make them explicit instead of certain

You will never fully map the true cause-effect network behind a real legacy system, and that's fine — the goal is to turn the black box into a gray one: write the assumed chain down (which causes, which problems, which measures) so it's discussable and falsifiable, rather than leaving it implicit. Every measure anyone proposes already assumes some causal chain; the difference this method makes is surfacing it.

## Handle catalog gaps honestly

If something the user describes doesn't match any existing `_problems/` entry, say so plainly instead of forcing a weak match. Ask whether they'd like a new problem entry drafted, following `problem-pattern-template.md` and the Guiding Principle in `CLAUDE.md`.

If they say yes: write it as a general, reusable *pattern* (like every other file in `_problems/`), not case notes — strip out anything specific to their company, product, or people. Follow the title-case and linking rules from `CLAUDE.md`, and link it to related existing problems where relevant.

## Write a takeaway report

Once the user is satisfied the loop has converged (themes are mutually consistent, theme 6 held up), write a summary to `diagnoses/<slug>-<YYYY-MM-DD>.md` (create the `diagnoses/` directory if missing — it's git-ignored, not catalog content) containing:

- **Reported Situation** — the problem as finally framed, in observable terms
- **Back-/Forward-Shifting Considered** — which direction(s) you tried, and where the framing landed
- **Root Causes** — linked problem titles with a one-line rationale each, and for each
  one whether the link to it is confirmed by both problem files or rests on one side
- **Recommended Solutions** — linked solution titles with a one-line rationale each
- **Thinking Errors Checked** — what you checked for, and anything it changed
- **Validity Check** — how theme 6 was tested and what held up
- **Open Gaps** — anything that didn't match an existing entry, and whether a new problem was drafted for it

Keep the chat response itself concise — the report file is where the detail lives; don't paste the whole report back into the chat.
