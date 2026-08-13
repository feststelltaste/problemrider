---
title: Exploratory Testing
description: Have a skilled person investigate the system deliberately and without
  a script, in timeboxed sessions with recorded findings, to discover what nobody
  thought to specify.
category:
- Testing
- Process
problems:
- insufficient-testing
- poor-test-coverage
- high-defect-rate-in-production
- missing-end-to-end-tests
- testing-complexity
- increased-manual-testing-effort
- regression-bugs
- reduced-feature-quality
- hidden-side-effects
- requirements-ambiguity
- inadequate-requirements-gathering
- quality-degradation
- cache-invalidation-problems
- deadlock-conditions
- improper-event-listener-management
- increased-risk-of-bugs
- negative-brand-perception
- partial-bug-fixes
- stack-overflow-errors
- unreleased-resources
- user-trust-erosion
layout: solution
related_solutions:
- slug: test-coverage-strategy
  similarity: 0.75
- slug: usability-tests
  similarity: 0.75
- slug: characterization-tests
  similarity: 0.75
- slug: fuzz-testing
  similarity: 0.7
- slug: regression-testing
  similarity: 0.7
- slug: code-reading-sessions
  similarity: 0.7
---

## Description

Exploratory testing is structured investigation of a system by a person who is simultaneously designing and executing tests, learning from each result what to try next. It is not ad-hoc clicking, and it is not a substitute for automated tests. It occupies a gap that automation cannot fill by construction: an automated test can only check something someone already thought of, while the defects that matter in a legacy system are disproportionately the ones nobody anticipated. A skilled explorer follows the system's actual behavior — an odd message, a slow response, a state that should not be reachable — down paths no specification describes. Legacy systems reward this unusually well, because they contain decades of accumulated behavior that no current document describes and that no test suite covers, so there is a great deal to find.

## How to Apply ◆

> The most productive exploratory sessions in a legacy system target the areas where nobody can tell you what the correct behavior is, because that is where nobody has been able to write a test.

- **Work in timeboxed sessions with a stated charter** — sixty to ninety minutes, with a one-sentence mission such as "investigate what happens to an order when the payment provider times out mid-transaction." An unbounded session becomes unfocused; a charter without a timebox becomes an investigation with no end.
- **Record what you did and what you found as you go**, including paths that produced nothing. The notes are what makes the session reproducible, reportable, and convertible into automated tests afterward.
- **Follow the system's cues rather than a plan.** A response that takes four seconds when others take fifty milliseconds, an error message mentioning a component that should not be involved, a field that accepts more characters than it should — these are the leads, and following them is the skill.
- **Vary deliberately along known fault lines**: boundaries, empty and maximal inputs, unusual sequences, interruption and resumption, concurrent access, and the back button. In legacy systems, sequences the designers did not anticipate are consistently the richest source.
- **Use production-like data.** Exploring against clean synthetic records finds far less, because the interesting behavior is triggered by the historical and malformed records that real systems contain.
- **Convert what you find into automated tests.** A defect found by exploration should not be findable by exploration a second time. The exploration finds the case; the automated test keeps it found.
- **Pick charters from risk**, not coverage: recently changed areas, code with no test coverage, integration points with external systems, and anything that has caused incidents before.
- **Have people other than the author explore.** The author's mental model is what produced the behavior, so they are structurally the least likely to find its blind spots. Pairing a developer with someone from support or operations is frequently very productive.
- **Report findings as observations with evidence**, not as verdicts. In a legacy system it is often genuinely unclear whether behavior is a defect or a long-standing intentional quirk, and the exploration's job is to surface it for someone to decide.
- **Schedule it regularly** rather than only before releases. Exploration used exclusively as a pre-release gate becomes a rushed regression check, which is the one thing it is worst at.

## Tradeoffs ⇄

> Exploration finds the defects automation structurally cannot, at the cost of skilled human time and results that are neither repeatable nor predictable.

**Benefits:**

- It finds defects that no automated suite would ever contain, because those tests would have had to be written by someone who already knew about the case.
- It surfaces undocumented behavior, which in a legacy system is a substantial fraction of the actual specification.
- It requires no test infrastructure, so it works in systems where automated testing is currently impractical — often the systems that need testing most.
- Findings convert directly into automated tests, so the practice builds a suite as a byproduct.
- It surfaces usability and coherence problems that pass every functional check, since a human notices confusion where an assertion does not.

**Costs and Risks:**

- It consumes skilled human time on every occasion and cannot be run automatically on every change, which makes it unsuitable as a regression mechanism.
- Results depend heavily on the explorer's skill and knowledge, so the practice is difficult to plan around or to guarantee.
- Coverage is unknown by construction. A session that finds nothing may mean the area is sound or that the explorer looked in the wrong place, and the two are indistinguishable.
- Findings can be ambiguous in a legacy system, and adjudicating whether long-standing odd behavior is a defect consumes time from people who may not know either.
- Without recorded notes and conversion to automated tests, the same defects are rediscovered repeatedly, and the practice acquires a reputation for producing effort rather than progress.

## How It Could Be

A team maintaining a hospital appointment system had 78 percent automated test coverage and a persistent rate of production defects that the suite never caught. They introduced weekly ninety-minute exploratory sessions with charters chosen from recently changed and historically troublesome areas. The third session's charter was "what happens when an appointment is rescheduled while a clinician is editing it." The explorer found that the second save silently overwrote the first, that the notification went to the original clinician rather than the current one, and that the audit trail recorded only one of the two changes. None of this was covered by any test, because the concurrent-edit scenario had never occurred to anyone who wrote tests. All three became automated tests once fixed. Over two quarters the sessions produced 61 findings, of which 34 were accepted as defects and 9 were classified as significant.

The classification ambiguity turned out to be informative in its own right. Fourteen findings were behaviors the team could not confidently call defects — an appointment type that permitted overlapping bookings, a cancellation window that behaved differently at month boundaries, a status transition that the state diagram forbade. Taking these to the clinical operations team revealed that eleven were intentional, undocumented accommodations of real clinical workflow, established years earlier and known only to long-serving staff. Those eleven were documented and turned into characterization tests, which protected them from being "fixed" by a future developer reading the state diagram. The remaining three were genuine defects that had been quietly producing scheduling problems for years.
