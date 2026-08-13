---
title: Parallel Run
description: Run the old and new implementations side by side on real traffic, compare their outputs, and only cut over once the differences are understood.
category:
- Architecture
- Testing
- Operations
problems:
- strangler-fig-pattern-failures
- fear-of-breaking-changes
- legacy-business-logic-extraction-difficulty
- history-of-failed-changes
- data-migration-complexities
- data-migration-integrity-issues
- regression-bugs
- hidden-side-effects
- insufficient-testing
- legacy-code-without-tests
- second-system-effect
- high-defect-rate-in-production
- schema-evolution-paralysis
- maintenance-paralysis
- release-anxiety
- entity-attribute-value-overuse
- retention-obligations-block-change
- upgrade-blocked-by-customization
layout: solution
---

## Description

A parallel run executes the replacement implementation alongside the one it is meant to replace, feeding both the same real inputs and comparing their outputs, while only the original's results are used. It is the strongest available answer to the question that blocks most legacy replacements: how do we know the new one behaves the same as the old one? Tests answer it only for cases someone thought of, and the cases that matter in a decades-old system are the ones nobody thought of — the customer record with a null in a field that was mandatory after 2004, the settlement type used by one client. Production traffic contains those cases and a test suite does not. The parallel run turns the cutover from a decision made on the strength of a test suite into one made on the strength of observed agreement over real data.

## How to Apply ◆

> The value of the technique is proportional to how little you understand the original, which makes it most appropriate exactly where the risk is highest.

- **Route real inputs to both implementations** while using only the original's output. The new path must have no side effects during the comparison period: no writes to shared tables, no messages published, no external calls. Getting this isolation wrong is the main way a parallel run causes the incident it was meant to prevent.
- **Compare outputs automatically and record every difference** with enough context to reproduce it — the input, both outputs, and a timestamp. Manual comparison does not scale past the first few hundred cases, and the interesting differences are rare.
- **Categorize differences rather than counting them.** A thousand discrepancies from one rounding rule is one finding; three discrepancies from three distinct causes is three. Progress is measured by categories closed, not by discrepancy rate.
- Expect that **some differences are bugs in the original**, and decide deliberately for each: reproduce the old behavior because consumers depend on it, or fix it and inform those consumers. Recording this decision is important, because the next person will otherwise read the deliberate reproduction of a bug as a mistake.
- **Run long enough to cover the business cycle.** Monthly and quarterly processing paths do not appear in a week. For financial and billing systems this usually means at least one full month-end, and often a quarter-end, before agreement means anything.
- **Watch the cost.** Doubling the computation is affordable for most request-response work and can be prohibitive for heavy batch processing. Where it is, sample — a consistent percentage of traffic, plus deliberate oversampling of rare input types, which are where the differences concentrate.
- **Cut over progressively rather than all at once** when the categories are closed: shift a small share of traffic to the new implementation as the authority, monitor, and increase. Keep the comparison running during this phase, with the roles reversed.
- **Keep the ability to revert** until well after full cutover. The differences that survive a parallel run are the ones that appear at frequencies the observation window did not cover, and they surface weeks later.
- **Remove the old implementation deliberately**, on a scheduled date. Parallel runs that are never ended leave two implementations to maintain, which is worse than the situation before the migration started.

## Tradeoffs ⇄

> A parallel run gives evidence of equivalence that no other technique provides, in exchange for real infrastructure work, doubled processing cost, and a longer timeline before any benefit is realized.

**Benefits:**

- Equivalence is demonstrated against actual production data, including the rare cases that constitute most of the risk and none of the test coverage.
- The cutover decision becomes evidential rather than a leap of faith, which is usually what unblocks a replacement that has stalled on justified fear.
- Defects in the original are discovered as a byproduct, frequently including ones that have been silently causing damage for years.
- The undocumented behavior of the original is captured concretely, which serves as the specification the replacement never had.
- Confidence accumulates visibly over time, which makes the effort defensible to stakeholders during the long period when it produces no user-facing change.

**Costs and Risks:**

- Everything is computed twice, which for heavy workloads can be expensive or infeasible without sampling.
- Building the routing, comparison, and reporting infrastructure is genuine work that delivers nothing by itself, and it must be built before any comparison data exists.
- Side effects in the shadow path cause exactly the production incident the technique exists to avoid; isolation must be verified rather than assumed.
- Differences can be numerous enough to be demoralizing, and teams sometimes lower their standard for acceptable divergence in order to finish.
- Maintaining two implementations during the run doubles the cost of any change made in the meantime, which creates pressure to freeze features and pressure to cut the run short.

## How It Could Be

A bank replaced a fee calculation engine that had accumulated rules over eighteen years, with no specification and no surviving author. The team built the replacement from the code, then ran it in shadow against production traffic for eleven weeks. The first week produced discrepancies on 6.2 percent of transactions, resolving into nine categories. Seven were defects in the new implementation. One was a rounding difference that turned out to be a bug in the original, present since 2013, which had been systematically undercharging one product by fractions of a cent — the accumulated amount was material enough to require a disclosure. The ninth category appeared only at month-end and involved a fee waiver applied to accounts closed mid-cycle, a rule that existed in no document and that the business side confirmed was intentional. Cutover happened at 0.02 percent divergence, all in categories deliberately accepted, and produced no incidents.

A second team applied the technique to a data migration rather than a computation. Rather than a single cutover weekend, they wrote every record change to both the old and new schemas for four months, with a nightly job comparing the two and reporting mismatches by table. The comparison surfaced three classes of problem no dry run had found: a character encoding difference affecting names with diacritics, a timezone assumption in one date column, and a set of records the old system's application layer could produce but its own schema constraints technically forbade. All three would have been discovered during a cutover weekend, at three in the morning, with a rollback deadline. Instead each was fixed during normal working hours, and the eventual cutover consisted of changing which schema the application read from.
