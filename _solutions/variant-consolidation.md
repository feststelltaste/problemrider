---
title: Variant Consolidation
description: Periodically fold the variants that several customers share back into
  the standard product, and retire the ones nothing depends on.
category:
- Architecture
- Business
- Process
problems:
- excessive-customization
- high-maintenance-costs
- code-duplication
- testing-complexity
- long-release-cycles
- increased-cost-of-development
- high-technical-debt
- slow-feature-development
- maintenance-cost-increase
- feature-creep
- technology-stack-fragmentation
- entity-attribute-value-overuse
- core-modification-of-standard-software
- custom-report-sprawl
- reimplemented-standard-functionality
- upgrade-blocked-by-customization
layout: solution
related_solutions:
- slug: customization-cost-attribution
  similarity: 0.8
- slug: explicit-extension-points
  similarity: 0.7
- slug: fit-to-standard-principle
  similarity: 0.7
- slug: feature-usage-measurement
  similarity: 0.65
- slug: consistent-terminology
  similarity: 0.65
- slug: large-scale-refactoring
  similarity: 0.6
---

## Description

Variant consolidation is the recurring practice of examining the accumulated customer-specific variants, folding the ones that several customers effectively share back into the standard product, and retiring the ones nothing depends on any more. It is the reduction half of managing a customizable product; extension points prevent new variation from spreading into the core, but they do nothing about what has already accumulated. Consolidation matters because customization portfolios have a characteristic shape: a set of variants that look distinct because they were requested separately and named after their requesters, but which do substantially the same thing, plus a substantial number that are dead — the requesting customer left, the process changed, or the need was temporary. Neither group is visible without someone deliberately looking, and nothing in normal operation ever prompts anyone to look.

## How to Apply ◆

> Variants are named after who asked for them, which is precisely what conceals that three of them are the same thing.

- **Review on a fixed cadence**, annually or semi-annually. Consolidation never happens opportunistically, because at no individual moment is it the most urgent thing available.
- **Group by what the variant does**, not by who requested it. Describing each variant in one sentence of behavior, with the customer name removed, is usually enough to reveal that several are the same. This step alone frequently halves the apparent count.
- **Identify the dead ones** using evidence: is the requesting customer still a customer, is the code path still executed, has the variant been touched in years. Dead variants are the cheapest reduction and require no negotiation with anyone.
- **Generalize the shared ones into the standard product** where several customers want substantially the same behavior. The generalization is usually a configuration option rather than a new variant, and it removes several maintenance burdens at once.
- **Negotiate the near-misses.** Where three variants differ only slightly, the difference is frequently negotiable — customers accepted a variant because it was offered, not because the specific detail mattered. Asking is cheap and often works.
- **Retire deliberately, with notice and a migration path.** A variant removed without warning is a support incident and a trust problem, and the resulting reputation makes the next consolidation harder.
- **Feed the cost figures into the conversation.** A customer asked to move to the standard behavior responds differently when the request is accompanied by what their variant costs and what they get in return — usually faster upgrades and quicker access to new capability.
- **Accept the ones that must stay.** Some variants are genuinely required by regulation, contract, or a customer's real process. Naming them as permanent, with a reason, closes the review rather than leaving them as perpetual candidates.
- **Record the reduction.** Variants retired, variants generalized, and the estimated maintenance released. Consolidation produces no visible feature, so its results have to be reported deliberately or the next review will not be funded.

## Tradeoffs ⇄

> Consolidation is the only thing that reduces an accumulated variant portfolio, but it requires customer negotiation and produces nothing customers asked for.

**Benefits:**

- The number of variants falls absolutely, which reduces test matrix size, upgrade cost, and the burden on every future change.
- Generalizing shared behavior into the product converts several maintenance liabilities into one supported feature that all customers get.
- Dead variants are found and removed at essentially no cost beyond the looking, and they are usually a substantial share of the portfolio.
- Customers moved onto standard behavior receive upgrades and new capability faster, which is a genuine benefit and makes the negotiation possible.
- Regular review prevents the portfolio from growing monotonically, which is what it does when nobody examines it.

**Costs and Risks:**

- It requires customer conversations that account management may resist, particularly where the relationship is fragile.
- Retiring a variant that turns out to be load-bearing for a customer's process is a serious incident and damages trust broadly, not just with that customer.
- Generalization can produce a configuration surface that is itself complex — several variants replaced by one feature with eight options is not obviously an improvement.
- The work delivers nothing any customer asked for, which makes it difficult to prioritize against requests that someone is waiting for.
- Near-miss negotiation takes time per customer and frequently fails, so the effort is spent whether or not the consolidation happens.

## How It Could Be

A vendor reviewed 47 customer variants for the first time in six years, describing each in one sentence with the customer name stripped. Eleven turned out to be three groups doing the same thing: four ways of suppressing a confirmation email, four ways of adding a reference code to an invoice, and three ways of rounding a total. Each group was generalized into a single configurable feature over about three weeks. Nine further variants were dead — six requesting customers had left, two related to a process the customer had since abandoned, and one had never been executed in production according to the logs. Those were removed with a notification and no objection. The portfolio fell from 47 to 30 in a quarter, and the release validation matrix shrank correspondingly.

The negotiation on near-misses was more mixed and more instructive. Six variants differed only in a date format on a printed document. The vendor approached all six customers offering the standard format. Four accepted immediately, one accepted in exchange for a different small change they had been wanting, and one refused on the grounds that their downstream system parsed the document — which turned out to be true and important, and that variant was recorded as permanent with the reason attached. The vendor's account team, initially opposed to approaching customers about removing things, reported afterwards that five of six conversations had been positive, because the offer had been framed as faster upgrades rather than as a withdrawal.
