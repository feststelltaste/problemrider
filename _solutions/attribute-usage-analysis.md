---
title: Attribute Usage Analysis
description: Measure which attributes are actually populated, queried, and varied,
  so that a generic data model can be replaced by evidence rather than by guesswork.
category:
- Database
- Code
- Process
problems:
- entity-attribute-value-overuse
- database-schema-design-problems
- excessive-customization
- schema-evolution-paralysis
- slow-database-queries
- difficult-to-understand-code
- high-technical-debt
- invisible-nature-of-technical-debt
- modernization-strategy-paralysis
- inadequate-requirements-gathering
- authorization-role-explosion
- custom-report-sprawl
- low-code-customization-sprawl
layout: solution
related_solutions:
- slug: typed-schema-extraction
  similarity: 0.7
- slug: feature-usage-measurement
  similarity: 0.65
- slug: role-model-rationalization
  similarity: 0.6
- slug: production-like-test-data
  similarity: 0.6
- slug: customization-cost-attribution
  similarity: 0.6
- slug: change-impact-analysis
  similarity: 0.6
---

## Description

Attribute usage analysis establishes, from the data itself, which attributes in a generic model are actually in use: how many entities have each one populated, how many distinct values each takes, whether anything reads it, and when it was last written. It exists because a generic model conceals its own structure — the schema is in the data, and nobody has looked at it. The consequence is that every discussion about replacing such a model stalls on the belief that the flexibility is needed, which nobody can confirm or refute. The analysis nearly always finds the same shape: a small number of attributes populated on almost every entity, which are the real structure being stored generically for no remaining reason, and a very long tail that is sparse, stale, or dead. That distribution is what makes an incremental fix possible.

## How to Apply ◆

> A generic model that has been in production for years is a record of what the organization actually needed, and it has never been read as one.

- **Start with population counts.** For each attribute name, how many entities have it set, as a share of the total. This single query usually splits the attributes into an obvious head and tail and takes minutes to run.
- **Count distinct values per attribute.** An attribute with one distinct value across a million entities is a constant. One with three is an enumeration that was never modelled. One with a million is free text. Each implies a different target design.
- **Check when each attribute was last written**, if the model carries timestamps. Attributes that have not been written in years are dead structure, and identifying them is the cheapest reduction available.
- **Find out what reads each attribute**, not only what writes it. Application code search, query logs, and reporting definitions together give a usable picture. An attribute written by an import and read by nothing is a candidate for deletion rather than migration.
- **Sample the values against their intended type.** Count how many entries in an attribute expected to be numeric, or a date, cannot be parsed as one. This produces the data quality finding that motivates the work, and it is usually worse than anyone expects.
- **Look for the same concept under several names.** Vocabulary drifts in a model nobody governs, and consolidating synonyms is often a large reduction in apparent complexity for very little effort.
- **Cross the usage with the customers or tenants that populate it**, where the model is multi-tenant. An attribute used by one tenant is a customization; one used by all of them is product structure.
- **Publish the distribution, not just the conclusion.** A chart showing 31 attributes above 90 percent population and 700 below one percent is a more persuasive argument for change than any description, and it is checkable.
- **Re-run it after each change** to confirm the tail is shrinking rather than being replaced by new tail.

## Tradeoffs ⇄

> The analysis is cheap and turns an unarguable design debate into an evidence-based one, but usage is not the same as importance and the data can mislead.

**Benefits:**

- The real structure hidden in the generic model becomes visible, which is the precondition for replacing any part of it.
- Dead and near-dead attributes are identified, and removing them is usually the largest and cheapest reduction available.
- The argument about whether the flexibility is needed becomes empirical, which is what unblocks a discussion that otherwise runs on assertion.
- Data quality problems surface with a count attached, which converts a suspicion into a finding somebody has to respond to.
- Distinguishing tenant-specific from universal attributes separates the customization problem from the data model problem, which need different responses.

**Costs and Risks:**

- Low usage is not low importance. A rarely populated attribute may be regulatory, contractual, or essential to one high-value customer, and deleting on frequency alone is dangerous.
- Attributes used only at long intervals — annual processes, year-end reporting — can look dead within any short observation window.
- Establishing what reads an attribute is genuinely hard where access is dynamic, generated, or comes from reporting tools outside the codebase.
- Running the analysis against production-scale data can be expensive, and against a sample it can miss exactly the rare attributes that matter.
- The findings can be used to justify removing flexibility that a future requirement will need, and that judgement is not in the data.

## How It Could Be

A team maintaining an order management system had argued for three years about whether their attribute-based product model could be replaced. The flexibility was said to be essential. Two days of analysis settled it. Of 430 distinct attribute names, 24 were populated on over 95 percent of orders — these were the order structure, stored generically since a decision made in 2013 for reasons nobody could reconstruct. A further 60 were populated on between 1 and 20 percent, concentrated by product line. The remaining 346 were below one percent, and 190 of those had not been written since 2020. The distribution was published as a single chart, and the three-year argument ended in one meeting.

The type sampling produced the finding that actually got the work funded. One attribute holding a monetary amount contained 1.4 million values, of which about 2,600 could not be parsed as a number: some had currency symbols, some used a comma as decimal separator, and 340 were empty strings. Downstream code handled the failures by treating the value as zero, silently. The team could not determine how long this had been happening or what it had cost, and the inability to answer that question was itself the most persuasive part of the report.
