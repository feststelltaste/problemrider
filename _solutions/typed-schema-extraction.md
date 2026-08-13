---
title: Typed Schema Extraction
description: Promote the attributes that are genuinely part of the domain into typed,
  constrained columns, and leave only the sparse remainder generic.
category:
- Database
- Architecture
- Code
problems:
- entity-attribute-value-overuse
- database-schema-design-problems
- slow-database-queries
- high-number-of-database-queries
- n-plus-one-query-problem
- difficult-to-understand-code
- increased-bug-count
- schema-evolution-paralysis
- testing-complexity
- imperative-data-fetching-logic
- high-technical-debt
- data-migration-integrity-issues
layout: solution
related_solutions:
- slug: attribute-usage-analysis
  similarity: 0.7
- slug: evolutionary-database-design
  similarity: 0.7
- slug: query-optimization-process
  similarity: 0.6
- slug: nosql-databases
  similarity: 0.6
- slug: index-lifecycle-management
  similarity: 0.6
- slug: production-like-test-data
  similarity: 0.6
---

## Description

Typed schema extraction moves the attributes that are actually part of the domain out of a generic attribute store and into real columns with real types and real constraints, while leaving genuinely sparse or genuinely unpredictable data in the flexible form. It is the incremental answer to a generic data model, and it works because such models are almost never uniformly generic. They contain a stable core that everyone uses — which gains nothing from being stored generically and loses type safety, constraints, indexing, and legibility — surrounded by a long tail where the flexibility is doing real work. Attempting to eliminate the generic model entirely fails, because the tail has genuine variety. Attempting to leave it alone fails, because the core is where the cost is. Extraction takes the core and stops.

## How to Apply ◆

> The attributes worth extracting identify themselves: they are populated on nearly every entity, they are filtered and sorted on, and they have an obvious type.

- **Select candidates from evidence**, not intuition: high population share, small number of distinct values or a clear type, and appearance in query filters. An attribute set on 98 percent of entities and used in every search is the archetype.
- **Extract in small groups**, not all at once. Three or four related attributes forming one concept is a manageable increment that can be verified and shipped independently.
- **Add the typed columns first and write to both**, keeping the generic rows as the source of truth. Nothing reads the new columns yet, so nothing can break, and the dual write establishes whether the data can even be typed.
- **Let the migration reveal the data quality.** Converting text values into a typed column will fail on the entries that were never valid, and those failures are findings that must be decided on rather than silently coerced. Expect this step to take longer than the schema change.
- **Move readers over one at a time**, verifying each against the generic path before switching. Comparing the two paths on live traffic is what makes the cutover a measurement rather than a leap.
- **Add the constraints once readers have moved**: not null where the data supports it, foreign keys where a relationship exists, checks where the value range is known. The constraints are most of the benefit, and adding them is the step that gets postponed.
- **Stop writing the generic rows, then delete them**, on a stated date. Leaving both paths populated indefinitely means carrying two models and gaining nothing, which is the most common way this work ends.
- **Keep the tail generic and say so explicitly.** Documenting which data legitimately remains flexible, and why, prevents the next developer from either extending the generic model back into the core or attempting to eliminate it entirely.
- **Consider a structured document column for the tail** where the database supports one. It keeps flexibility while allowing validation and indexing of known paths, which is strictly better than untyped attribute rows for most remaining cases.
- **Re-run the usage analysis afterwards** to confirm the tail is not regrowing, and pair it with a rule that new fields go into the typed model unless there is a stated reason.

## Tradeoffs ⇄

> Extraction recovers what the database is for — types, constraints, indexes, legibility — at the cost of a careful migration and the loss of the ability to add a field without a schema change.

**Benefits:**

- Queries become efficient and indexable, and reconstructing an entity stops requiring a join or pivot across many rows.
- The database enforces correctness again, so a whole class of defect is rejected at write time rather than surfacing far away at read time.
- The domain becomes visible in the schema, which is a substantial improvement in legibility for anyone new to the system.
- Reporting can run against the real model rather than a separately maintained flattened copy, removing that copy and its drift.
- The migration itself surfaces accumulated data quality problems that the untyped column had been concealing, often for years.

**Costs and Risks:**

- Adding a field now requires a schema change, which is exactly the friction that produced the generic model — so this work is wasted unless schema changes have also been made routine.
- The migration will find data that cannot be typed, and each case is a decision requiring domain knowledge that may no longer exist in the organization.
- Dual-write periods carry the risk of the two representations diverging, and the reconciliation is real work.
- Extracting too much removes flexibility the tail genuinely needs, and re-introducing it later is harder than having left it alone.
- The final deletion of the generic rows is easy to defer, and deferring it means the organization carries both models permanently.

## How It Could Be

A product catalogue stored every attribute generically: 380 attribute names, of which usage analysis showed 22 were populated on more than 95 percent of products and appeared in nearly every search filter. Those 22 — name, category, price, currency, status, dimensions, and a handful of others — were the catalogue's actual structure. The team extracted them in five increments over a quarter, dual-writing each group and comparing the two paths on production traffic before moving readers. Product search latency fell from a median of 1.9 seconds to 90 milliseconds, because the search could finally use indexes. The remaining 358 attributes stayed generic, which was correct: they were product-type-specific properties that genuinely varied, and no typed schema would have accommodated them.

The typing step was where the real work turned out to be. Converting the price attribute to a numeric column failed on 4,100 of 2.3 million products. Investigation found four distinct causes: two legacy import formats, a period when a defect had written the currency into the price field, and roughly 900 products whose price had genuinely been entered as a text range because the business had no other way to express "price on request." The first three were data errors and were corrected. The fourth was a real requirement that nobody had ever modelled, and it became a separate nullable field with an explicit flag — a domain concept that had spent six years hiding inside an untyped column.
