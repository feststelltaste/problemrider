---
title: Entity-Attribute-Value Overuse
description: Business data is stored as generic attribute rows instead of typed columns, so the database can no longer enforce, index, or explain the data it holds.
category:
- Database
- Architecture
- Code
related_problems:
solutions:
- attribute-usage-analysis
- typed-schema-extraction
- data-modeling
- domain-driven-design
- evolutionary-database-design
- backward-compatible-schema-migrations
- parallel-run
- characterization-tests
- change-impact-analysis
- design-by-contract
- input-validation
- materialized-views
- cqrs
- index-lifecycle-management
- explicit-extension-points
- data-quality-checks
- variant-consolidation
layout: problem
---

## Description

Entity-attribute-value overuse occurs when data that has a known, stable structure is stored generically — one row per attribute, with the attribute name in one column and its value in another, usually as text. The pattern is introduced for a real reason: it allows new fields to be added without a schema change, which is attractive when schema changes are slow, when each customer needs different fields, or when the requirements are genuinely unknown. What it trades away is everything the database does for you. Types, constraints, foreign keys, defaults, and meaningful indexes all become impossible, because the database can no longer see what it is storing. That validation does not disappear; it moves into application code, where it is enforced inconsistently or not at all, and the data quietly accumulates values that the intended model would have rejected.

## Indicators ⟡

- A table with columns along the lines of `entity_id`, `attribute_name`, and `value`, where `value` is a text column holding numbers, dates, and flags
- Retrieving one business object requires a join or pivot across many rows, and the query is generated rather than written
- The set of valid attribute names exists only in application code, in a lookup table nobody maintains, or in nobody's head
- Reporting is done against a separate flattened copy of the data, because reporting against the live model is impractical
- Simple questions — how many customers have this field set, what values does it take — require a specialist to answer
- Type errors surface at read time, far from where the wrong value was written
- The same conceptual field appears under several attribute names that accumulated over the years

## Symptoms ▲

- [Slow Database Queries](slow-database-queries.md)
<br/>  Reconstructing one object requires many rows to be joined or pivoted, and filtering on an attribute cannot use an index the way a typed column would.
- [High Number of Database Queries](high-number-of-database-queries.md)
<br/>  Application code frequently fetches attributes individually rather than as one object, multiplying round trips for a single logical read.
- [N Plus One Query Problem](n-plus-one-query-problem.md)
<br/>  Loading a collection of entities and then their attributes per entity is the natural shape of code written against this model.
- [Difficult to Understand Code](difficult-to-understand-code.md)
<br/>  The domain model is invisible in the schema, so understanding what an entity actually consists of requires reading the code that assembles it.
- [Testing Complexity](testing-complexity.md)
<br/>  Test data must be constructed attribute by attribute, and the absence of constraints means invalid combinations are constructible and must be tested for.
- [Increased Bug Count](increased-bug-count.md)
<br/>  Defects that a typed schema would have made impossible — a date in a numeric field, a missing mandatory attribute — become ordinary runtime failures.
- [Imperative Data Fetching Logic](imperative-data-fetching-logic.md)
<br/>  Assembling objects from attribute rows pushes data access logic into procedural application code rather than into declarative queries.
- [Database Schema Design Problems](database-schema-design-problems.md)
<br/>  The generic model is itself the schema problem, and it prevents the normalization, typing, and relationships a designed schema would provide.

## Causes ▼

- [Excessive Customization](excessive-customization.md)
<br/>  When every customer needs different fields, a generic model looks like the only way to avoid a schema per installation.
- [Schema Evolution Paralysis](schema-evolution-paralysis.md)
<br/>  When changing the schema is slow, risky, or requires a release, developers route around it by storing new data generically.
- [Frequent Changes to Requirements](frequent-changes-to-requirements.md)
<br/>  A model expected to change constantly is built to absorb change, and the generic form is the most absorbent and least informative option available.
- [Fear of Breaking Changes](fear-of-breaking-changes.md)
<br/>  Adding a column feels risky in a system nobody fully understands, while adding an attribute row touches no existing structure and therefore feels safe.
- [Inadequate Requirements Gathering](inadequate-requirements-gathering.md)
<br/>  Where the domain was never modelled, a generic container defers the modelling indefinitely — and the deferral becomes permanent.
- [Database Schema Design Problems](database-schema-design-problems.md)
<br/>  Weak schema design practice makes the generic model attractive, because it removes the need to make design decisions at all.

## Detection Methods ○

- Look for tables whose column names are generic — attribute, key, name, property — paired with a value column of text type
- Count the distinct attribute names in use and compare against how many are set on more than a small percentage of entities; a short head and a long tail is the characteristic distribution
- Check whether any constraint, foreign key, or check exists on the value column; typically none does
- Measure how many rows the database reads to produce one business object, and compare against what a typed model would require
- Sample the value column and count entries that cannot be parsed as their intended type
- Search for the same concept stored under several attribute names, which indicates the vocabulary has never been governed
- Check whether reporting runs against this model or against a separate flattened copy — a copy is strong evidence the model is unqueryable

## Examples

An insurance policy system stored policy details in an attribute table because product managers needed to add fields without waiting for a release. After nine years the table held 640 million rows across roughly 1,100 distinct attribute names. Loading one policy required assembling 40 to 80 rows. Analysis of the attribute names found that 31 of them were set on more than 90 percent of policies — these were the actual policy structure, stored generically for no remaining reason — while 700 were set on fewer than a hundred policies each, and 140 had not been written since 2019. A question from the regulator about how many policies had a particular endorsement took four days to answer, because the value column contained the endorsement code sometimes as a code, sometimes as a description, and in one product line as a comma-separated list.

The absence of constraints had produced a quieter problem. Because the value column accepted anything, a defect in an import routine had written dates in two different formats for eleven months before anyone noticed, and the noticing happened when a renewal calculation produced a policy expiring in the year 20024. A typed date column would have rejected the write at the moment it occurred, next to the code that caused it. Instead the bad data was distributed across two years of records and the correction required reasoning about which format each value had been written in.
