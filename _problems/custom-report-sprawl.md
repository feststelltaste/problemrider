---
title: Custom Report Sprawl
description: Hundreds of bespoke reports, forms, and extracts accumulate with no record
  of who uses them, so none can be changed or removed with confidence.
category:
- Business
- Database
- Process
related_problems:
- slug: excessive-customization
  similarity: 0.7
- slug: low-code-customization-sprawl
  similarity: 0.65
- slug: customization-outside-version-control
  similarity: 0.65
- slug: entity-attribute-value-overuse
  similarity: 0.6
- slug: authorization-role-explosion
  similarity: 0.6
- slug: reimplemented-standard-functionality
  similarity: 0.6
solutions:
- feature-usage-measurement
- variant-consolidation
- strategic-code-deletion
- attribute-usage-analysis
- consistent-terminology
- ubiquitous-language
- clear-ownership-model
- materialized-views
- data-strategy
- customization-cost-attribution
- master-data-stewardship
layout: problem
---

## Description

Custom report sprawl occurs when the bespoke outputs of a commercially purchased software system — reports, forms, extracts, dashboards, printed documents — accumulate over years without any of them ever being retired. Each was requested by someone, built quickly, and never revisited. Because outputs are cheap to add and invisible when unused, the inventory grows monotonically until it contains several hundred items of which a small fraction are actually consulted. The cost is not the storage but the coupling: every one of these outputs reads the data model directly, so a schema change, an upgrade, or a data migration must consider all of them. They also produce inconsistent answers, because the same business figure has been calculated slightly differently in eleven places by eleven people over a decade.

## Indicators ⟡

- The report inventory has hundreds of entries and nobody can name the ones that matter
- Two reports of the same figure disagree, and which is correct is a matter of opinion
- A schema change is estimated by first establishing which reports would break, and that takes days
- Reports carry the name of the person who requested them, sometimes long departed
- Users export report output into spreadsheets and do further work there, indicating the report does not answer their question
- No report has ever been decommissioned, and there is no process by which one would be

## Symptoms ▲

- [Duplicated Effort](duplicated-effort.md)
<br/>  The same figure is computed repeatedly in different outputs, and each computation is maintained separately.
- [Shadow Systems](shadow-systems.md)
<br/>  Where reports do not answer the actual question, users build spreadsheets alongside them, which then become load-bearing and invisible.
- [Schema Evolution Paralysis](schema-evolution-paralysis.md)
<br/>  Every data model change must account for an unknown number of outputs reading the affected structures directly.
- [High Maintenance Costs](high-maintenance-costs.md)
<br/>  The inventory must be carried through every upgrade and migration, and its size is unrelated to the value it delivers.
- [Testing Complexity](testing-complexity.md)
<br/>  Regression testing an upgrade means verifying outputs whose correct results nobody can independently state.
- [Invisible Nature of Technical Debt](invisible-nature-of-technical-debt.md)
<br/>  Reports are not thought of as code, so the inventory does not appear in any technical assessment of the system.
- [Increased Manual Work](increased-manual-work.md)
<br/>  Reconciling contradictory outputs becomes a recurring manual task in the business functions that consume them.

## Causes ▼

- [Excessive Customization](excessive-customization.md)
<br/>  Reports are the cheapest form of customization to request and the least likely to be refused, so they accumulate fastest.
- [Inadequate Requirements Gathering](inadequate-requirements-gathering.md)
<br/>  A request for a specific report is fulfilled as stated rather than investigated, so a report that nearly answers the question is built alongside one that nearly does.
- [Entity-Attribute-Value Overuse](entity-attribute-value-overuse.md)
<br/>  Where the data model is not directly queryable, every question requires a purpose-built extract rather than an ad hoc query.
- [Lack of Ownership and Accountability](lack-of-ownership-and-accountability.md)
<br/>  No one owns the output inventory, so nothing prompts a review and nothing is ever removed.
- [Short-Term Focus](short-term-focus.md)
<br/>  Building the requested report is quick; establishing whether an existing one already serves the need is not, so it is skipped.

## Detection Methods ○

- Count the custom outputs and, if the platform records execution, count how many have been run in the past year
- Identify outputs that have never been executed, or were last executed years ago
- Look for several outputs computing the same business figure and compare their definitions
- Check whether any output has an owner, a stated purpose, or a documented definition of the figures it produces
- Measure how much of the last upgrade's regression effort went to verifying outputs
- Ask a business function which outputs they rely on and compare their list against the inventory

## Examples

An organization running a document and records platform had 780 custom outputs accumulated over twelve years. Execution logging, enabled for a quarter to answer the question, showed that 61 accounted for over 95 percent of all runs, that 430 had not been executed at all in three months, and that 190 had not been executed in the two years for which logs were retained. The team had estimated a planned data model change at four months, almost entirely because the impact on outputs was unknown. With the usage data, the change was scoped against the 61 that mattered and completed in five weeks. The 190 dormant outputs were decommissioned after a notice period, during which two were claimed — both annual, both legitimate, and both now recorded as such.

The inconsistency problem was harder and more revealing. Four outputs reported monthly processed volume, and they disagreed by up to eleven percent. Investigation found that each had been built for a different department at a different time, and each used a defensible but different definition of what counted as processed and when. None was wrong. The organization had been holding meetings for years in which departments presented incompatible numbers, and the cause had always been assumed to be a data quality problem rather than four undocumented definitions of a term nobody had ever agreed.
