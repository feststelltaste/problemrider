---
title: Master Data Ownership Gaps
description: Core reference data is shared across modules and departments with no owner, so its quality degrades and nobody is responsible for correcting it.
category:
- Database
- Management
- Business
related_problems:
solutions:
- master-data-stewardship
- clear-ownership-model
- data-quality-checks
- data-strategy
- canonical-data-model
- data-modeling
- consistent-terminology
- ubiquitous-language
- continuous-data-verification
- plausibility-checks
- data-deduplication
layout: problem
---

## Description

Master data ownership gaps occur when the reference data on which many processes depend — customers, suppliers, products, cost centres, organizational units — is created and edited by several departments with no one accountable for its overall quality. Packaged systems make this likely by design, because their modules share master data and each module's users maintain the fields they care about. The result is a shared resource with distributed maintenance and no steward: duplicate entries created because searching was harder than adding, fields left blank by whoever did not need them, and inconsistent conventions that each department considers correct. Because no single department experiences the full cost, the degradation is visible only in downstream symptoms — failing interfaces, contradictory reports, manual reconciliation — that are attributed to other causes.

## Indicators ⟡

- The same customer, supplier, or product exists several times under slightly different entries
- Departments maintain their own lists alongside the system because the system's version cannot be trusted
- Data quality problems are raised repeatedly, addressed as individual corrections, and recur
- No one can say who is permitted to create a master record, or who approves it
- Naming and coding conventions differ by department and each considers its own to be the standard
- Interfaces to other systems fail on records that are valid in the source and unusable downstream

## Symptoms ▲

- [Shadow Systems](shadow-systems.md)
<br/>  Departments maintain private lists because they cannot rely on the shared data, and those lists become load-bearing without anyone deciding so.
- [Increased Manual Work](increased-manual-work.md)
<br/>  Reconciliation, correction, and duplicate merging become permanent recurring tasks in several departments simultaneously.
- [Poor Interfaces Between Applications](poor-interfaces-between-applications.md)
<br/>  Downstream systems receive records that are internally valid but violate assumptions nobody documented, and integrations fail intermittently.
- [Data Migration Complexities](data-migration-complexities.md)
<br/>  Any migration must first resolve the accumulated duplicates and inconsistencies, which is frequently the largest part of the effort.
- [Duplicated Effort](duplicated-effort.md)
<br/>  Several departments independently maintain, correct, and reconcile overlapping views of the same entities.
- [Custom Report Sprawl](custom-report-sprawl.md)
<br/>  Contradictory outputs proliferate because each department reports on its own interpretation of the shared data.

## Causes ▼

- [Lack of Ownership and Accountability](lack-of-ownership-and-accountability.md)
<br/>  Shared data crosses departmental boundaries, and organizations rarely assign ownership to anything that does.
- [Organizational Structure Mismatch](organizational-structure-mismatch.md)
<br/>  Departments are organized around functions while the data is organized around entities, so no department's remit covers a record's whole life.
- [Inadequate Requirements Gathering](inadequate-requirements-gathering.md)
<br/>  Master data governance is rarely stated as a requirement during implementation, so no process for it is ever designed.
- [Excessive Customization](excessive-customization.md)
<br/>  Department-specific fields and validations accumulate on shared records, and each set is meaningful to one department and noise to the others.
- [Short-Term Focus](short-term-focus.md)
<br/>  Creating a duplicate record takes a minute and resolves today's task; searching properly or fixing the underlying entry does not.

## Detection Methods ○

- Measure duplicate rates in the main master data objects using fuzzy matching on names, identifiers, and addresses
- Count records with incomplete mandatory-in-practice fields, and check whether completeness varies by creating department
- Ask who owns customer master data and observe how long the answer takes and how many names it contains
- Track how much effort goes into data correction and reconciliation across departments, which is usually unmeasured and substantial
- Review interface failure logs for rejections caused by source data rather than by technical faults
- Look for departmental spreadsheets that duplicate master data, which indicate where trust has already been lost

## Examples

A manufacturer's supplier master contained 41,000 records, of which analysis using fuzzy matching on name, tax identifier, and bank details identified approximately 6,800 as probable duplicates. Purchasing created records to complete an order, finance created them to process an invoice, and neither searched thoroughly because searching was slower than creating. The consequences appeared elsewhere: spend analysis understated concentration with individual suppliers, which had caused a negotiation to be entered from a weaker position than the facts supported, and a payment run had twice sent duplicate payments to the same supplier under two records.

The ownership question turned out to be the whole problem. Asked who owned supplier master data, the organization produced three answers — purchasing, finance, and the enterprise system team — each of which regarded its role as maintaining a portion. No one was responsible for whether the object as a whole was correct. The intervention was not technical: one named steward per master data object, a defined creation process with a mandatory search step, and a monthly duplicate report sent to the steward rather than to a distribution list. Duplicate creation fell by roughly ninety percent within two quarters, and the backlog of existing duplicates became a bounded piece of work with an owner rather than a permanent condition.
