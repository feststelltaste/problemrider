---
title: Master Data Stewardship
description: Give each shared reference object one accountable steward, a defined creation process, and a measured quality standard, so that data crossing departments has an owner.
category:
- Database
- Management
- Business
problems:
- master-data-ownership-gaps
- shadow-systems
- poor-interfaces-between-applications
- data-migration-complexities
- duplicated-effort
- lack-of-ownership-and-accountability
- custom-report-sprawl
- increased-manual-work
- inconsistent-execution
- data-migration-integrity-issues
- entity-attribute-value-overuse
- system-integration-blindness
layout: solution
---

## Description

Master data stewardship assigns one accountable person per shared reference object — customer, supplier, product, cost centre — who is responsible for the quality of that object as a whole, supported by a defined creation process and a measured standard. It addresses a structural gap rather than a technical one. Shared data is maintained by several departments, each of which cares about the fields it uses and none of which experiences the full cost of the object being wrong. Under that arrangement quality degrades reliably, and the degradation surfaces as symptoms in other places — failing interfaces, contradictory reports, duplicate payments — that get attributed to the systems reporting them rather than to the data. A steward is the mechanism by which an object that crosses organizational boundaries acquires a single point of responsibility.

## How to Apply ◆

> Data that crosses departments has no owner in most organizations, because ownership is assigned along the same functional lines the data crosses.

- **Name one steward per object**, a person rather than a committee or a department. The steward does not have to maintain the data; they have to be accountable for whether it is correct and be the addressee for questions about it.
- **Define the creation process**, including a mandatory search step before a new record may be created. Duplicates arise because creating is faster than searching, and the process is what changes that calculation.
- **Agree the quality standard per object**: which fields are mandatory in practice regardless of which department needs them, what the conventions are, and what constitutes a duplicate. Departments will disagree, and settling it is the steward's first job.
- **Measure quality continuously** — duplicate rate, completeness, convention violations — and send the report to the steward rather than to a distribution list. A measure with no addressee produces no action.
- **Separate creation authority from editing authority.** Restricting who may create records, while allowing broad editing of the fields a department owns, addresses duplication without making the data hard to maintain.
- **Work the existing backlog as a bounded piece of work** with an owner and an end, rather than as a permanent condition. Duplicate resolution is finite once new duplicate creation has been stopped.
- **Give downstream consumers a way to report problems** to the steward. Consuming systems detect quality issues first and usually have nowhere to send them, so they work around them silently instead.
- **Address the causes rather than the instances.** A recurring quality problem in one field usually indicates a process or a validation gap, and correcting records individually forever is the alternative to finding it.
- **Review the standard when the business changes.** New markets, new legal entities, and new product types alter what correct means, and a standard set once drifts out of usefulness.

## Tradeoffs ⇄

> Stewardship fixes the accountability gap that causes shared data to degrade, at the cost of a role somebody has to fill and process friction at the point of creation.

**Benefits:**

- Shared data acquires a single point of accountability, which is the structural condition its quality depends on.
- Duplicate creation falls sharply once searching is mandatory and creation authority is restricted, which is the largest single source of degradation.
- Downstream failures decline, since interfaces and reports depend on the quality of exactly this data.
- Migrations become substantially cheaper, because resolving accumulated duplicates and inconsistencies is usually the largest component of migration effort.
- Contradictory reporting is reduced, because the definitions the reports rely on are agreed and owned rather than assumed.

**Costs and Risks:**

- The steward role is real work that has to be resourced, and it is frequently added to someone's existing job where it is then not done.
- Restricting creation authority adds friction at the moment a user is trying to complete a task, which generates resistance and workarounds.
- Stewardship crosses departmental boundaries and therefore requires authority the steward may not have, which makes the role frustrating and hard to fill.
- Quality standards can become bureaucratic, demanding fields nobody uses because a committee thought they might matter.
- The existing backlog of bad data can be large enough to be demoralizing, and clearing it delivers no visible capability.

## How It Could Be

A manufacturer's supplier master contained roughly 6,800 probable duplicates in 41,000 records, created because purchasing and finance each added records to complete their own task and neither searched thoroughly. The intervention was organizational rather than technical: one named steward for supplier data, a creation process requiring a search whose result was recorded, creation authority restricted to a small group, and a monthly duplicate report sent to the steward personally. New duplicate creation fell by roughly ninety percent within two quarters. The existing backlog was then worked as a bounded project rather than as an ambient condition, taking about five months.

The downstream effect was larger than the data quality effect. Two consequences that had been attributed elsewhere resolved themselves: spend analysis had been understating concentration with individual suppliers, which had weakened a negotiating position the previous year, and a payment run had twice issued duplicate payments under two records for the same supplier. Neither had been recognized as a master data problem — the first had been treated as a reporting deficiency and the second as a payment process failure. The steward's most useful contribution in the first year was being an addressee: three departments that had been silently working around supplier data problems for years finally had somewhere to send them.
