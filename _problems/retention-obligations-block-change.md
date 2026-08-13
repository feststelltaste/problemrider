---
title: Retention Obligations Block Change
description: Legal retention duties attach to data whose format and system nobody can change, so the obligation freezes the system that holds it.
category:
- Database
- Operations
- Security
related_problems:
solutions:
- retention-and-disposal-policy
- data-archiving
- audit-trail-management
- system-decommissioning
- datensparsamkeit
- risk-quantification
- application-portfolio-inventory
- parallel-run
- checksums
- clear-ownership-model
layout: problem
---

## Description

Retention obligations block change when data an organization is legally required to keep — for years or decades — is held in a system that cannot be modified, migrated, or retired without putting that obligation at risk. The requirement is usually not merely that the data continue to exist, but that it remain retrievable, readable, complete, and demonstrably unaltered. That combination makes migration far harder than moving records: the organization must be able to show that what it produces years later is what was originally recorded. Faced with that burden and an unclear legal boundary, the safe answer is always to change nothing, and the system holding the data becomes frozen. Because nobody has established what must actually be kept and for how long, the freeze extends to everything rather than to the subset genuinely covered.

## Indicators ⟡

- The retention period for the data is stated in years, and nobody can produce the source obligation
- The system cannot be decommissioned because of retained data, and no plan exists for the data itself
- Old instances are kept running solely so that records remain retrievable
- Nobody can say what proportion of the retained data is genuinely subject to an obligation
- Deletion has never been performed, and no process exists by which it would be
- Legal and technical staff have never jointly examined what the obligation actually requires

## Symptoms ▲

- [Obsolete Technologies](obsolete-technologies.md)
<br/>  Systems are kept alive past their supported life purely as data custodians, along with the runtimes and hardware they need.
- [High Maintenance Costs](high-maintenance-costs.md)
<br/>  Licences, infrastructure, patching, and monitoring continue for systems that serve no operational purpose.
- [Modernization Strategy Paralysis](modernization-strategy-paralysis.md)
<br/>  Every option for the system founders on the retained data, and because the obligation is unexamined, no option can be evaluated properly.
- [Data Migration Complexities](data-migration-complexities.md)
<br/>  Migrating retained records requires demonstrating that meaning and integrity are preserved, which is a far stronger requirement than moving them.
- [Regulatory Compliance Drift](regulatory-compliance-drift.md)
<br/>  Retention practice was set once and never revisited, so it no longer matches obligations that have changed since.
- [Legacy Skill Shortage](legacy-skill-shortage.md)
<br/>  Systems kept alive for retention require skills nobody is acquiring, and the pool shrinks throughout the retention period.
- [Vendor Dependency Entrapment](vendor-dependency-entrapment.md)
<br/>  A frozen system cannot be replaced or renegotiated, which removes every commercial option with its supplier.

## Causes ▼

- [Regulatory Compliance Drift](regulatory-compliance-drift.md)
<br/>  Obligations were never mapped to specific data, so the whole system is treated as covered rather than the portion that is.
- [Lack of Ownership and Accountability](lack-of-ownership-and-accountability.md)
<br/>  Retention sits between legal, operations, and the business, and none of them owns establishing what must actually be kept.
- [Fear of Breaking Changes](fear-of-breaking-changes.md)
<br/>  The consequence of getting retention wrong is legal rather than technical, so the risk-averse answer is to freeze everything.
- [Poor Documentation](poor-documentation.md)
<br/>  The meaning of retained records depends on structures and codes that were never documented, so nobody can assert that a migrated copy means the same thing.
- [Entity-Attribute-Value Overuse](entity-attribute-value-overuse.md)
<br/>  Where the stored form is untyped and self-describing only by convention, demonstrating that a migration preserved meaning becomes very difficult.

## Detection Methods ○

- Ask for the specific legal source of each retention period in use, and how many can be produced
- Measure what share of retained data is within an obligation period and what share is past it
- Establish whether any deletion has ever occurred and what process would be used
- Count systems running solely to retain data, and total their annual cost
- Check whether legal and technical staff have jointly assessed what retrievable, readable, and unaltered require in practice
- Test retrieval of a record from the oldest retained period and record how long it takes and what it requires

## Examples

An insurer kept three superseded policy administration systems running solely because policy documents had to remain retrievable for periods extending to thirty years after a contract ended. The combined annual cost of licences, infrastructure, and the specialist contractor retained to keep one of them running was substantial and had been renewed for nine years without examination. A joint review by legal and technology established that the obligation attached to the policy document and a defined set of transaction records, not to the operational system, and that an archive preserving those artifacts with an integrity guarantee would satisfy it. Two of the three systems were decommissioned within a year.

The same review found the opposite problem alongside it. Roughly 40 percent of the retained data was past every applicable period and should have been deleted years earlier — which was not merely a storage question, since retaining personal data beyond its lawful period is itself a breach in the applicable jurisdiction. The organization had assumed for a decade that retention was a matter of keeping things, and had never considered that the obligation had an upper bound as well as a lower one. Nobody had been responsible for asking, because retention had been treated as a legal topic by technology and as a technical topic by legal.
