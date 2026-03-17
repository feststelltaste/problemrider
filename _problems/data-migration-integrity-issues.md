---
title: Data Migration Integrity Issues
description: Data loses integrity, consistency, or meaning during migration from legacy
  to modern systems due to schema mismatches and format incompatibilities
category:
- Code
- Database
- Operations
related_problems:
- slug: cross-system-data-synchronization-problems
  similarity: 0.75
- slug: data-migration-complexities
  similarity: 0.7
- slug: database-schema-design-problems
  similarity: 0.6
- slug: schema-evolution-paralysis
  similarity: 0.6
- slug: legacy-configuration-management-chaos
  similarity: 0.55
- slug: legacy-business-logic-extraction-difficulty
  similarity: 0.55
layout: problem
---

## Description

Data migration integrity issues occur when transferring data from legacy systems to modern platforms results in data corruption, loss of relationships, semantic meaning changes, or consistency violations. These problems arise from fundamental differences between legacy and modern data models, encoding formats, constraint systems, and business rule implementations. Unlike simple data transfer challenges, these issues threaten the fundamental trustworthiness and usability of the migrated data in the new system.

## Indicators ⟡

- Legacy data models that don't map cleanly to modern database schemas or data structures
- Discovery of implicit business rules embedded in legacy data formats or constraints
- Character encoding inconsistencies between legacy and target systems
- Complex relationships in legacy data that have no equivalent in the target system design
- Data validation rules that differ significantly between source and target systems
- Legacy systems using proprietary data formats or custom serialization methods
- Missing or incomplete data dictionaries for legacy system fields and their meanings

## Symptoms ▲

- [Silent Data Corruption](silent-data-corruption.md)
<br/>  Data integrity issues during migration can go undetected initially, with corrupted data producing incorrect results without triggering errors.
- [Customer Dissatisfaction](customer-dissatisfaction.md)
<br/>  Users who encounter incorrect data, missing records, or corrupted information after migration become frustrated and lose trust in the system.
- [Increased Manual Work](increased-manual-work.md)
<br/>  Integrity issues require extensive manual data reconciliation and correction efforts after migration is completed.
- [System Outages](system-outages.md)
<br/>  Severe data integrity issues discovered after migration may force emergency halts for re-migration, causing unplanned downtime.
- [Increased Error Rates](increased-error-rates.md)
<br/>  Migrated data with integrity issues triggers validation failures and application errors in the new system.
## Causes ▼

- [Database Schema Design Problems](database-schema-design-problems.md)
<br/>  Fundamental schema differences between legacy and modern systems create mapping challenges that risk data integrity during migration.
- [Legacy Business Logic Extraction Difficulty](legacy-business-logic-extraction-difficulty.md)
<br/>  Business rules embedded in legacy data formats and constraints are difficult to identify and preserve during migration, leading to semantic data loss.
- [Data Migration Complexities](data-migration-complexities.md)
<br/>  The overall complexity of migration processes increases the likelihood of errors that compromise data integrity.
- [Poor Documentation](poor-documentation.md)
<br/>  Missing or incomplete documentation about legacy data fields, formats, and their meanings leads to incorrect mapping and transformation during migration.
- [Obsolete Technologies](obsolete-technologies.md)
<br/>  Legacy systems using proprietary data formats, outdated encodings like EBCDIC, or custom serialization create conversion challenges that risk data integrity.
## Detection Methods ○

- Implement comprehensive data validation and reconciliation testing before and after migration
- Perform statistical analysis comparing record counts, data distributions, and relationship integrity
- Use data profiling tools to identify inconsistencies between source and target data
- Conduct user acceptance testing with real business scenarios on migrated data
- Implement automated data quality checks to monitor ongoing data integrity
- Compare business report outputs between legacy and new systems for consistency
- Monitor application error logs for data-related validation failures after migration
- Conduct regular audits of critical business data for accuracy and completeness

## Examples

A financial institution migrates customer account data from a mainframe system to a modern database. The legacy system stored account balances as packed decimal fields with implicit currency information based on branch location, while customer names were stored in EBCDIC encoding with embedded formatting codes. During migration, decimal precision is lost due to floating-point conversion, causing penny discrepancies in thousands of accounts. Customer names become corrupted due to encoding issues, and the implicit currency logic is lost, causing international accounts to display incorrect balances. The migration appears successful with correct record counts, but the data integrity issues surface weeks later when customers report incorrect statements and regulatory reports fail audit requirements. The bank must halt operations to perform emergency data reconciliation and re-migration, costing millions in downtime and regulatory penalties.
