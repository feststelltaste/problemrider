---
title: Data Quality Checks
description: Ensuring data quality through validation, cleansing, and enrichment
category:
- Database
- Testing
problems:
- silent-data-corruption
- data-migration-integrity-issues
- data-migration-complexities
- inconsistent-behavior
- unpredictable-system-behavior
- unbounded-data-growth
- entity-attribute-value-overuse
- master-data-ownership-gaps
layout: solution
related_solutions:
- slug: data-integrity
  similarity: 0.7
- slug: continuous-data-verification
  similarity: 0.7
- slug: checksums
  similarity: 0.7
- slug: data-enrichment
  similarity: 0.7
- slug: plausibility-checks
  similarity: 0.7
- slug: code-quality-gates
  similarity: 0.7
---

## Description

Data quality checks are automated rules — mandatory fields, valid value ranges, referential integrity, format constraints, and business-specific validations — that are run periodically against a database, or applied at the point of entry, to detect and report data that violates the organization's expectations for correctness. The mechanism works at two points in the data lifecycle: checks applied at entry prevent new bad data from being created, while checks run periodically against existing data surface problems that already exist, categorized by severity so cleanup effort can be prioritized. In legacy systems this is essential precisely because entry-point validation was often weak or absent for years, allowing duplicate records, orphaned references, and inconsistent formats to accumulate silently until they surface as debugging mysteries or, worse, as corrupted reports that no one questioned until the numbers were visibly wrong. Data quality checks are especially critical before any migration, since a comprehensive quality assessment run against the legacy source lets the team quantify and address issues before they are carried into a new system, rather than discovering and fixing them after go-live when the cost of remediation is far higher. Because cleansing itself can be risky when the "correct" shape of the data is not well understood, quality checks are typically implemented as a detection and reporting mechanism first, with cleansing scripts applied in controlled, reviewed batches rather than as an automatic corrective action.

## How to Apply ◆

- Define data quality rules based on business requirements: mandatory fields, valid ranges, referential integrity, format constraints, and business logic validations.
- Implement automated data quality checks that run periodically against legacy databases to detect and report quality issues.
- Add validation at data entry points in the legacy system to prevent bad data from entering the system going forward.
- Create data cleansing scripts for known quality issues (duplicates, invalid formats, orphaned records) and run them in controlled batches.
- Monitor data quality metrics over time and set alerts when quality drops below acceptable thresholds.
- Run comprehensive data quality assessments before any data migration to identify and address issues proactively.

## Tradeoffs ⇄

**Benefits:**
- Prevents data quality issues from propagating through the system and causing downstream errors.
- Reduces the time spent debugging issues caused by bad data in legacy systems.
- Improves confidence in reports and analytics derived from legacy data.
- Identifies data quality problems before they cause issues during migration.

**Costs:**
- Implementing comprehensive data quality checks for a large legacy database requires significant effort.
- Data cleansing can be risky if business rules for "correct" data are not well understood.
- Automated checks add processing overhead and may impact database performance.
- False positives in quality checks can create alert fatigue.

## How It Could Be

A legacy accounting system accumulated twenty years of transaction data with various quality issues: duplicate customer records, transactions with missing reference numbers, and amounts stored in inconsistent decimal formats. Before migrating to a new ERP system, the team implements a suite of data quality checks that scan the entire database and categorize issues by severity. They discover that 8% of customer records are duplicates and that thousands of transactions reference deleted accounts. The team builds cleansing scripts that merge duplicate customers (preserving transaction history) and reconcile orphaned transactions. Running these checks before migration prevents carrying years of data quality problems into the new system and avoids the costly task of fixing them after go-live.
