---
title: Data Quality Checks
description: Ensuring data quality through validation, cleansing, and enrichment
category:
- Database
- Testing
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/data-quality-checks
problems:
- silent-data-corruption
- data-migration-integrity-issues
- data-migration-complexities
- inconsistent-behavior
- unpredictable-system-behavior
- unbounded-data-growth
layout: solution
---

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

## Examples

A legacy accounting system accumulated twenty years of transaction data with various quality issues: duplicate customer records, transactions with missing reference numbers, and amounts stored in inconsistent decimal formats. Before migrating to a new ERP system, the team implements a suite of data quality checks that scan the entire database and categorize issues by severity. They discover that 8% of customer records are duplicates and that thousands of transactions reference deleted accounts. The team builds cleansing scripts that merge duplicate customers (preserving transaction history) and reconcile orphaned transactions. Running these checks before migration prevents carrying years of data quality problems into the new system and avoids the costly task of fixing them after go-live.
