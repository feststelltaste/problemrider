---
title: Datensparsamkeit
description: Collecting and storing only necessary data
category:
- Database
- Security
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/datensparsamkeit
problems:
- unbounded-data-growth
- data-protection-risk
- regulatory-compliance-drift
- slow-database-queries
- high-database-resource-utilization
layout: solution
---

## How to Apply ◆

- Audit the legacy system's data collection points and identify data that is collected but never used or no longer needed.
- Define retention policies for each data category and implement automated archival or deletion of expired data.
- Remove unnecessary data fields from forms, APIs, and database tables, collecting only what serves a documented business purpose.
- Anonymize or pseudonymize personal data that must be retained for analytics but does not need to be identifiable.
- Review third-party integrations that inject data into the legacy system and eliminate unnecessary data flows.
- Document the business justification for each retained data field to support compliance audits.

## Tradeoffs ⇄

**Benefits:**
- Reduces data storage costs and improves database query performance by keeping tables smaller.
- Minimizes regulatory and compliance risk by limiting the amount of sensitive data held.
- Simplifies data migration when the legacy system is eventually replaced.
- Reduces the blast radius of data breaches by limiting what can be exposed.

**Costs:**
- Determining which data is "unnecessary" requires business knowledge that may be difficult to obtain for legacy systems.
- Deleting historical data may be irreversible and could remove information needed for future analysis.
- Retrofitting data minimization into a legacy system with decades of data accumulation is a substantial effort.
- Stakeholders may resist data deletion due to "we might need it someday" concerns.

## Examples

A legacy HR system has been collecting detailed personal data for over fifteen years, including fields like marital status, number of children, and emergency contacts of former employees who left a decade ago. A GDPR compliance audit reveals that the system retains far more personal data than legally justified. The team implements a data minimization strategy: they define retention periods for each data category, build automated jobs to anonymize records of former employees after the legally required retention period, and remove six data collection fields from the application that serve no current business purpose. The legacy database shrinks by 30%, query performance improves noticeably, and the organization's compliance posture strengthens significantly.
