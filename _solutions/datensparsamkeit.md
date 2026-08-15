---
title: Datensparsamkeit
description: Only collect and store personal data that is necessary for the purpose
category:
- Security
- Architecture
problems:
- data-protection-risk
- regulatory-compliance-drift
- unbounded-data-growth
- high-database-resource-utilization
- silent-data-corruption
- insufficient-audit-logging
- slow-database-queries
- inadequate-test-data-management
- retention-obligations-block-change
layout: solution
related_solutions:
- slug: data-flow-control
  similarity: 0.7
- slug: encryption
  similarity: 0.7
- slug: least-privilege
  similarity: 0.7
- slug: retention-and-disposal-policy
  similarity: 0.7
- slug: backup-and-recovery
  similarity: 0.65
- slug: privacy-by-design
  similarity: 0.65
---

## Description

Datensparsamkeit — data minimization — restricts what personal and sensitive data a system collects and how long it retains it to only what is strictly necessary for an active, specific business purpose, replacing the default legacy posture of collecting and keeping everything indefinitely just in case. Applying it means auditing every stored data element to establish whether the business purpose it once served is still valid, removing collection of fields at the source once that purpose no longer applies, and implementing automated retention and deletion or anonymization processes rather than relying on manual cleanup that tends to be deferred indefinitely. This is particularly consequential for legacy systems because they were typically built and extended long before data protection regulation made minimization a legal expectation, and as a result they tend to have quietly accumulated years of personal data for customers who are no longer active and fields that were once required but are no longer used by any current process. Reducing that footprint directly reduces the harm a future data breach or unauthorized access incident can cause — data that was never retained cannot be exfiltrated — and it simultaneously narrows the scope of compliance obligations and the security controls needed to satisfy them. The corresponding risk is that historical data, once deleted or anonymized, cannot be recovered if a legitimate future business need for it emerges, and that data dependencies across interconnected legacy systems are rarely documented well enough to guarantee that removing data in one system will not silently break something in another.

## How to Apply ◆

> Legacy systems tend to collect and retain all available data indefinitely, often storing sensitive information that is no longer needed for any business purpose. Datensparsamkeit (data minimization) reduces risk by limiting data collection and retention to what is strictly necessary.

- Audit all personal and sensitive data stored in the legacy system. For each data element, determine the specific business purpose it serves and whether that purpose is still valid. Data collected "just in case" or "because we always have" should be eliminated.
- Implement data retention policies with specific expiration periods for each data category. Personal data should be automatically deleted or anonymized when it is no longer needed for its stated purpose.
- Remove unnecessary data collection from input forms and APIs. Legacy systems often collect fields that were once required but are no longer used — removing these fields at the source prevents unnecessary data from entering the system.
- Anonymize or pseudonymize data used for analytics, testing, and development environments. Full production data with real personal information should never be used in non-production contexts.
- Implement automated data cleanup processes that purge expired data according to retention policies. Manual cleanup processes are unreliable and tend to be deferred indefinitely.
- Review third-party data sharing agreements to ensure that only necessary data is shared with external partners and that shared data is subject to the same minimization and retention standards.
- Document the legal basis and business justification for each category of personal data the system collects and retains. This documentation is required by GDPR and serves as the foundation for minimization decisions.

## Tradeoffs ⇄

> Data minimization reduces the risk and cost of data breaches, simplifies compliance, and reduces storage costs, but it requires careful analysis of data dependencies and may limit future analytics capabilities.

**Benefits:**

- Reduces the impact of data breaches by limiting the amount of sensitive data that can be exfiltrated — you cannot lose data you do not have.
- Simplifies compliance with data protection regulations (GDPR, CCPA) that mandate data minimization as a core principle.
- Reduces storage costs and database complexity by eliminating unnecessary data accumulation.
- Decreases the scope and cost of security controls by reducing the volume of data that requires protection.

**Costs and Risks:**

- Historical data that is deleted cannot be recovered if a future business need is identified, requiring careful analysis before purging.
- Data dependencies across interconnected legacy systems may not be fully documented, and deleting data in one system can break functionality in another.
- Implementing data retention policies in legacy databases with no temporal metadata requires adding expiration tracking infrastructure.
- Stakeholders may resist data minimization due to concerns about losing future analytics or reporting capabilities.

## How It Could Be

> The following scenarios illustrate how data minimization reduces risk in legacy systems.

A legacy customer relationship management system has accumulated 15 years of customer data, including home addresses, phone numbers, dates of birth, and purchase histories for customers who have not made a purchase in over 10 years. A GDPR subject access request reveals that the system stores personal data for 3.2 million customers, only 400,000 of whom are active. The team implements a data retention policy: inactive customer records are anonymized after 3 years (purchase history retained without personal identifiers for business analytics, personal data deleted). Active customer records are reviewed annually to remove fields no longer needed for current business processes. The personal data footprint shrinks by 75%, and when a subsequent security incident exposes database records, the impact assessment is dramatically smaller because the exposed records contain anonymized data for the majority of entries.

A legacy healthcare appointment system collects and permanently stores patient insurance policy numbers, Social Security numbers, and full medical histories — even though the appointment system only needs name, date of birth, and insurance verification status to function. The team works with the clinical and billing teams to identify the minimum data set needed for appointment scheduling and removes all unnecessary fields from the system. Insurance policy numbers are verified against the insurance provider's API at booking time but are not stored. Social Security numbers are removed entirely, as they are already maintained in the separate clinical records system. The simplified data model reduces the system's PCI and HIPAA compliance scope and eliminates an entire category of data breach risk.
