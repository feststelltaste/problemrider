---
title: Production-Like Test Data
description: Build test data from anonymized production data so that tests encounter the messy, historical records that synthetic data never contains.
category:
- Testing
- Database
- Process
problems:
- inadequate-test-data-management
- testing-complexity
- inadequate-test-infrastructure
- testing-environment-fragility
- insufficient-testing
- flaky-tests
- high-defect-rate-in-production
- regression-bugs
- data-migration-complexities
- data-migration-integrity-issues
- poor-test-coverage
- increased-manual-testing-effort
- increased-bug-count
- schema-evolution-paralysis
- test-debt
- incorrect-index-type
layout: solution
---

## Description

Production-like test data means constructing test datasets from real production data, anonymized and reduced, rather than generating them synthetically. The reason is that synthetic data is generated from the schema and from the developer's model of the domain — which means it contains exactly the cases the developer already thought of, and none of the cases that make legacy systems fail. Real data in an old system is far stranger: records created before a column existed, encodings from a migration in 2009, customers with names the validation would now reject, and states that the current code believes are impossible. These are the inputs that break things in production and never appear in a test suite. Anonymized production data brings that distribution into testing, which is usually the fastest available improvement in the realism of a legacy test environment.

## How to Apply ◆

> The most valuable property of production data in a legacy system is its history: twenty years of records written by a dozen versions of the application, several of which had different ideas about what was valid.

- **Anonymize before the data leaves production**, in the extraction step, not after it lands somewhere else. Every architecture where raw production data reaches a lower environment "temporarily" eventually leaks, and this is both a legal and a reputational exposure.
- **Preserve the shape while replacing the values.** Referential integrity, cardinalities, distributions, and edge-case structure must survive anonymization, or the dataset loses exactly what made it valuable. Replacing every name with "Test User" destroys the encoding and length cases that were the point.
- Use **consistent pseudonymization** so that the same real value maps to the same replacement everywhere. Without it, joins break and multi-table scenarios become untestable.
- **Handle indirectly identifying data**, not just names and identifiers. A rare combination of postcode, birth date, and product can identify a person as effectively as a name, particularly in small populations, and naive anonymization routinely misses this.
- **Reduce the volume while keeping the variety.** A random one percent sample loses the rare cases that matter most. Sample by taking a slice of ordinary records plus a deliberate sweep of every distinct enumeration value, boundary date, and unusual state present in the full dataset.
- **Automate refresh on a schedule** so the test data tracks how production evolves. A dataset extracted once and used for three years slowly stops resembling the system, and the resemblance was the entire justification.
- **Involve the data protection function early** and document the anonymization approach. This is a legal question in most jurisdictions, and an approach that was never reviewed tends to be discovered during an audit rather than a design discussion.
- **Combine with synthetic generation** rather than replacing it. Anonymized data covers the historical and the strange; synthetic generation covers volume for load testing and new cases that production has not produced yet.
- **Treat the extraction pipeline as production code** — reviewed, tested, and version controlled. A defect in the anonymization is a data breach, not a test failure.

## Tradeoffs ⇄

> Real data finds the defects that synthetic data structurally cannot, at the cost of a genuine privacy risk and a pipeline that has to be built and maintained.

**Benefits:**

- Tests encounter the historical and malformed records that actually break legacy systems and that no developer would think to generate.
- Data migrations can be rehearsed against realistic input, which is where most migration failures originate — the records that violate assumptions nobody knew they were making.
- Query performance behaves realistically, since data distribution and volume drive execution plans in ways that uniform synthetic data does not reproduce.
- Defects found in production can be reproduced in a test environment, which is often impossible when the test data has no comparable case.
- Undocumented data states become visible, and each one found is a piece of the system's actual specification recovered.

**Costs and Risks:**

- Anonymization can fail. Incomplete anonymization in a lower environment is a data breach, and lower environments have weaker access controls precisely because they were assumed not to hold real data.
- Indirect identification is subtle and easy to get wrong, particularly for small populations or rare attribute combinations.
- The extraction and anonymization pipeline is real software that must be maintained as the schema evolves, and it breaks silently when a new column appears.
- Realistic volumes make test environments larger and slower, which pushes against the fast feedback that tests need.
- Some jurisdictions and sectors restrict this approach heavily regardless of anonymization quality, and the legal review can take longer than building the pipeline.

## How It Could Be

A team maintaining a pension administration system had a synthetic test dataset of 500 members, all with well-formed records, generated from the current schema. Production held 340,000 members with records going back to 1987. Their defect pattern was consistent: changes passed all tests and then failed in production on records that predated some schema change. They built an anonymization pipeline producing a 4,000-member extract that deliberately included every distinct combination of scheme type, status, and contribution history present in the full population. The first run of the existing test suite against the new dataset produced 31 failures, all genuine: unhandled null contribution periods, two date formats the code assumed had been migrated away, and a member category that had been closed to new entrants in 1998 and which three code paths did not handle. Production defects attributed to unexpected data fell by roughly seventy percent over the following two quarters.

The same dataset changed how the team approached a subsequent schema migration. Their previous migration had been tested against the synthetic data, run clean, and then failed in production on 1,200 records, requiring a rollback at four in the morning. Rehearsing the next migration against the anonymized extract found four failure classes before the change was even scheduled, including a set of records whose foreign key pointed at a table row that had been deleted years earlier — a state the schema forbade and that the data nevertheless contained. The migration ran in production without incident, and the orphaned-record problem was fixed as its own piece of work rather than as an emergency.
