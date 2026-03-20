---
title: Requirements Traceability Matrix
description: Maintaining explicit bidirectional mappings from requirements through design, code, and tests
category:
- Requirements
- Process
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/requirements-traceability-matrix
problems:
- requirements-ambiguity
- inadequate-requirements-gathering
- misaligned-deliverables
- insufficient-testing
- poor-test-coverage
- regulatory-compliance-drift
- legacy-system-documentation-archaeology
- feature-gaps
layout: solution
---

## How to Apply ◆

> In legacy systems, a requirements traceability matrix helps teams understand which parts of the codebase implement which business requirements — knowledge that is often completely lost over years of undocumented changes.

- Start by inventorying the known business requirements the legacy system fulfills, drawing from any available documentation, user interviews, and analysis of the existing codebase.
- Create a matrix that maps each requirement to the code modules, database objects, and tests that implement or verify it, even if the mapping is initially incomplete.
- Use the matrix to identify untested requirements — these are high-risk areas where changes could break critical functionality without any automated detection.
- When planning modernization work, use the matrix to determine the full impact of replacing or modifying a specific business capability.
- Update the matrix as part of every change to the system, making traceability maintenance a standard practice rather than a one-time documentation exercise.
- Use the matrix during compliance audits to demonstrate that regulatory requirements are implemented and verified, which is especially important in regulated industries modernizing legacy systems.

## Tradeoffs ⇄

> A traceability matrix provides invaluable visibility into legacy systems but requires sustained effort to create and maintain.

**Benefits:**

- Makes the relationship between business requirements and implementation explicit, reducing the risk of accidentally removing or breaking critical functionality during modernization.
- Enables impact analysis for proposed changes by showing exactly which requirements, code, and tests are affected.
- Supports compliance and audit requirements by providing documented evidence that regulatory requirements are implemented and tested.
- Helps identify orphaned code — implementation that no longer maps to any active requirement and can potentially be removed.

**Costs and Risks:**

- Building the initial matrix for a legacy system with poor documentation is a significant effort that may require weeks of reverse engineering.
- If the matrix is not maintained as the system evolves, it becomes misleading — worse than having no matrix at all.
- Overly detailed matrices create maintenance overhead that teams may abandon under delivery pressure.
- The matrix is only as good as the team's understanding of the legacy system's requirements, which may itself be incomplete or incorrect.

## How It Could Be

> The following scenario illustrates how a traceability matrix supports legacy modernization in a regulated environment.

A pharmaceutical company was modernizing its laboratory information management system (LIMS) that had been in use for 18 years. Regulatory requirements mandated that every calculation in the system be traceable to a validated requirement and covered by a documented test. The team built a traceability matrix by reverse-engineering the legacy codebase, mapping 340 regulatory requirements to specific code modules and existing test cases. The matrix revealed that 45 requirements had no corresponding tests and 23 had tests that no longer passed. This analysis drove the test remediation plan and provided regulators with confidence that the modernization would maintain compliance. During the migration, the matrix served as a checklist — each requirement was individually verified in the new system before the corresponding legacy module was decommissioned.
