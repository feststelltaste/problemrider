---
title: Runbooks
description: Providing detailed instructions for processing tasks and incidents
category:
- Operations
- Communication
quality_tactics_url: https://qualitytactics.de/en/reliability/runbooks
problems:
- slow-incident-resolution
- knowledge-silos
- knowledge-dependency
- implicit-knowledge
- poor-documentation
- constant-firefighting
- difficult-developer-onboarding
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Document step-by-step procedures for all known legacy system failure modes and common operational tasks
- Include diagnostic steps, resolution procedures, escalation paths, and rollback instructions
- Store runbooks in a searchable, version-controlled system accessible to all on-call engineers
- Write runbooks at a level that enables someone unfamiliar with the legacy system to follow them
- Update runbooks after every incident where the existing documentation was insufficient
- Include links to monitoring dashboards, log queries, and configuration locations relevant to each procedure
- Review and test runbooks periodically to ensure they remain accurate as the system evolves
- Assign ownership for each runbook to ensure accountability for keeping them current

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces mean time to resolution by providing immediate guidance during incidents
- Captures institutional knowledge about legacy systems that might otherwise exist only in people's heads
- Enables less experienced team members to handle incidents confidently
- Reduces the dependency on specific individuals for legacy system operational knowledge

**Costs and Risks:**
- Runbooks require ongoing maintenance effort to stay current
- Overly rigid runbooks can discourage critical thinking during novel incidents
- Stale runbooks can provide incorrect guidance that worsens incidents
- Writing comprehensive runbooks takes significant initial effort

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A financial company's legacy settlement system had operational procedures known only to two senior engineers who had maintained it for over a decade. When one engineer left and the other was on vacation during a critical incident, the on-call team spent four hours diagnosing an issue that normally took 15 minutes to resolve. The team subsequently invested two weeks in creating runbooks for the 20 most common incident types, including database connection reset procedures, batch job restart sequences, and data reconciliation steps. The next similar incident was resolved in 12 minutes by a junior engineer following the runbook.
