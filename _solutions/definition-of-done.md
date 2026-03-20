---
title: Definition of Done
description: Define clear criteria for the completion of functionality
category:
- Process
- Testing
- Code
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/definition-of-done/
problems:
- poor-test-coverage
- insufficient-testing
- high-bug-introduction-rate
- quality-degradation
- inconsistent-quality
- high-defect-rate-in-production
- quality-compromises
- quality-blind-spots
- partial-bug-fixes
- lower-code-quality
- reduced-feature-quality
- inadequate-error-handling
- poor-documentation
layout: solution
---

## How to Apply ◆

> In legacy contexts where "done" has historically meant "the developer says it works," a formal Definition of Done is often the first mechanism that prevents partially-finished changes from silently eroding an already fragile system.

- Start with a minimal DoD that the team can realistically meet on every work item, even under legacy pressure: code reviewed, all existing tests still passing, change deployed to a test environment. Add criteria as the team's infrastructure matures.
- Explicitly include a "no new debt introduced without a corresponding backlog entry" criterion — this makes the accumulation of workarounds visible rather than letting them slip in under time pressure.
- Add a regression check criterion requiring that the change has been verified against the specific legacy behaviors most likely to be disturbed — integration points with external systems, batch jobs, and scheduled processes are common blind spots.
- Require that any change touching an undocumented area includes at least a brief inline comment or decision record explaining what the code does and why — this creates incremental documentation without requiring a separate documentation sprint.
- Include a rollback verification criterion for database-touching changes: the migration must be tested in reverse before the story is considered done, given that legacy systems rarely have automated rollback coverage.
- Distinguish clearly between the DoD (the universal quality bar applied to every work item) and story-specific acceptance criteria, which define what each particular feature must do — legacy teams often conflate these and end up with neither properly enforced.
- Review and extend the DoD at retrospectives whenever a production incident can be traced to something that was "done" but missing a quality check — each incident is evidence of a gap in the DoD.
- Make the DoD visible in the team's daily workflow (sprint board, wiki, team channel) rather than a document written once and forgotten, since legacy teams under constant firefighting pressure will not look for it unless it is unavoidable.

## Tradeoffs ⇄

> A Definition of Done imposes short-term friction in exchange for the long-term benefit of preventing quality from degrading incrementally with each release.

**Benefits:**

- Prevents the "hardening sprint" pattern common in legacy projects, where months of deferred quality work pile up just before a release and cause delays or quality compromises.
- Creates a shared quality language across developers, testers, and operations staff who may otherwise have fundamentally different implicit definitions of "done," a gap especially wide in long-running legacy teams.
- Makes undone work visible: if the team consistently cannot meet the DoD within a sprint, it is a signal that the scope is too large or the quality infrastructure is too weak, both of which need addressing explicitly.
- Forces incremental improvement of the legacy system's quality infrastructure — test environments, deployment pipelines, documentation practices — because the DoD creates demand for them.
- Supports gradual increases in the quality bar: a team that consistently meets a simple DoD can raise it, creating a ratchet mechanism for continuous improvement rather than periodic cleanup campaigns.

**Costs and Risks:**

- In teams accustomed to shipping under pressure with minimal process, introducing a DoD is often perceived as bureaucracy, particularly when the criteria cannot be met within the existing sprint capacity — this creates pressure to waive the standard, which is worse than having none.
- Legacy systems often lack the test infrastructure, deployment automation, and documentation tooling needed to satisfy a meaningful DoD; the DoD may expose gaps in the engineering platform that require separate investment.
- An overly ambitious DoD in a legacy context can stall progress if teams feel that almost nothing can be "done" given the state of the surrounding system — the DoD must be calibrated to what is achievable now, not what would be ideal.
- Without management support for the time required to meet DoD criteria, teams experience the DoD as an additional unfunded mandate on top of an already demanding delivery schedule.

## Examples

> The following scenarios illustrate how teams introduced a Definition of Done into legacy environments and what impact it had.

A manufacturing company's ERP customization team had spent years adding features to a SAP system with no consistent quality gate. Developers considered a change done when the business user confirmed it worked in their manual test. After two costly rollbacks caused by changes that had not been tested against the nightly batch jobs, the team introduced a DoD that included a mandatory batch job simulation step in the staging environment before any change could close. This single criterion eliminated the rollback pattern within one quarter, because it forced the team to build the test infrastructure they had been deferring for years.

A healthcare provider's patient records system had accumulated hundreds of undocumented fields added over fifteen years of ad-hoc requests. When a new team took over modernization, they introduced a DoD criterion requiring that any change to the data model include a brief description of the field's purpose in a shared data dictionary. The team did not attempt to document all existing fields at once; instead, the DoD ensured that every field touched during normal development was documented as a side effect of the work already being done. Within two years, the most frequently accessed parts of the data model were fully documented without a dedicated documentation project.

A retail bank running a loan origination system had a persistent problem with features that passed QA but broke in production because of differences in configuration between environments. The team introduced a DoD criterion requiring that all new configuration parameters be added to an environment configuration checklist and verified in the production-equivalent staging environment before a story was closed. This criterion forced the creation of a configuration management process that had never existed, eliminating an entire class of production incidents that had previously required emergency patches.
