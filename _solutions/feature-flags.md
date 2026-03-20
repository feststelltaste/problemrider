---
title: Feature Toggles
description: Enable or disable functions through configuration switches
category:
- Operations
- Architecture
- Process
quality_tactics_url: https://qualitytactics.de/en/maintainability/feature-toggles/
problems:
- fear-of-breaking-changes
- deployment-risk
- large-risky-releases
- release-instability
- frequent-hotfixes-and-rollbacks
- missing-rollback-strategy
- release-anxiety
- long-lived-feature-branches
- merge-conflicts
- fear-of-change
layout: solution
---

## How to Apply ◆

> In legacy modernization, feature toggles decouple the act of deploying new code from the act of exposing it to users, making it possible to ship changes to a fragile production environment in small, reversible steps.

- Use release toggles to ship new implementations of legacy behavior to production in an inactive state; the old code path remains live until the team has verified the new one under real conditions.
- Wrap each replaced legacy component behind an ops toggle that can instantly revert traffic to the old code without a redeployment — this is the safety net that allows teams to move faster in systems where rollbacks are traditionally slow and risky.
- Introduce toggles at the seam between old and new implementations rather than scattering conditionals through business logic; use the strategy pattern so the toggle selects which implementation to inject at startup.
- Apply percentage rollouts to gradually expose a new implementation to increasing fractions of users, validating behavior on live data before full cut-over — particularly important when the legacy system has undocumented edge cases that only appear at production scale.
- Coordinate toggle state with the legacy system's release windows; in organizations where the legacy system deploys quarterly, toggles allow new service code to be deployed continuously while the business-visible feature release is controlled separately.
- Establish a maximum lifespan for every release toggle at creation time and treat expired toggles as bugs; in legacy contexts, toggles left in place become permanent dead code that looks exactly like the technical debt the team was trying to escape.
- Log which toggle state was active during every request so that discrepancies between old and new behavior can be traced during incident investigation.
- Before removing a toggle, run both code paths in shadow mode — execute the new path but compare its output to the old path's result without using it — to verify parity before the final cut-over.

## Tradeoffs ⇄

> Feature toggles give legacy modernization teams a powerful mechanism for incremental, reversible change, but they must be managed as carefully as the code they protect or they become a source of the same complexity they were intended to reduce.

**Benefits:**

- Eliminates the need for risky big-bang cut-overs by allowing new implementations to coexist with legacy code in production and be activated incrementally.
- Enables instant, no-redeployment rollback of a specific new feature without reverting unrelated bug fixes or other improvements that were shipped in the same deployment.
- Allows continuous deployment of code to a legacy production environment even when business approval for a release is days or weeks away.
- Reduces the blast radius of a defect in new code by limiting exposure to a controlled percentage of traffic or a specific user group before full rollout.
- Gives operations teams a runtime control surface for disabling resource-intensive new features during peak load without waiting for a code change.

**Costs and Risks:**

- Each active toggle adds a code path that must be tested in both states; legacy systems with already poor test coverage can quickly accumulate combinations that nobody has verified.
- Toggles left in place after their purpose has ended are a form of technical debt that is particularly insidious in legacy codebases — they look like intentional configuration but hide dead code.
- In organizations with manual deployment processes and long release cycles, toggle state management can become a coordination problem as multiple teams independently control overlapping toggles.
- The business and operations staff who need to manage toggle state during incidents often lack tooling for it in legacy environments, leading to toggle changes being routed through developers as ad-hoc requests.
- Nesting toggles — where one toggled code path contains another conditional on a second toggle — creates interaction complexity that is nearly impossible to reason about, a risk that grows in systems where multiple modernization streams are in flight simultaneously.

## How It Could Be

> The following scenarios illustrate how feature toggles have enabled safer, more controlled change in legacy system modernization.

A public sector agency was replacing a twenty-year-old benefits calculation engine with a re-implemented version based on current legislation. Because the old engine had never been formally tested, there was no way to verify equivalence before going live. The team wrapped the new engine behind a release toggle and ran both engines in parallel during a shadow period: every calculation request was processed by both, and the results were compared and logged. Discrepancies were investigated and the new engine was corrected. Only after six weeks of shadow-mode agreement on all case types did the team flip the toggle to route live traffic to the new engine. The old engine remained available behind the toggle as a fallback for three months.

A logistics company was migrating its pricing service from a monolithic Delphi application to a modern REST API. Pricing logic was complex, varied by customer contract, and had accumulated fifteen years of special cases. The team deployed the new pricing API alongside the old system and used a permission toggle to route specific customer accounts to the new service. Starting with low-volume, internally managed accounts, they expanded the enabled set progressively over twelve weeks. Several edge cases surfaced for specific contract types during the rollout; because each affected customer was already identified by the toggle's targeting rules, the team could disable the new path for just those accounts while fixing the issue, without affecting the rest of the rolled-out population.

A European bank was replacing its interest rate calculation module, which ran nightly as part of a batch process. The batch environment had no automated deployment pipeline — releases went through a four-week change approval process. The team introduced a simple database-backed toggle that the calculation module read at startup. New calculation code was deployed during a standard maintenance window. The toggle remained set to the old code path until the business had signed off on the parallel run results, at which point the operations team updated a single row in the configuration table and the next batch run used the new implementation. No additional deployment was needed, and the approval process had been satisfied weeks earlier when the code was deployed in the inactive state.
