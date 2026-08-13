---
title: Role Model Rationalization
description: Rebuild an exploded permission model from evidence of what people actually
  do, and establish a process by which access is removed as well as granted.
category:
- Security
- Operations
- Process
problems:
- authorization-role-explosion
- authorization-flaws
- regulatory-compliance-drift
- increased-manual-work
- lack-of-ownership-and-accountability
- invisible-nature-of-technical-debt
- excessive-customization
- slow-incident-resolution
- customization-outside-version-control
- user-frustration
layout: solution
related_solutions:
- slug: authorization-concept
  similarity: 0.7
- slug: least-privilege
  similarity: 0.65
- slug: authorization
  similarity: 0.65
- slug: clear-ownership-model
  similarity: 0.65
- slug: clear-roles-and-ownership
  similarity: 0.6
- slug: feature-usage-measurement
  similarity: 0.6
---

## Description

Role model rationalization reconstructs a sprawling permission model from evidence — what users actually do, rather than what they have been granted — and pairs the reconstruction with a process that removes access as well as adding it. Both halves are necessary. Rationalizing without changing the process produces a clean model that regrows to its previous size within a few years, since the mechanism that caused the explosion is untouched. Changing the process without rationalizing leaves the existing sediment in place forever. The reconstruction is possible because most systems record what was actually used, and the gap between granted and used permissions is typically enormous. That gap is what makes the work tractable: the target model is derived from observed behavior rather than designed from an organizational chart that never matched reality anyway.

## How to Apply ◆

> The reason nobody removes a permission is that nobody can establish who depends on it — and usage data answers exactly that question.

- **Collect actual usage** over a period covering the full business cycle, including period-end and annual processes. Anything shorter will classify a legitimately rare permission as unused, and one such mistake will stop the programme.
- **Compare granted against used per user**, and treat the difference as the working set. This comparison alone usually establishes that a large majority of assigned permissions have never been exercised by the person holding them.
- **Derive candidate roles from clusters of actual usage**, then reconcile those clusters against job functions with the business. Roles derived purely from data are unmaintainable; roles derived purely from job descriptions do not match what people do. The reconciliation is the work.
- **Handle the exceptions explicitly** rather than widening a role to accommodate them. A user needing something outside their role should receive a separate, time-limited grant with a record, which is how the model stays coherent.
- **Remove in stages with a monitored period.** Withdraw the permission but log the attempt that would have used it, then remove for real once a cycle passes without a hit. This converts removal from a gamble into a measurement.
- **Establish the removal process before finishing the cleanup**: a leaver process that revokes, a mover process that removes the old as well as adding the new, and a periodic review with a named owner per role. Without these, the cleanup is a one-off.
- **Give every role an owner and a stated purpose.** An unowned role cannot be reviewed, and the absence of an owner is the condition under which roles become permanent.
- **Enforce separation-of-duty rules in the model**, not as a periodic audit finding. Conflicts detected at assignment time are prevented; conflicts detected annually are reported.
- **Report the model's size as a tracked measure**, so regrowth is visible early rather than being discovered at the next audit.

## Tradeoffs ⇄

> Rationalization restores a model that can be described and audited, but removing access carries a real risk of blocking legitimate work and the effort is substantial.

**Benefits:**

- The question of who can perform a sensitive action becomes answerable, which is the capability that both security and audit actually require.
- Excess access is removed, which reduces the exposure from a compromised account and from insider misuse.
- Provisioning becomes faster and more consistent, because a defined role exists to assign rather than a colleague's set to copy.
- Access reviews become meaningful, since managers are asked about a comprehensible role rather than a list of technical permissions.
- The removal process stops the regrowth, which is what distinguishes this from the periodic cleanups that most organizations have already tried.

**Costs and Risks:**

- Removing a permission that turns out to be needed blocks someone's work, and a few visible instances can end the programme.
- Usage data over an insufficient period systematically misclassifies rare but essential permissions, and the rarest are often the most critical.
- The reconciliation between usage clusters and job functions is slow, political, and requires business participation that is hard to obtain.
- Rationalization without process change is temporary, and the effort will have to be repeated within a few years.
- Some systems record usage poorly or not at all, in which case the evidential basis is weak and the work becomes far riskier.

## How It Could Be

An enterprise resource planning installation had 3,100 roles for 2,400 users. Usage was collected for thirteen months to cover the annual close. The comparison showed that across all users, roughly 71 percent of granted permissions had never been exercised by the holder. Clustering actual usage produced 140 candidate roles, which reconciliation with the business turned into 190 — the additional 50 covering legitimate variations the data had merged. Removal was staged: permissions were withdrawn but attempts logged, and over the following cycle 340 logged attempts identified genuinely needed access that the usage window had missed, all of which was restored before any user was blocked. The model ended at 190 roles with an owner each.

The process change was what made it stick, and it was the part the organization had skipped in two previous cleanups. A mover process that removed old access as well as granting new, a leaver process that revoked, quarterly review by role owners rather than by line managers, and a tracked count reported alongside other operational measures. Two years later the model stood at 210 roles rather than the several thousand it had reached after each previous cleanup. The security team's assessment was that the earlier attempts had failed not because the cleanup was wrong but because nothing had changed about how access was granted the day after it finished.
