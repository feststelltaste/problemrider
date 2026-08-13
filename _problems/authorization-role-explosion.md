---
title: Authorization Role Explosion
description: The role and permission model has grown to thousands of entries that only ever accumulate, so nobody can say who is able to do what.
category:
- Security
- Operations
- Process
related_problems:
solutions:
- role-model-rationalization
- authorization-concept
- role-based-access-control
- least-privilege
- domain-based-authorization-concept
- clear-ownership-model
- customization-under-version-control
- security-audits
- attribute-usage-analysis
- quality-ratchet
layout: problem
---

## Description

Authorization role explosion occurs when the permission model of a packaged system grows continuously and is never reduced. Each new requirement produces a new role rather than a change to an existing one, because changing an existing role risks removing access someone depends on and nobody can determine who that is. Roles are copied for individuals, accumulate permissions across job changes, and outlive the positions they were created for. The result is a model that no longer describes the organization: it is a sedimentary record of every access request ever granted. The practical consequences are that nobody can answer who can perform a sensitive action, access reviews become impossible to conduct meaningfully, and every audit produces findings that are addressed by adding more roles.

## Indicators ⟡

- The number of roles is close to, or exceeds, the number of users
- Roles carry names like a person's name, a project that ended, or a version number
- New access is granted by copying an existing user's roles rather than by assigning a defined role
- Access reviews are conducted by asking managers to approve lists they cannot meaningfully assess
- Nobody can answer, within a day, who is able to perform a specific sensitive transaction
- Roles are only ever added; there is no process by which any role is removed
- Access problems are resolved by adding a permission, never by examining why the existing one was insufficient

## Symptoms ▲

- [Authorization Flaws](authorization-flaws.md)
<br/>  Accumulated permissions produce combinations nobody intended, including access that violates separation-of-duty requirements.
- [Regulatory Compliance Drift](regulatory-compliance-drift.md)
<br/>  Access controls that cannot be described cannot be demonstrated to an auditor, regardless of whether they are adequate in practice.
- [Slow Incident Resolution](slow-incident-resolution.md)
<br/>  Determining whether an action was permitted, and by whom, requires reconstructing an effective permission set across many overlapping roles.
- [Increased Manual Work](increased-manual-work.md)
<br/>  Provisioning, reviewing, and correcting access consume continuous administrative effort that grows with the size of the model.
- [Invisible Nature of Technical Debt](invisible-nature-of-technical-debt.md)
<br/>  The permission model is rarely counted as debt at all, so its accumulation is not reported and its cost is not attributed to anything.
- [User Frustration](user-frustration.md)
<br/>  Users are blocked by missing permissions and over-granted elsewhere, and the resolution cycle for each is slow.
- [Lack of Ownership and Accountability](lack-of-ownership-and-accountability.md)
<br/>  Roles created ad hoc have no owner, so there is nobody to ask whether one is still needed or what it was for.

## Causes ▼

- [Excessive Customization](excessive-customization.md)
<br/>  Customization multiplies the transactions and objects that require authorization, and the permission model grows with them.
- [Fear of Breaking Changes](fear-of-breaking-changes.md)
<br/>  Removing a permission might block someone, and since nobody can determine who, the safe action is always to add rather than to change.
- [Customization Outside Version Control](customization-outside-version-control.md)
<br/>  Role definitions that carry no history and no authorship cannot be reviewed, so their growth is unexamined.
- [Lack of Ownership and Accountability](lack-of-ownership-and-accountability.md)
<br/>  Without an owner for the authorization model as a whole, no one is responsible for its coherence and everyone is responsible for their own requests.
- [Short-Term Focus](short-term-focus.md)
<br/>  Granting the requested access resolves today's blockage; redesigning the model does not, and is therefore never the chosen action.
- [Inadequate Requirements Gathering](inadequate-requirements-gathering.md)
<br/>  Access is requested in terms of what a specific person needs to do today rather than in terms of a role the organization actually has.

## Detection Methods ○

- Count roles against users and against job functions; a role count exceeding the number of distinct job functions by an order of magnitude is diagnostic
- Identify roles assigned to no user, to exactly one user, or unchanged for several years
- Compute effective permissions for a sample of users and compare against what their job requires
- Test the model with a specific question — who can approve a payment above a threshold — and time the answer
- Check whether any role has ever been removed, and what process would be used if one were
- Look for role names containing personal names, project names, or dates

## Examples

An enterprise resource planning installation supporting 2,400 users had 3,100 roles. Analysis found that 890 were assigned to exactly one user, 400 to none, and that the largest single category consisted of roles created by copying another user's set during onboarding and then amended. Asked by an auditor who could post a journal entry above a materiality threshold, the team took nine days to produce an answer, and the answer identified 34 users of whom the finance director recognized 19 as appropriate. The remaining 15 had accumulated the permission through role combinations created for other purposes, in three cases through a role named after a project that had closed in 2017.

The accumulation mechanism was visible in how a typical case arose. A user moved from procurement to finance. Their existing roles were retained, because removing them might break something and nobody could establish what depended on them, and finance roles were added. Over eleven years the organization had processed roughly 4,000 such moves. The permission model contained no mechanism by which anything was ever taken away, and every audit finding had historically been closed by creating a more restrictive role and assigning it in addition to what was already there.
