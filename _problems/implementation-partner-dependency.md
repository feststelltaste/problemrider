---
title: Implementation Partner Dependency
description: Only the external consultancy understands how the system was built, so the organization cannot change, assess, or leave its own installation without them.
category:
- Dependencies
- Team
- Management
related_problems:
solutions:
- vendor-management-practice
- knowledge-rotation
- internal-technical-coaching
- code-reading-sessions
- documentation-as-code
- architecture-decision-records
- customization-under-version-control
- pair-and-mob-programming
- technical-skills-development
- risk-quantification
- structured-onboarding-program
layout: problem
---

## Description

Implementation partner dependency occurs when the knowledge of how a packaged system was configured, extended, and integrated resides with an external consultancy rather than inside the organization that owns it. It develops naturally: the partner implements, internal staff operate, and the understanding of why things are as they are never transfers because no one arranges for it to. The dependency is more constraining than ordinary supplier dependency, because it concerns the organization's own configuration rather than a product. The partner can be replaced only by another partner willing to spend months learning what the first one knows, and that cost makes the incumbent's rates effectively unchallengeable. Organizations frequently do not recognize the position until they attempt to change partners or bring work in-house and discover that neither is available.

## Indicators ⟡

- Any non-trivial change requires the partner, including changes that appear to be configuration
- Internal staff can operate the system but cannot explain why it behaves as it does
- Estimates from the partner cannot be independently assessed, and are accepted because there is no basis to challenge them
- Documentation of the implementation is thin, out of date, or consists of the partner's original design documents
- The same individual consultants have worked on the account for years and their absence is a scheduling risk
- Obtaining a competing quote is considered impractical because a competitor would have to learn the system first

## Symptoms ▲

- [Knowledge Dependency](knowledge-dependency.md)
<br/>  Critical understanding of the organization's own system sits outside it, and cannot be drawn on except through a commercial arrangement.
- [High Maintenance Costs](high-maintenance-costs.md)
<br/>  Rates cannot be tested against alternatives, and work that internal staff could do must be bought instead.
- [Slow Feature Development](slow-feature-development.md)
<br/>  Changes move at the speed of the partner's availability and contracting cycle rather than the organization's need.
- [Vendor Relationship Strain](vendor-relationship-strain.md)
<br/>  An imbalanced dependency produces resentment on the customer side and complacency on the supplier side, and both show in the relationship.
- [Modernization Strategy Paralysis](modernization-strategy-paralysis.md)
<br/>  Assessing options requires understanding the current installation, which the organization cannot do without asking the party with an interest in the answer.
- [Legacy Skill Shortage](legacy-skill-shortage.md)
<br/>  Internal capability never develops, because every occasion on which it might have been built was outsourced.
- [Poor Documentation](poor-documentation.md)
<br/>  Documentation is a deliverable the partner is paid to produce and has no operational need to maintain, so it stops matching reality quickly.

## Causes ▼

- [Dependency on Supplier](dependency-on-supplier.md)
<br/>  The commercial relationship was structured around delivery rather than around capability transfer, and nothing in it required knowledge to move.
- [Legacy Skill Shortage](legacy-skill-shortage.md)
<br/>  Without internal people who could receive the knowledge, there was nobody for the partner to hand over to even where handover was intended.
- [Staff Availability Issues](staff-availability-issues.md)
<br/>  Internal staff are fully committed to operations, so participating in implementation work is deferred and the transfer never happens.
- [Short-Term Focus](short-term-focus.md)
<br/>  Buying the work is faster than building the capability, and this comparison is made on every occasion with the same outcome.
- [Poor Contract Design](poor-contract-design.md)
<br/>  Contracts specify deliverables rather than knowledge transfer, documentation standards, or the customer's ability to continue without the supplier.
- [High Turnover](high-turnover.md)
<br/>  Internal knowledge that did develop leaves with the people who held it, while the partner's account team remains stable, widening the asymmetry.

## Detection Methods ○

- Ask what proportion of changes in the last year required the partner, and how many were configuration rather than development
- Establish whether anyone internal could explain the three most important customizations without consulting the partner
- Estimate what a competing supplier would need to spend to become productive on the account; that figure is the switching cost
- Check whether the contract contains any obligation regarding documentation quality or knowledge transfer, and whether it has been enforced
- Test the position: assign an internal person a real change and measure what they can accomplish unaided
- Review whether partner estimates have ever been successfully challenged, and on what basis

## Examples

A regional utility had run an enterprise resource planning installation for nine years, implemented and maintained throughout by one consultancy. When procurement sought competitive quotes for a planned extension, two suppliers declined to bid and the third quoted an amount that included four months of familiarization. The incumbent's quote was lower and was accepted, as it had been for nine years. An internal review afterwards established that no employee could describe how the organization's own pricing rules were implemented, that the design documentation dated from the original implementation, and that three of the partner's consultants held effectively all operational knowledge of the installation.

The organization's response is instructive because the obvious one would not have worked. They did not attempt to replace the partner, which would have realized the switching cost immediately. Instead they added two internal people to the next three change projects as participants rather than observers, required that the partner's work land in an internal repository with the design recorded, and made a rule that the partner could not be the only party who had touched any area. After eighteen months the internal team handled roughly a third of changes unaided, and the next competitive quote drew three bids — not because the system had become simpler, but because it had become describable to an outsider.
