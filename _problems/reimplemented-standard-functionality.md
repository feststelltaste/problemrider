---
title: Reimplemented Standard Functionality
description: Capability the package already provides has been built again as a custom development, adding maintenance burden while removing the benefit of buying the product.
category:
- Architecture
- Process
- Business
related_problems:
solutions:
- fit-to-standard-principle
- functional-gap-analysis
- standard-software
- variant-consolidation
- customization-cost-attribution
- feature-usage-measurement
- domain-immersion
- lightweight-design-review
- technology-radar
- strategic-code-deletion
layout: problem
---

## Description

Reimplemented standard functionality occurs when an organization builds, as a custom development inside a packaged system, capability that the package already offers. It happens because nobody established what the standard could do before deciding to build. The reasons are mundane: the product's documentation is large and unfamiliar, the standard implementation looks slightly different from what was requested, an external consultancy earns more from development than from configuration, or the requirement arrived phrased as a solution rather than a need. The organization then pays twice — the purchase price for functionality it does not use, and the ongoing maintenance for a version it must keep working itself. The custom implementation typically also lags: the vendor improves the standard feature over the years while the local copy remains as it was written.

## Indicators ⟡

- A vendor release note describes a feature you already have, and nobody is sure whether to switch
- Consultants propose development for requirements that sound generic rather than specific to your business
- Nobody on the team can say what the standard product does in an area without opening it and looking
- Custom developments carry names that closely mirror standard module names
- Training material for the standard product describes screens your users never see
- A requirement was implemented as stated by the requester rather than after asking what problem it solved

## Symptoms ▲

- [High Maintenance Costs](high-maintenance-costs.md)
<br/>  The organization maintains functionality it could have received as part of the product, including keeping it working across upgrades.
- [Increased Cost of Development](increased-cost-of-development.md)
<br/>  Effort is spent building and then extending capability that was already purchased and available.
- [Wasted Development Effort](wasted-development-effort.md)
<br/>  The initial build produced nothing the organization did not already have, and its cost is rarely recognized as waste because the result works.
- [Obsolete Technologies](obsolete-technologies.md)
<br/>  Custom implementations do not benefit from the vendor's improvements, so the local version falls behind the standard it replaced.
- [Testing Complexity](testing-complexity.md)
<br/>  Custom capability must be regression tested at every upgrade, while standard capability is tested by the vendor.
- [Difficulty Quantifying Benefits](difficulty-quantifying-benefits.md)
<br/>  The waste is invisible because the custom implementation works, so nothing prompts anyone to compare it against the standard.
- [Excessive Customization](excessive-customization.md)
<br/>  Each reimplementation adds to the volume of local code that every future change and upgrade must accommodate.

## Causes ▼

- [Inadequate Requirements Gathering](inadequate-requirements-gathering.md)
<br/>  Requirements are recorded as requested rather than investigated, so nobody asks whether the underlying need is already met.
- [Legacy Skill Shortage](legacy-skill-shortage.md)
<br/>  Nobody in the organization knows the product deeply enough to recognize that the requested capability already exists.
- [Dependency on Supplier](dependency-on-supplier.md)
<br/>  An implementation partner paid for development has no incentive to point out that configuration would suffice, and often does not know either.
- [Poor Documentation](poor-documentation.md)
<br/>  Product capability is documented by the vendor in volumes nobody reads, so what the standard offers is effectively unknown internally.
- [Market Pressure](market-pressure.md)
<br/>  Under time pressure, building the known thing is more predictable than investigating whether an unknown standard feature fits.
- [Eager to Please Stakeholders](eager-to-please-stakeholders.md)
<br/>  A requester who describes the solution they want gets it built, rather than being asked what outcome they need.

## Detection Methods ○

- For each substantial custom development, ask what standard capability was evaluated and rejected, and why; the absence of an answer is the finding
- Compare the custom development inventory against the product's module and feature list, looking for overlap in purpose
- Review vendor release notes from the last several years for features that duplicate something you maintain
- Ask the vendor or an independent expert to review the custom inventory; this is a service most vendors will provide
- Check whether any evaluation of the standard preceded the last five development decisions
- Look for custom developments whose functional description would apply equally to any organization in your sector

## Examples

A public sector organization running a document management platform had commissioned a custom approval workflow during implementation, at a cost of roughly nine months of consultancy. Seven years later, a new administrator working through the product's training material found that the standard product had shipped an equivalent workflow capability in the release before the one they had implemented on. The custom workflow had been built because the requirement, as written, specified a two-stage approval with a delegation rule — and the standard capability supported both, configured rather than coded. Nobody had looked. Migrating to the standard took six weeks and removed a component that had consumed regression testing effort at every upgrade for seven years.

An enterprise resource planning deployment showed the pattern at a smaller scale but higher frequency. A review of 61 custom developments against the product's capability list found 14 that duplicated standard functionality outright and a further 9 that duplicated it with a minor variation. The most instructive was a custom report that recalculated inventory valuation: the standard report produced the same numbers, but formatted the currency differently, and in 2013 a controller had asked for the format to be changed. Rather than the format, the calculation had been rebuilt.
