---
title: Excessive Customization
description: So much customer-specific or site-specific behavior accumulates that
  no two installations are alike, and every change has to be validated against every
  variant.
category:
- Architecture
- Business
- Process
related_problems:
- slug: customization-outside-version-control
  similarity: 0.7
- slug: reimplemented-standard-functionality
  similarity: 0.7
- slug: core-modification-of-standard-software
  similarity: 0.7
- slug: custom-report-sprawl
  similarity: 0.7
- slug: low-code-customization-sprawl
  similarity: 0.65
- slug: upgrade-blocked-by-customization
  similarity: 0.65
solutions:
- explicit-extension-points
- customization-cost-attribution
- variant-consolidation
- feature-usage-measurement
- attribute-usage-analysis
- product-strategy-alignment
- explicit-prioritization-framework
- definition-of-ready
- modularization-and-bounded-contexts
- feature-toggles
- standard-software
- decision-rights-and-escalation
- total-cost-of-ownership-transparency
- large-scale-refactoring
- fit-to-standard-principle
- role-model-rationalization
layout: problem
---

## Description

Excessive customization occurs when a system accumulates so much customer-specific, site-specific, or department-specific behavior that there is no longer a single product — there is a family of divergent variants that happen to share a name. Each individual customization was justified: a customer needed something, a deal depended on it, a department had a genuinely different process. Collectively they destroy the economics of the product, because every change must now be designed against every variant, tested against every variant, and deployed to installations that each behave slightly differently. The condition is self-reinforcing. Once upgrading is expensive, customers fall behind, and each installation drifts further from every other, which makes the next upgrade more expensive still.

## Indicators ⟡

- No two installations run the same configuration, and nobody can produce a list of the differences
- Estimating a change requires asking which customers it affects, and the answer takes days to establish
- Some customers are several versions behind and upgrading them is treated as a project rather than a routine
- Sales commitments regularly include behavior that does not exist yet and will apply to one customer only
- The test suite has customer-specific cases, or worse, testing is done per installation after deployment
- New developers are told that a module "works differently for the big client" and there is no document explaining how
- Nobody can say what the standard product does without qualifying it

## Symptoms ▲

- [High Maintenance Costs](high-maintenance-costs.md)
<br/>  Every variant carries its own maintenance burden, and the total grows with the number of installations rather than with the size of the product.
- [Increased Cost of Development](increased-cost-of-development.md)
<br/>  A change that would be small in a single-variant product must be designed, implemented, and verified against every divergent installation.
- [Testing Complexity](testing-complexity.md)
<br/>  The number of configurations that must be verified multiplies with each customization, and full coverage of the combinations quickly becomes impossible.
- [Slow Feature Development](slow-feature-development.md)
<br/>  New functionality has to accommodate every existing variant before it can ship, which turns straightforward work into an exercise in compatibility.
- [Increased Bug Count](increased-bug-count.md)
<br/>  Defects appear in specific variants under specific configurations, and the combinations that were never tested are exactly where they surface.
- [Regression Bugs](regression-bugs.md)
<br/>  A change verified against the standard product breaks a customer whose variant depended on the previous behavior in a way nobody had recorded.
- [Long Release Cycles](long-release-cycles.md)
<br/>  Releasing means validating against many installations, which lengthens the cycle until releases become infrequent enough to be risky in themselves.
- [Knowledge Silos](knowledge-silos.md)
<br/>  Each variant tends to be understood by whoever built it, and that knowledge is rarely written down because the variant was supposed to be temporary.
- [High Technical Debt](high-technical-debt.md)
<br/>  Conditional branches for specific customers accumulate throughout the codebase and are never removed, because nobody is certain who still depends on them.

## Causes ▼

- [Eager to Please Stakeholders](eager-to-please-stakeholders.md)
<br/>  Every customer request is accepted because refusing feels like poor service, and the cumulative cost of acceptance is invisible at the moment each decision is made.
- [Market Pressure](market-pressure.md)
<br/>  Competitive deals are won by promising to accommodate whatever the prospect asks for, and the engineering consequence arrives after the contract is signed.
- [Feature Creep](feature-creep.md)
<br/>  Scope expands continuously without anything being removed, and customer-specific behavior is the form that expansion takes in a product with many installations.
- [Inadequate Requirements Gathering](inadequate-requirements-gathering.md)
<br/>  Requests are implemented as stated rather than investigated, so a need that several customers share is built several times as several variants.
- [Short-Term Focus](short-term-focus.md)
<br/>  The immediate deal is worth more than the long-term maintainability of the product, and this trade is made repeatedly by people who never carry its cost.
- [Product Direction Chaos](product-direction-chaos.md)
<br/>  Without a clear definition of what the standard product is, there is no basis on which any request could be declined.
- [Project Authority Vacuum](project-authority-vacuum.md)
<br/>  No one has the standing to refuse a customization, so the default answer is yes and the decision is never actually made by anyone.

## Detection Methods ○

- Count the configuration flags, feature toggles, and customer-specific branches in the codebase, and check the trend over the last two years
- Search the code for conditionals naming a specific customer, site, or tenant — these are rarely documented anywhere else
- Compare the deployed configuration across installations and count the fields that differ
- Measure the distribution of version numbers across installations; a wide spread indicates upgrading has become expensive
- Track how much of each release's effort goes to accommodating variants rather than to new capability
- Ask how long it takes to answer "which customers does this change affect" — if it is more than an hour, the variance is no longer tracked
- Review the last ten customer commitments and count how many introduced behavior that applies to exactly one installation

## Examples

A mid-sized vendor of clinical scheduling software had 34 hospital installations built from one codebase over eleven years. Each hospital had negotiated adjustments during procurement: different rules for how a cancelled appointment released its slot, different escalation paths, different report layouts, and in four cases a different definition of what counted as a completed visit. None of this was configuration — it was conditional logic in the codebase keyed to an installation identifier. Adding a straightforward feature meant reading the module, finding the seven customer-specific branches inside it, and reasoning about each. Their release cycle had grown from six weeks to nine months, and eleven hospitals were running versions more than two years old because the upgrade cost per site had grown to several weeks of consultancy.

The self-reinforcing character was visible in how the situation had developed. Early customizations were small and the product's economics absorbed them. As the count grew, each release became more expensive to validate, so releases became less frequent. Less frequent releases meant customers waited longer for requests, which made them more insistent that their requests be met exactly, which produced more customizations. By the time the vendor recognized the pattern, roughly 40 percent of engineering capacity went to reconciling variants and no single person could describe what the standard product did.
