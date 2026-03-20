---
title: Standard Software
description: Use proven standard software instead of developing ordinary functionality yourself
category:
- Architecture
- Dependencies
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/standard-software
problems:
- maintenance-overhead
- high-maintenance-costs
- maintenance-cost-increase
- obsolete-technologies
- legacy-skill-shortage
- increased-cost-of-development
- slow-feature-development
- technology-lock-in
layout: solution
---

## How to Apply ◆

> Legacy systems frequently contain custom implementations of functionality that is now available as mature, well-maintained standard software — replacing these custom components reduces maintenance burden significantly.

- Audit the legacy system to identify custom-built components that replicate functionality available in proven standard software, such as custom authentication systems, report generators, workflow engines, or scheduling frameworks.
- Evaluate standard software candidates against the specific requirements encoded in the legacy implementation, paying special attention to edge cases and customizations that may not be supported out of the box.
- Plan migrations from custom to standard software incrementally, running the custom and standard solutions in parallel during transition periods to validate behavior parity.
- Prioritize replacement of custom components that are the most expensive to maintain, require the most specialized knowledge, or present the greatest security risk.
- Accept that standard software may not replicate 100% of legacy behavior — evaluate whether the unsupported edge cases are still genuine requirements or legacy artifacts that can be eliminated.
- Negotiate support and maintenance agreements for standard software to ensure the organization has access to vendor expertise and timely security patches.

## Tradeoffs ⇄

> Standard software eliminates maintenance burden for common functionality but introduces vendor dependencies and may require adapting business processes.

**Benefits:**

- Dramatically reduces the maintenance effort for functionality that is not a competitive differentiator, freeing developers to focus on business-critical custom features.
- Benefits from the vendor's ongoing investment in security patches, performance improvements, and feature development.
- Reduces the risk of knowledge loss since standard software has broader community documentation and available expertise compared to bespoke legacy components.
- Accelerates developer onboarding because team members are more likely to have experience with standard tools.

**Costs and Risks:**

- Introduces vendor dependency and the risk that the vendor discontinues the product, changes licensing terms, or raises prices.
- Standard software may impose workflow constraints that require the organization to adapt its processes, which can face resistance from users accustomed to legacy behavior.
- Migration from custom to standard software requires careful data migration and integration work that is often underestimated.
- Over-reliance on standard software for core business logic can limit competitive differentiation and flexibility.

## Examples

> The following scenario illustrates the benefits and challenges of replacing custom legacy components with standard software.

A mid-size manufacturing company had maintained a custom-built ERP module for inventory management for 12 years. The module required two full-time developers to maintain, ran on an outdated application server, and could only be modified by a single developer who understood its complex stored procedure architecture. After evaluating three commercial inventory management systems, the company selected one that covered 85% of their requirements out of the box. The remaining 15% consisted of custom labeling workflows that the team implemented as extensions. The migration took eight months but eliminated the ongoing maintenance cost of the custom module and removed a critical single point of failure in their development team. Two years later, the standard software vendor had delivered reporting features that the custom system never had the resources to build.
