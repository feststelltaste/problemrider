---
title: Breaking Changes
description: API updates break existing client integrations, causing compatibility
  issues and forcing costly emergency fixes.
category:
- Architecture
- Code
- Testing
related_problems:
- slug: api-versioning-conflicts
  similarity: 0.65
- slug: rapid-system-changes
  similarity: 0.65
- slug: regression-bugs
  similarity: 0.6
- slug: increasing-brittleness
  similarity: 0.55
- slug: fear-of-breaking-changes
  similarity: 0.55
- slug: brittle-codebase
  similarity: 0.55
layout: problem
---

## Description

Breaking changes occur when modifications to APIs, interfaces, or system behaviors cause existing client integrations to fail or behave incorrectly. These changes violate backward compatibility expectations and force clients to update their code, often unexpectedly and on short notice. Breaking changes can severely damage relationships with integration partners, cause production outages, and create emergency support situations.

## Indicators ⟡

- Client applications stop working after API updates
- Integration partners report sudden failures in their systems
- Support tickets spike immediately following API releases
- Client developers express frustration about unexpected changes
- Emergency rollbacks are needed to restore client functionality

## Symptoms ▲

- [Cascade Failures](cascade-failures.md)
<br/>  Breaking API changes cause dependent services to fail in chain reaction as each tries to use the modified interface.
- [Regression Bugs](regression-bugs.md)
<br/>  Previously working client integrations start exhibiting bugs after API changes break their assumptions.
- [Stakeholder Dissatisfaction](stakeholder-dissatisfaction.md)
<br/>  Integration partners and customers lose trust when their systems break due to unexpected API changes.
- [Budget Overruns](budget-overruns.md)
<br/>  Emergency fixes and unplanned client migration work caused by breaking changes drive costs beyond plan.
- [Fear of Breaking Changes](fear-of-breaking-changes.md)
<br/>  Repeated incidents of breaking changes create organizational anxiety about any future API modifications.

## Causes ▼
- [API Versioning Conflicts](api-versioning-conflicts.md)
<br/>  Poor API versioning practices make it impossible to evolve APIs without breaking existing consumers.
- [Change Management Chaos](change-management-chaos.md)
<br/>  Changes deployed without proper coordination or impact assessment break client integrations unexpectedly.
- [Legacy API Versioning Nightmare](legacy-api-versioning-nightmare.md)
<br/>  Without proper versioning strategies, API modifications inevitably break existing client integrations.
- [Rapid System Changes](rapid-system-changes.md)
<br/>  Rapid modifications to APIs and architecture increase the likelihood of breaking existing integrations and functionality.

## Detection Methods ○

- **Integration Test Monitoring:** Automated tests that verify API compatibility with existing client patterns
- **Client Usage Analytics:** Monitor how different API endpoints and parameters are actually used
- **Version Compatibility Testing:** Test new API versions against existing client code and integration patterns
- **Client Feedback Channels:** Establish communication channels for clients to report compatibility issues
- **Change Impact Assessment:** Systematic evaluation of how proposed changes affect existing integrations
- **Breaking Change Alerts:** Automated detection of changes that could break existing client code

## Examples

An e-commerce API changes the data structure of product information responses, moving the price field from a simple number to a complex object with currency and tax information. Hundreds of client applications that parse the price field directly break immediately, causing shopping cart failures and order processing issues across multiple retail websites. The API provider must maintain both old and new response formats while clients scramble to update their code. Another example involves a payment processing API that changes authentication requirements without sufficient notice, causing all client transactions to fail during peak shopping hours, resulting in millions of dollars in lost sales and emergency support calls.
