---
title: Rule-Based Systems
description: Defining rules that govern the behavior of the software
category:
- Architecture
- Code
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/rule-based-systems
problems:
- complex-and-obscure-logic
- legacy-business-logic-extraction-difficulty
- difficult-code-comprehension
- hardcoded-values
- spaghetti-code
- poor-domain-model
- maintenance-overhead
layout: solution
---

## How to Apply ◆

> In legacy systems, extracting tangled business logic into explicit rules makes behavior visible, testable, and modifiable by domain experts rather than requiring deep code archaeology.

- Identify business logic in the legacy system that is implemented as deeply nested conditionals, sprawling switch statements, or procedural code mixed with infrastructure concerns.
- Extract these decision points into a rule engine or a declarative rules format where each rule has a clear condition and action, making the logic readable without understanding the surrounding code.
- Involve domain experts in validating extracted rules against their understanding of the business, since legacy code often contains rules whose original purpose has been forgotten.
- Implement rules in a format that allows non-developers to review and potentially modify them — this reduces the bottleneck of requiring developer intervention for every business rule change.
- Add comprehensive tests for each extracted rule in isolation, then test rule interactions to catch conflicts or gaps.
- Maintain a rule catalog that documents the origin, purpose, and rationale for each rule, preventing future knowledge loss.

## Tradeoffs ⇄

> Rule-based systems make business logic explicit and maintainable but introduce a new layer of complexity that must be managed.

**Benefits:**

- Makes business logic visible and understandable to domain experts who cannot read the legacy codebase.
- Enables business rule changes without modifying application code, reducing change cycle time for regulatory or policy updates.
- Simplifies testing by allowing individual rules to be verified independently.
- Supports gradual extraction of logic from the legacy system — rules can be migrated incrementally without a big-bang rewrite.

**Costs and Risks:**

- Rule engines introduce a new technology dependency and require team expertise to manage effectively.
- Complex rule interactions can create emergent behavior that is difficult to predict and debug, especially when hundreds of rules interact.
- Performance overhead of rule evaluation may be significant for systems with large rule sets executing in real-time.
- Over-enthusiastic adoption can lead to moving logic into rules that would be better expressed in conventional code, making the system harder to understand.

## How It Could Be

> The following scenario shows how rule-based extraction tames legacy business logic complexity.

A health insurance company had a claims adjudication system where pricing logic was scattered across 50,000 lines of COBOL code with hundreds of nested IF-ELSE blocks representing different plan types, provider networks, and regulatory overrides. The team extracted these decisions into a modern rules engine, creating approximately 800 individual rules organized by business domain. For the first time, the compliance team could review and approve rule changes directly rather than relying on developers to interpret the COBOL. The extraction also revealed 34 rules that conflicted with each other and 12 that were no longer applicable due to regulatory changes that had never been fully reflected in the code. The rule-based system reduced the time to implement annual regulatory updates from three months to three weeks.
