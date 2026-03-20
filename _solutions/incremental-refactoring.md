---
title: Refactoring
description: Regular revision of the code to improve the internal structure
category:
- Code
- Process
quality_tactics_url: https://qualitytactics.de/en/maintainability/refactoring/
problems:
- spaghetti-code
- god-object-anti-pattern
- high-coupling-and-low-cohesion
- bloated-class
- excessive-class-size
- circular-dependency-problems
- code-duplication
- copy-paste-programming
- tight-coupling-issues
- monolithic-functions-and-classes
- refactoring-avoidance
- feature-creep-without-refactoring
- workaround-culture
- accumulation-of-workarounds
- increasing-brittleness
- clever-code
- complex-and-obscure-logic
- poor-encapsulation
- tangled-cross-cutting-concerns
- over-reliance-on-utility-classes
- global-state-and-side-effects
- hardcoded-values
layout: solution
---

## How to Apply ◆

> In legacy systems, refactoring must be introduced as a disciplined, continuous practice rather than a one-time cleanup project, because the scale of accumulated debt makes any attempt to address it all at once both impractical and risky.

- Apply the Boy Scout Rule consistently: whenever a developer works on a legacy module to fix a bug or add a feature, they leave that specific code incrementally cleaner than they found it, without touching unrelated areas. This keeps refactoring effort proportional to actual work rather than a separate cost.
- Before refactoring any legacy code, write characterization tests that capture the existing behavior — including bugs and undocumented quirks. These tests are not assertions about correctness; they record what the code actually does so that structural changes can be verified as behavior-preserving.
- Use named refactoring operations — Extract Method, Move Method, Replace Conditional with Polymorphism — rather than ad-hoc editing. Named operations have defined mechanics that reduce the risk of accidentally changing behavior in untested code paths.
- Commit each refactoring step separately from feature or bug-fix changes. In legacy codebases this discipline is especially important: mixing structural changes with behavioral ones makes it impossible to identify which change introduced a regression.
- Focus refactoring effort on code that is actively being modified. The deeply tangled module that nobody has touched in two years carries risk if refactored without a compelling reason; the payment processing class that receives new requirements every month is high-value and high-priority.
- Use automated IDE refactoring tools wherever possible. In legacy codebases with weak type systems, poorly named identifiers, or no test coverage, automated rename and extract operations are safer than manual editing because the tool tracks all references.
- Identify and address the most dangerous code smells first: God Objects that accumulate unrelated responsibilities, deeply nested conditional logic, and classes with hundreds of lines of duplicate code cause the most harm to maintainability and should be decomposed before adding new functionality.
- Track refactoring effort separately in sprint planning and communicate its value to stakeholders in terms of delivery speed: "this refactoring will reduce the time required to add new payment methods from two weeks to three days" is a concrete business case that justifies the investment.

## Tradeoffs ⇄

> Incremental refactoring is the only sustainable approach to improving legacy code quality over time, but it requires consistent team discipline and a reliable test safety net to avoid introducing new defects.

**Benefits:**

- Reduces the cost and risk of future changes by incrementally simplifying the code structure, making each subsequent modification to the legacy system faster and safer.
- Prevents the accumulation of new layers of technical debt by continuously improving the code that is actively being worked on, rather than allowing decay to compound.
- Builds genuine team understanding of the legacy codebase: developers who refactor a module learn its structure far more deeply than those who only read it, improving the team's collective ability to maintain it.
- Creates natural opportunities to add test coverage while verifying that refactoring preserved behavior, gradually building the safety net that the legacy codebase lacks.
- Avoids the high risk and organizational disruption of big-bang rewrites by delivering continuous structural improvement while keeping the system operational and feature delivery ongoing.

**Costs and Risks:**

- Refactoring legacy code without adequate test coverage is genuinely dangerous: a behavior-preserving transformation that appears safe can alter an undocumented side effect that other parts of the system depend on.
- Legacy codebases often contain deeply entangled modules where even small structural changes require modifications across many files, increasing merge conflict risk when multiple developers are active.
- Teams under constant pressure to deliver new features and fix production defects on legacy systems rarely protect refactoring time, causing the practice to fade quickly without explicit management support.
- Poorly executed refactoring — changing structure and behavior simultaneously, or making steps too large to reverse safely — can make legacy code harder to understand and maintain than before.
- Stakeholders who measure progress in visible features may perceive refactoring as non-productive, creating friction when developers advocate for structural improvements to a system that "already works."

## How It Could Be

> The following scenarios illustrate how incremental refactoring addresses the structural decay found in real legacy systems without requiring a disruptive overhaul.

A healthcare software company maintained a patient scheduling system where the core booking logic had grown to a single 2,400-line class over fifteen years of incremental feature additions. Every new scheduling rule was added as another branch in an already deeply nested conditional structure. The team introduced a policy of refactoring one method per ticket worked: whenever a developer touched the class for any reason, they extracted the method they modified into a clearly named function and committed that extraction separately before making the functional change. After eight months of consistent application, the class had been decomposed into eight focused collaborating classes, reducing the average time to implement new scheduling rules from four days to half a day.

A manufacturing company's inventory management system contained extensive copy-paste duplication across its stock calculation routines: the same rounding and currency conversion logic appeared in over forty separate methods, each having drifted slightly from the others over years of independent modification. The team spent two days writing characterization tests that captured the output of each method across a representative set of inputs. Using those tests as a safety net, they systematically applied Extract Method to create a single shared calculation function, then replaced all forty call sites. The characterization tests revealed three methods that had genuinely different behavior — intentional specializations that had never been documented — which the team preserved as explicit named variants. Defects in currency rounding dropped to zero in the six months following the refactoring.

A telecom provider operating a legacy billing platform found that new engineers typically took four to six months before they could safely modify the rating engine, a dense module with global state, hardcoded thresholds, and no tests. Rather than scheduling a dedicated refactoring project — two previous attempts had been cancelled when business priorities shifted — the team embedded refactoring directly into their bug-fix workflow. Every defect fix in the rating engine was preceded by a refactoring commit that isolated the relevant logic from global state. Over twelve months, the practice transformed the rating engine's structure sufficiently that new hire ramp-up time on that module dropped to six weeks, and the team was able to introduce a long-delayed multi-currency feature in three weeks instead of the estimated three months.
