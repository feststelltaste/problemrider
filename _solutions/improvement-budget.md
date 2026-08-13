---
title: Improvement Budget
description: Reserve a fixed, protected share of every period's capacity for maintenance, refactoring, and risk reduction, so improvement never competes with features for approval.
category:
- Management
- Process
- Code
problems:
- maintenance-paralysis
- increasing-brittleness
- increased-technical-shortcuts
- feature-creep-without-refactoring
- system-stagnation
- delayed-bug-fixes
- partial-bug-fixes
- maintenance-cost-increase
- short-term-focus
- time-pressure
- reduced-innovation
- inability-to-innovate
- high-technical-debt
- accumulation-of-workarounds
- competing-priorities
- developer-frustration-and-burnout
- high-turnover
- increased-stress-and-burnout
- reduced-individual-productivity
- team-demoralization
- tool-limitations
- unmotivated-employees
- deadline-pressure
- maintenance-bottlenecks
- overworked-teams
- reduced-team-productivity
- refactoring-avoidance
- test-debt
- brittle-codebase
layout: solution
---

## Description

An improvement budget is a fixed share of each period's capacity — commonly ten to twenty percent — reserved for work that improves the system rather than extending it: refactoring, test coverage, dependency upgrades, tooling, documentation, and removal of dead code. Its defining property is that the budget is allocated once, at the level of policy, rather than justified item by item. This is the entire point. Improvement work loses every individual comparison against a feature, because its benefit is diffuse and deferred while the feature's is specific and immediate. A team that must win that argument each time will lose it each time, which is why systems degrade even when everyone involved agrees that maintenance matters. Moving the decision from the item level to the capacity level is what makes improvement structurally possible.

## How to Apply ◆

> Legacy systems accumulate degradation faster than teams can address it opportunistically, and the modules most in need of improvement are usually the ones nobody wants to touch without protected time.

- Agree the **share explicitly with whoever controls the team's capacity**, and record it where it can be pointed at later. Ten percent is enough to stop degradation in a stable system; a system already in trouble typically needs twenty percent or more for a sustained period before the trend reverses.
- Make the budget **capacity, not calendar time**. "Every Friday" fails on the first busy Friday, and then every Friday after that. A reserved share of each iteration's capacity survives pressure better because taking from it requires an explicit decision rather than a quiet default.
- **Let the team decide what the budget is spent on**, within a stated scope. The team knows which modules cost it the most, and requiring approval for each item reintroduces exactly the per-item competition the budget exists to eliminate.
- Prioritize the budget with **evidence rather than preference**: change frequency crossed with defect density identifies the areas where improvement returns the most. Refactoring a module that nobody has touched in four years is effort that returns nothing, however unpleasant that module is to read.
- Require the same **visibility as feature work** — the items appear on the same board, in the same review, with the same definition of done. Invisible improvement work is indistinguishable from no improvement work when its value is questioned six months later.
- Record the **outcome of what the budget bought** in concrete terms: build time reduced from eleven to four minutes, this class of production defect eliminated, this dependency now supported again. Improvement budgets are cut when their effects cannot be named, and the effects are nearly always nameable if someone writes them down at the time.
- Define in advance the **conditions under which the budget may be suspended** — a genuine production emergency, a hard regulatory deadline — and require that suspended capacity is repaid rather than forgiven. Without a repayment rule, suspension becomes permanent by accumulation.
- Pair the budget with the **boy scout rule** for opportunistic improvement: small cleanups within the area a change already touches are part of normal work and are not charged to the budget. The budget funds the improvements too large to happen incidentally.
- Review the share **quarterly against the trend**, not the sentiment. If defect rates, cycle times, and incident frequency are still worsening, the budget is too small to matter and should be increased or discontinued honestly rather than maintained as a gesture.

## Tradeoffs ⇄

> A protected budget is the only reliable way to fund improvement in a feature-driven organization, but it commits capacity in advance to work whose returns are real, delayed, and hard to attribute.

**Benefits:**

- Improvement work actually happens, rather than being perpetually scheduled for after the current deadline — a condition that no legacy system has ever reached.
- The team stops having to justify maintenance individually, which removes a recurring and demoralizing argument and a substantial amount of management overhead.
- Degradation slows measurably. Systems with a sustained improvement budget show flatter growth in defect rates, build times, and change costs than comparable systems without one.
- Developers regain some control over the environment they work in, which is one of the strongest predictors of retention among maintainers of difficult systems.
- The gradual accumulation of shortcuts under deadline pressure becomes visible, because the budget provides an obvious place to repay them and their absence from it becomes conspicuous.

**Costs and Risks:**

- Ten to twenty percent of capacity genuinely does not go to features, and in a capacity-constrained team this is a real reduction in feature delivery that must be acknowledged rather than argued away.
- Budgets are cut first under pressure. A budget that survives only in calm periods provides little value, since degradation accelerates precisely during the busy ones.
- Without evidence-based selection, the budget can be spent on the improvements that are most satisfying rather than most valuable — rewriting a tidy module while the genuinely dangerous one remains untouched.
- Returns are delayed and diffuse, making the budget hard to defend against a stakeholder who wants attribution for this quarter.
- A budget can become an alibi: the existence of ten percent improvement capacity can be used to argue that the system's problems are being handled when the actual need is several times larger.

## How It Could Be

A team maintaining a manufacturing execution system had a build that took thirty-eight minutes, a test suite that failed intermittently, and four dependencies past end of support. Every attempt to address these lost to the feature backlog for two years. Their engineering manager negotiated a fifteen percent improvement budget with the product director on a six-month trial, with the explicit condition that outcomes would be reported each month. In the first quarter the team cut the build to nine minutes, quarantined and fixed the twelve flakiest tests, and upgraded two dependencies. The reported outcome that secured the budget permanently was not any of those directly: it was that the number of feature items completed per month rose by eighteen percent, because a nine-minute build changed how often developers could integrate.

A different team used their budget more strategically. Rather than distributing it across many small cleanups, they spent two consecutive quarters on a single subsystem that their change-frequency and defect data identified as responsible for roughly forty percent of production incidents despite being about eight percent of the codebase. The work was unglamorous — extracting a 3,000-line class, adding characterization tests, and removing three layers of accumulated workarounds. Incidents from that subsystem fell to near zero over the following year, and the on-call rotation, which had been a significant factor in two resignations, stopped being a reason people left.
