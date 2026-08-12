---
title: Feature Usage Measurement
description: Instrument the system to record which features are actually used and by whom, so that maintenance effort and deletion decisions rest on evidence.
category:
- Business
- Process
- Requirements
problems:
- gold-plating
- feature-creep
- feature-factory
- high-maintenance-costs
- code-duplication
- delayed-value-delivery
- system-stagnation
- resource-waste
- wasted-development-effort
- increased-cost-of-development
- reduced-innovation
- product-direction-chaos
- budget-overruns
- duplicated-work
- maintenance-cost-increase
- project-resource-constraints
layout: solution
---

## Description

Feature usage measurement is the instrumentation of a system to record which of its capabilities are actually exercised, how often, and by which kinds of user. It answers a question that legacy organizations are usually unable to answer at all: of everything this system does, what matters. The absence of that answer has two expensive consequences. Maintenance effort is spread evenly across features regardless of their value, so the code path used by four people a year receives the same protection as the one used continuously. And nothing is ever removed, because removal requires someone to assert that a feature is unused, and without data nobody will take that risk. Every unremoved feature is permanent weight: code to maintain, tests to keep passing, and a constraint on every future change. Measurement converts deletion from a gamble into a decision.

## How to Apply ◆

> Legacy systems accumulate features across decades of requests, and the usual estimate that a substantial share are rarely or never used is almost always confirmed once someone finally measures.

- **Instrument at the level of user-meaningful capability**, not at the level of function calls. "How many users generated a custom report this quarter" is actionable; a call count for an internal method is not.
- **Record who uses it, not only how often.** A feature used twice a year by the regulator is not a candidate for deletion, and a feature used constantly by one departing customer is a different kind of risk. Aggregate counts conceal both.
- **Measure over a full business cycle** before drawing conclusions. Annual reporting features, year-end processes, and seasonal capabilities are invisible in a quarter of data, and deleting one of those is the mistake that discredits the entire practice.
- **Start with the areas where you suspect the answer**, rather than instrumenting everything. The features that are expensive to maintain and believed to be marginal are where the measurement pays for itself first.
- Combine usage with **maintenance cost** — change frequency, defect count, incident involvement. The deletion candidates are the intersection of rarely used and expensive to keep, and that intersection is usually small, obvious, and worth acting on.
- **Announce and observe before removing.** Mark the feature as deprecated, notify identifiable users, and then instrument the period after the announcement. Removing on the strength of data alone occasionally finds the one user who matters, in an unpleasant way.
- **Remove in a revertable way**: disable behind a flag first, wait a full cycle, then delete the code. The gap between disabling and deleting is where the unmeasured consumers announce themselves.
- Use the data to **inform building as well as removing**. A team that ships features and never learns whether they are used has no feedback loop at all, and this is the mechanism by which a feature factory sustains itself.
- **Respect privacy constraints** in what is collected. Usage measurement rarely needs to identify individuals; aggregate by role, tenant, or segment, and involve the data protection function before instrumenting anything user-facing.

## Tradeoffs ⇄

> Usage data enables deletion and directs maintenance effort, but instrumentation is work, the data is easy to misread, and removing a feature is irreversible in a way that keeping one is not.

**Benefits:**

- Deletion becomes possible, and deletion is the only intervention that reduces a system's size and complexity absolutely rather than reorganizing it.
- Maintenance and testing effort can be concentrated on what is actually used, which is a substantial reallocation in a system with a long tail of marginal features.
- Requests for new features can be discussed against evidence about how comparable past features fared, which is the strongest available argument against building on speculation.
- The value delivered by the team becomes visible in terms other than output volume, which changes how a feature factory is discussed.
- Modernization scope shrinks. Features confirmed unused do not need to be migrated, and this frequently removes a meaningful fraction of the work.

**Costs and Risks:**

- Instrumentation must be built and maintained, and instrumenting a legacy system with no telemetry framework is not a small task.
- Low usage is not the same as low value. Regulatory, contractual, and disaster-recovery capabilities may be exercised almost never and be indispensable.
- Removing a feature is irreversible in practice, and a wrong removal damages trust in the data and in the team far more than an unremoved feature costs.
- Measurement windows that miss annual or seasonal cycles produce confidently wrong conclusions.
- Usage tracking raises legitimate privacy concerns and may require legal review, consent, or restriction to aggregate data.

## How It Could Be

A team maintaining a corporate expense management system carried 340 distinct capabilities accumulated over fourteen years, and every modernization estimate collapsed under the weight of migrating all of them. They instrumented the top-level user actions and measured for thirteen months to cover a full fiscal cycle. The result: 61 capabilities had zero recorded use, and a further 88 were used by fewer than five people in the entire year. Crossing that with maintenance data showed that 30 of the unused capabilities sat in modules with high change frequency, meaning developers were repeatedly maintaining code that nothing exercised. After a deprecation announcement and a further quarter of observation, 54 capabilities were removed. Two produced complaints, both from a single finance user, and both were restored within a day. The migration scope shrank by roughly a fifth.

The measurement also changed a decision about what to build. The team had a longstanding request for a more sophisticated approval routing engine, justified by the claim that the existing simple rules were insufficient. The usage data showed that 94 percent of expense reports followed one of three routing paths, and that the flexible options in the current engine — added five years earlier for the same reason — had been configured by two of forty departments. Rather than building the more sophisticated engine, the team simplified the existing one around the three dominant paths and handled the remainder manually. The maintenance burden of the routing subsystem fell substantially, and the feature request was withdrawn once its sponsor saw the distribution.
