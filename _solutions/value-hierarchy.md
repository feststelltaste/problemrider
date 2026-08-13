---
title: Value Hierarchy
description: Maintain an explicit chain from each piece of technical work up to a business objective, so that value can be traced rather than asserted.
category:
- Business
- Management
- Architecture
problems:
- difficulty-quantifying-benefits
- modernization-roi-justification-failure
- unclear-goals-and-priorities
- feature-factory
- short-term-focus
- product-direction-chaos
- invisible-nature-of-technical-debt
- competing-priorities
- wasted-development-effort
- delayed-value-delivery
- system-stagnation
- reduced-innovation
- competitive-disadvantage
- declining-business-metrics
- feature-bloat
- feedback-isolation
- high-maintenance-costs
- increased-cost-of-development
- market-pressure
- project-resource-constraints
- resource-waste
- slow-development-velocity
- stakeholder-confidence-loss
- stakeholder-frustration
- high-technical-debt
layout: solution
---

## Description

A value hierarchy is an explicit, maintained chain of reasoning that connects each piece of technical work to a business objective through the intermediate steps: this refactoring shortens the change cycle in this subsystem, which shortens time to market for this product line, which serves this year's stated goal of responding faster than a named competitor. It exists because the argument for technical work is almost always a chain of several links, and technical people habitually state only the first link while business people can only evaluate the last. The gap between them is where modernization proposals die — not because the connection is absent, but because nobody has written it down and each side assumes the other should see it. Making the chain explicit does two things: it lets value be traced and challenged link by link, and it exposes the work whose chain does not actually reach anything.

*The idea of arranging value into an explicit hierarchy is drawn from the Cloud Native patterns community, where it appears as a strategy pattern for transformation efforts.*

## How to Apply ◆

> In a legacy context the chain is usually three or four links long, and the middle links — the ones about change cost and risk — are exactly the ones nobody outside engineering can supply.

- **Start from the business objectives that already exist**, in the words the organization already uses. A hierarchy built on objectives invented by engineering is an engineering document, and it will be read as one.
- **Write each link as a claim that could be false.** "Reducing the build from 30 minutes to 5 increases the number of changes we can deliver per month" is checkable. "Improves developer experience" is not, and an unfalsifiable link breaks the chain wherever it appears.
- **Insist that every chain terminates.** Work whose chain runs out after two links — it makes the code nicer, and then nothing — should either be reframed until it reaches something, or acknowledged as work being done for its own sake. Both outcomes are more useful than a chain that trails off.
- **Keep the intermediate layer honest.** The middle links usually concern change cost, risk, and capacity, and these are where engineering has knowledge nobody else does. This layer is the team's actual contribution to the argument; a hierarchy that jumps straight from a refactoring to revenue will not be believed.
- **Attach measures where they exist**, at whichever link they exist. Not every link can be measured, but a chain with two measured links is far stronger than one with none, and the measured links anchor the unmeasured ones.
- **Use it in both directions.** Downward, it turns an objective into a set of candidate technical investments. Upward, it turns a proposed piece of work into a justification. The upward direction is what teams need; the downward direction is what makes leadership use the hierarchy at all.
- **Review it when objectives change.** A hierarchy built on last year's goals quietly justifies work that no longer serves anything, and this is a common way modernization programmes outlive their rationale.
- **Let it kill proposals.** A hierarchy that has never caused work to be dropped is decorative. The value comes from it being applied to work the team wants to do, not only to work it is defending.
- **Keep it small.** A diagram with 200 nodes will not be maintained or read. A handful of objectives, each with a few chains beneath them, is what stays usable.

## Tradeoffs ⇄

> An explicit chain makes technical value arguable rather than assertable, but it takes maintenance and can be used to refuse work whose value is real and hard to articulate.

**Benefits:**

- Technical work acquires a stated connection to business outcomes, which is what allows it to compete for funding rather than being classified as overhead.
- Weak links become visible and can be argued about specifically, which is far more productive than a general disagreement about whether technical work matters.
- Work whose chain does not reach anything gets identified, and some of it turns out to be genuinely optional.
- Engineering's distinctive knowledge — about change cost and risk — occupies a defined place in the argument rather than being the whole argument.
- The same structure serves prioritization, since chains that reach the highest-weighted objectives with the least effort are the obvious candidates.

**Costs and Risks:**

- Maintaining the hierarchy is ongoing work, and it goes stale quickly when objectives shift, at which point it justifies the wrong things.
- Chains can be constructed to reach any conclusion. A determined advocate can connect almost any work to almost any objective through enough plausible-sounding links.
- Work with real but hard-to-articulate value — reducing a risk nobody has yet experienced, keeping an option open — is systematically disadvantaged by a framework that demands an explicit chain.
- The exercise can become a compliance ritual attached to proposals after the decision, which produces documentation without changing anything.
- Long chains are weak chains: each link multiplies the uncertainty, and a four-link argument can be dismissed on the strength of doubt about any one of them.

## How It Could Be

A platform team's proposals were consistently declined while product feature requests were approved, and both sides had concluded the other did not understand the business. The team built a hierarchy starting from the three objectives in the company's published annual plan. Under "reduce time from customer request to delivered change," they placed the intermediate claim that the median change to the ordering subsystem took 19 days of which 11 were waiting on a shared test environment, and beneath that the specific work: ephemeral per-branch environments. Three links, two of them measured. The proposal was approved in one meeting after two years of declines. The decisive element was not the environments, which had been proposed before, but the middle link — the 11 days — which nobody outside the team had known and which no previous proposal had stated.

The hierarchy also ended a project the team wanted. They had been advocating a migration from one message broker to another for eighteen months. Building the chain honestly, they found it reached "the newer broker is better supported and we would prefer to work with it" and stopped there — no link to change cost, no link to risk that the existing broker was actually producing, and no objective it served. The team dropped it. Two developers were unhappy about this, and the same discipline applied a quarter later produced a chain for a database connection pooling fix that ran cleanly through incident hours to an availability commitment in a customer contract. That work had never been proposed, because nobody had thought of it as interesting.
