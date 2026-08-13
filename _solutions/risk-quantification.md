---
title: Risk Quantification
description: Express legacy risk as expected loss in money — likelihood times impact — so that avoided harm can compete with revenue in a funding decision.
category:
- Business
- Management
- Security
problems:
- modernization-roi-justification-failure
- difficulty-quantifying-benefits
- obsolete-technologies
- technology-lock-in
- legacy-skill-shortage
- single-points-of-failure
- knowledge-silos
- regulatory-compliance-drift
- increasing-brittleness
- system-stagnation
- system-outages
- vendor-dependency
- high-defect-rate-in-production
- competitive-disadvantage
- deployment-risk
- high-maintenance-costs
- invisible-nature-of-technical-debt
- missing-rollback-strategy
- project-resource-constraints
- technology-stack-fragmentation
- vendor-dependency-entrapment
- high-technical-debt
- implementation-partner-dependency
- retention-obligations-block-change
- upgrade-blocked-by-customization
- voided-vendor-support
layout: solution
---

## Description

Risk quantification expresses a risk as an expected annual loss — how likely it is to occur, multiplied by what it would cost if it did — so that avoiding it can be compared against other uses of money. It addresses the specific failure that sinks most legacy modernization cases: the value of the work is mostly avoided harm, and avoided harm has no natural unit, so it is described qualitatively while the alternative proposal arrives with a revenue figure. Qualitative always loses to quantitative in a funding decision, regardless of which is more important. The technique does not claim precision. A range built from stated assumptions is not an accurate prediction of what will happen, and it is not trying to be: it is a translation, converting "this is dangerous" into a form the decision process can actually weigh.

## How to Apply ◆

> Legacy risk is unusually amenable to this because the failure modes are known — the organization has often already experienced smaller versions of them.

- **Name the specific failure scenario**, not the general condition. "The mainframe is old" cannot be quantified. "The one remaining COBOL developer leaves and a production defect in the settlement batch takes six weeks instead of two days to fix" can be, because both terms are estimable.
- **Estimate likelihood and impact as ranges**, with the reasoning written down. Precision is not the goal and pretending to it invites attack; "somewhere between 10 and 25 percent in any given year" with a stated basis is more defensible than a single confident number.
- **Build impact from components** the organization already prices: hours of outage times revenue per hour, recovery effort, contractual penalties, regulatory fines, notification and remediation costs, and where relevant the cost of customers who leave. Each component can be checked independently.
- **Use the organization's own history.** Past incidents are the best available evidence for both terms, and most legacy risks have already produced near-misses or smaller instances that were handled and forgotten. Reconstructing those is usually the most persuasive part of the analysis.
- **Model the risk that grows.** Unlike most risks, legacy risk generally rises over time — an eroding skill pool, an approaching end-of-support date, an accumulating data volume. Showing the expected loss as a curve rather than a number is what makes the timing argument.
- **Present the residual risk after the proposed work**, not just the current risk. The benefit is the difference between the two, and a proposal that implies the risk goes to zero will not be believed by anyone experienced.
- **Involve the functions that already do this.** Finance, insurance, and risk management have established methods and, more importantly, established credibility. A number produced jointly with them is treated very differently from one produced by engineering alone.
- **Keep the assumptions visible and separate from the conclusion**, so that a sceptic can argue with an input rather than dismiss the output. The most productive outcome of such an analysis is frequently a disagreement about one specific probability, which is a conversation that can converge.
- **Do not quantify everything.** Some risks are genuinely unquantifiable, and forcing a number onto them produces figures that discredit the ones that were done properly.

## Tradeoffs ⇄

> Quantification lets risk reduction compete for funding on equal terms, at the price of false precision, contestable assumptions, and a real chance of being wrong in public.

**Benefits:**

- Avoided harm acquires a unit and can be compared against revenue-generating alternatives, which is the only way risk-reduction work wins a prioritization decision.
- The rising-risk curve makes the timing argument that a static description cannot, converting "we should do this" into "the expected loss exceeds the cost of fixing it from next year onward."
- Disagreements become specific and resolvable — about one probability or one impact component — instead of being a general clash of intuitions.
- Reconstructing past near-misses frequently uncovers that the organization has already paid substantial amounts for a risk it believed was hypothetical.
- The analysis is reusable. Once the model exists, re-running it annually shows whether the risk profile is improving, which is itself a management instrument.

**Costs and Risks:**

- Probability estimates for rare events are genuinely unreliable, and the arithmetic gives them an appearance of rigour they do not have.
- A single quantified number invites the response that the risk is affordable, which is a legitimate decision but may not be the one the analysis was intended to support.
- The exercise takes real effort and specialist input, and it can consume more time than the decision warrants for smaller risks.
- Quantified risks compete with each other, and one that is genuinely severe but hard to estimate will lose to one that is moderate and easy to estimate.
- If the quantified risk never materializes, the analysis can be retrospectively characterized as scaremongering, which makes the next one harder.

## How It Could Be

An organization's payment reconciliation ran on a platform with one remaining developer who understood it, aged 61. Three modernization proposals had been declined as insufficiently justified. The fourth quantified the specific scenario rather than describing the general concern. Likelihood of losing that knowledge within three years — retirement, illness, or resignation — was estimated at 60 to 80 percent, using the organization's own actuarial assumptions for a person of that age and tenure. Impact was built from four components: an estimated 4 to 9 months to rebuild the knowledge externally, contractor rates for that period, the reconciliation backlog that would accumulate meanwhile at a rate they could measure from a two-week absence the previous year, and a regulatory reporting obligation that would be missed with a stated penalty. The expected loss came to €1.4 to €3.1 million against a modernization cost of €900,000. The work was funded within a month.

The residual-risk framing was what made the case survive scrutiny. The proposal did not claim the risk would be eliminated — the replacement would still concentrate knowledge, just less severely, in a technology with a larger available skill pool. It stated a residual expected loss of €300,000 to €600,000 and named the further measures that would reduce it. A CFO who had rejected the previous three proposals commented that this was the first one that had not sounded like advocacy. The two-week absence the previous year, reconstructed and costed as part of the analysis, turned out to have cost the organization roughly €70,000 in contractor time and delayed reporting — an incident that had been handled, invoiced, and never connected to the underlying risk by anyone.
