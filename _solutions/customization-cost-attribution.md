---
title: Customization Cost Attribution
description: Track what each customer-specific variant costs to maintain and attribute it to whoever asked for it, so that agreeing to one becomes a priced decision.
category:
- Business
- Management
- Process
problems:
- excessive-customization
- eager-to-please-stakeholders
- high-maintenance-costs
- increased-cost-of-development
- market-pressure
- short-term-focus
- feature-creep
- difficulty-quantifying-benefits
- maintenance-cost-increase
- competing-priorities
- product-direction-chaos
- invisible-nature-of-technical-debt
- core-modification-of-standard-software
- custom-report-sprawl
- reimplemented-standard-functionality
layout: solution
---

## Description

Customization cost attribution records what each variant costs to keep alive — the engineering time spent maintaining it, the testing it adds, the upgrade effort it imposes — and attributes that cost to the customer, deal, or department that requested it. It addresses the structural reason customization accumulates: the person who agrees to a variant does not pay for it. A salesperson closing a deal, or an executive accommodating a large customer, incurs a cost that lands years later on an engineering budget nobody connects to that decision. Under that arrangement, agreeing is always locally rational and the accumulation is guaranteed. Attribution does not forbid customization; it makes the decision priced, so that the organization can decide whether a variant is worth its cost rather than discovering the answer a decade later.

## How to Apply ◆

> Nobody in a heavily customized product can say what any individual customization costs, which is precisely why there are so many of them.

- **Identify variants as discrete, named things** with an owner and a requesting party. Customization that exists only as scattered conditionals cannot be costed, so the inventory is the prerequisite and is usually revealing on its own.
- **Attribute the direct effort**: time spent on defects specific to a variant, time spent adapting it during releases, and support handling attributable to it. Even coarse tracking against a small number of variants produces a usable picture within a quarter.
- **Include the upgrade cost**, which is usually the largest component and the most invisible. What it takes to bring an installation carrying this variant onto a new version, measured from what it actually took last time.
- **Include the tax on everything else.** A variant that must be considered whenever a shared module changes imposes a cost on work that has nothing to do with it. This is diffuse and real, and a rough allocation is better than treating it as zero.
- **Report per variant, per year**, in the same terms the business uses for the revenue the variant was meant to secure. The comparison between the two is the entire point and is frequently uncomfortable.
- **Bring the figure into the decision, before it is made.** A request evaluated with an estimated annual maintenance cost attached is a different conversation from one evaluated on implementation effort alone. Implementation is the down payment; maintenance is the loan.
- **Consider charging it.** Where the commercial model allows, an explicit ongoing fee for a variant converts the discipline from an internal argument into a market test — and customers frequently decline once the variant has a price.
- **Review the portfolio annually** and identify variants whose cost exceeds any plausible value. These are candidates for retirement, and the retirement conversation is far easier with a number attached.
- **Record the decisions that are made anyway.** Some variants will be approved despite an unfavourable cost, for strategic reasons. Recording that as a deliberate choice is a legitimate outcome and preserves the credibility of the practice.

## Tradeoffs ⇄

> Attribution converts an invisible accumulating cost into a priced decision, but the measurement is imprecise and the findings create conflict with the people who requested the variants.

**Benefits:**

- The decision to customize becomes priced rather than free, which changes behavior at the point where the accumulation actually starts.
- Variants whose cost exceeds their value become identifiable, and retiring them is the only intervention that reduces the burden absolutely.
- Sales and product acquire the information they need to trade a customization against something else, which they currently do not have.
- The engineering cost of commercial decisions becomes visible to the people making them, closing a feedback loop that is normally absent entirely.
- Charging for variants, where possible, provides a genuine market test of whether a customization is wanted enough to pay for.

**Costs and Risks:**

- Attribution is imprecise. Shared work is hard to allocate, and every figure can be argued with by anyone who dislikes the conclusion.
- Tracking effort per variant is administrative overhead on engineers, and it decays quickly if the resulting numbers are never used.
- The findings create conflict with sales and account management, whose incentives the practice directly opposes.
- Costing a variant requested by a strategically important customer can produce a number that will be overridden regardless, which risks the practice being seen as futile.
- Focusing on cost alone ignores that some variants have strategic value beyond their revenue, and a purely cost-driven portfolio review will recommend retiring things it should not.

## How It Could Be

A vendor with 34 installations tracked variant-attributable effort for two quarters against a list of 47 named customizations. The distribution was extreme: 6 variants accounted for roughly 60 percent of the attributable effort, and 19 had consumed no measurable effort at all. One variant — a bespoke settlement export for a customer acquired eight years earlier — was costing an estimated 34 developer-days a year, against an annual contract value that the account manager confirmed was among their smallest. Presented as a single comparison, the variant was retired in one conversation with the customer, who accepted the standard export after a two-hour walkthrough. Nobody had asked in eight years, because nobody had known the cost.

The forward-looking effect mattered more than the retirements. The vendor added an estimated annual maintenance figure to every customization request, alongside the implementation estimate. In the following year, 14 requests were evaluated this way. Five were approved as before. Four were met by extending the standard product instead, once the comparison showed a variant was more expensive than generalizing. Three were declined, and the customers accepted the decision when shown the reasoning. Two were approved with the customer paying an explicit annual fee. The head of sales, who had opposed the practice initially, later described the maintenance figure as the strongest negotiating instrument he had, because it converted a request into a trade rather than an expectation.
