---
title: Benefits Realization Tracking
description: Go back after the work is done and report whether the promised benefit actually arrived — including when it did not.
category:
- Business
- Management
- Process
problems:
- difficulty-quantifying-benefits
- modernization-roi-justification-failure
- planning-credibility-issues
- stakeholder-confidence-loss
- short-term-focus
- feature-factory
- wasted-development-effort
- declining-business-metrics
- invisible-nature-of-technical-debt
- resource-waste
- delayed-value-delivery
- inability-to-innovate
- budget-overruns
- feature-bloat
- feedback-isolation
- increased-cost-of-development
- quality-degradation
- stakeholder-frustration
layout: solution
---

## Description

Benefits realization tracking is the practice of returning to a completed investment after enough time has passed and reporting, against the numbers used to justify it, what actually happened. Almost no organization does this. Business cases are scrutinized intensely before approval and never examined afterwards, which has a specific and damaging consequence: nobody learns which kinds of claim turn out to be true. The technical team's estimates are treated with the same generic scepticism regardless of their track record, because there is no track record. Tracking benefits is therefore less about accountability than about building the evidence base that makes the next proposal credible. It is also the only mechanism that reliably detects the investments that quietly did not work, which otherwise continue to be cited as successes indefinitely.

## How to Apply ◆

> The measures in a legacy business case usually take a year or more to move, which is exactly why nobody is still looking when they do.

- **Record the claim at approval time**, in the form it will later be checked: which measure, by how much, by when. A business case whose benefit is stated as "improved maintainability" cannot be verified, and that ambiguity is frequently deliberate on both sides.
- **Schedule the review when the investment is approved**, not when it completes. An unscheduled review does not happen, and a date set at approval is far harder to quietly drop than one proposed afterwards.
- **Allow a realistic interval.** Reviewing three months after a modernization completes measures disruption, not benefit. Twelve months is typical for legacy work, with an interim check at six.
- **Compare against the baseline that justified the spend**, not against a fresh measurement of the current state. If no baseline was recorded, say so plainly — that finding is itself the argument for recording one next time.
- **Report the misses as prominently as the hits.** A tracking practice that only surfaces successes is marketing, and everyone recognizes it as such within two cycles. The credibility that makes the practice worth doing comes entirely from its willingness to report failure.
- **Separate "the benefit did not materialize" from "the work was not done."** These have different lessons: the first says the theory was wrong, the second says execution was. Conflating them prevents either from being learned.
- **Look for benefits that were not predicted.** Legacy improvements routinely produce effects nobody claimed — a retired system that freed an unrelated licence, a refactoring that made an unplanned feature cheap. Capturing these improves the accuracy of future estimates, which usually understate rather than overstate.
- **Keep a running record across investments.** The pattern over ten reviews — which categories of claim hold up, which are consistently optimistic — is worth far more than any individual review and is what eventually changes how proposals are received.
- **Keep the review cheap.** Half a day against three or four measures. A heavyweight process will be skipped, and a skipped review provides no evidence at all.

## Tradeoffs ⇄

> Tracking benefits is what makes future proposals credible, but it creates a record of failures and requires someone to still care a year later.

**Benefits:**

- Proposals from a team with a verified track record are received differently, which compounds over time into materially easier funding for technical work.
- Estimation improves, because the organization learns which categories of claim are systematically optimistic and by roughly how much.
- Investments that quietly did not work are detected, rather than being cited as successes for years and used to justify repeating the approach.
- Unpredicted benefits get captured, and these are frequently substantial in legacy work where second-order effects are hard to foresee.
- The knowledge that a review will happen improves the honesty of business cases at the point they are written, which may be the largest effect.

**Costs and Risks:**

- It produces documented failures, which is politically uncomfortable and creates a strong incentive to let the practice lapse.
- Attribution a year later is genuinely hard: several things changed in the interval, and both credit and blame are contestable.
- The people who approved and delivered the work have often moved on, leaving nobody with the context or the motivation to conduct the review.
- Used punitively, it makes future business cases conservative and vague, which is the opposite of the intended effect.
- Twelve-month intervals sit awkwardly against annual planning cycles, so the finding often arrives after the decision it should have informed.

## How It Could Be

An organization had approved eleven technical investments over four years and reviewed none of them. Their engineering director instituted a rule: every approved investment above a threshold carried a scheduled twelve-month review, half a day, against the measures in the original case. The first four reviews were uncomfortable. Two investments had delivered roughly what was claimed. One had delivered about a third of it — a test automation effort whose promised reduction in manual testing effort had been eaten by a simultaneous expansion of scope that nobody had accounted for. One had delivered nothing measurable at all, because the measure named in the business case had never been instrumented and could not be reconstructed. The last finding changed practice more than the others: baseline instrumentation became a precondition of approval.

By the ninth review the pattern had become the useful output. Claims about incident reduction had held up well, averaging around 80 percent of what was projected. Claims about developer productivity had held up poorly, averaging under 30 percent, consistently because the projected time savings were absorbed by other work rather than converted into throughput. Claims about licence and infrastructure savings had been almost exactly right, being the easiest to estimate. The organization did not stop funding productivity improvements — it started discounting those claims by a stated factor and requiring the case to work at the discounted figure. Two proposals that would previously have been approved were declined on that basis, and one that had been declined twice was approved once its author reframed the benefit as incident reduction, which the record showed was the claim type that actually held.
