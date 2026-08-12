---
title: Total Cost of Ownership Transparency
description: Measure and publish what a legacy system actually costs to keep running — maintenance, incidents, licences, and lost capacity — so that investment decisions rest on numbers instead of impressions.
category:
- Business
- Management
- Process
problems:
- budget-overruns
- increased-cost-of-development
- maintenance-cost-increase
- high-maintenance-costs
- invisible-nature-of-technical-debt
- short-term-focus
- system-stagnation
- planning-credibility-issues
- delayed-value-delivery
- inability-to-innovate
- obsolete-technologies
- high-technical-debt
- project-resource-constraints
- stakeholder-confidence-loss
layout: solution
---

## Description

Total cost of ownership transparency is the practice of measuring what a system actually costs to keep operating and publishing that figure alongside the cost of changing it. The components are individually mundane and collectively invisible: developer time spent on maintenance rather than new capability, incident response and on-call load, licences and infrastructure, the effort consumed by manual workarounds, and the opportunity cost of capacity that goes to keeping the system alive. Organizations routinely fund the modernization of systems whose running cost they cannot state, and routinely decline to fund it for the same reason. The argument for investing in a legacy system almost always fails not because the case is weak but because it is made qualitatively — "the code is bad, it's getting harder" — against a proposal that arrives with a number attached.

## How to Apply ◆

> The costs that make a legacy system expensive are distributed across budgets that nobody adds together: development salaries, incident hours, licence renewals, and the manual work that business departments absorbed years ago and no longer mention.

- **Split development effort by category** and track it: new capability, maintenance and defect work, unplanned incident response, and mandatory work such as compliance and dependency upgrades. Two or three categories tracked consistently are worth more than a detailed taxonomy tracked for a month. The ratio is usually the headline finding.
- **Quantify the incident load** in hours, not incident counts: time to detect, time to resolve, people involved, and the out-of-hours share. Counting incidents understates the cost, because the expensive ones are the long ones.
- Include the **direct running costs** that are usually held in a different budget: licences, support contracts, infrastructure, and the specialist contractors retained because nobody internally can maintain a component.
- **Find the manual workarounds outside the technology budget.** Legacy systems push cost into the organizations that use them — the reconciliation someone does every month, the spreadsheet that exists because a report does not. This cost is frequently larger than the entire IT cost of the system, and it is invisible until someone asks the departments.
- Express the **opportunity cost explicitly**: if seventy percent of capacity goes to keeping the system running, then thirty percent is what the organization has available for everything it wants. Stating it this way converts a technical complaint into a business constraint.
- **Track the trend, not just the level.** A maintenance share rising from forty to sixty percent over three years is a far more compelling argument than sixty percent in isolation, because it forecasts where the line reaches one hundred.
- **Attribute costs to parts of the system** where the data allows, using change frequency and incident data. "This subsystem is eight percent of the code and forty percent of the incident hours" directs investment in a way that a system-wide figure cannot.
- **Publish on a fixed cadence** to the people who make funding decisions, in their terms — cost, risk, capacity — rather than in technical terms. A number that appears once in a business case is an argument; a number that appears quarterly is a management instrument.
- **Report the effect of improvements against the same measures.** Investment that cannot demonstrate a movement in the numbers that justified it will not be granted a second time.

## Tradeoffs ⇄

> Making costs visible is usually the decisive step in getting modernization funded, but the measurement itself costs effort and the resulting numbers can be used in ways the team did not intend.

**Benefits:**

- Investment decisions rest on comparable figures rather than on competing assertions, which is the condition under which maintenance work can win against feature work.
- The trend forecasts the future state, which is the argument that moves decisions from someday to now.
- Cost attribution directs improvement effort to the parts of the system that actually consume the budget rather than the parts that are most unpleasant to work in.
- The hidden organizational cost of manual workarounds becomes visible, and it is frequently the single largest number in the analysis.
- Improvements become defensible after the fact, which makes the next investment easier to obtain.

**Costs and Risks:**

- Consistent effort tracking is disliked by developers and degrades quickly if the categories are fine-grained or the data is used to evaluate individuals.
- The numbers can be turned against the team — a high maintenance share read as inefficiency rather than as a property of the system — particularly where the reader is looking for a reason to outsource.
- Opportunity cost and workaround cost are estimates, and estimates are attacked when the conclusion is unwelcome. Conservative figures with stated assumptions survive scrutiny better than aggressive ones.
- Collecting the organizational cost requires cooperation from departments outside technology, who may have no incentive to make their workarounds visible.
- The measurement can become an end in itself, consuming effort in reporting that would be better spent reducing the costs being reported.

## How It Could Be

An engineering manager had failed twice to obtain funding to modernize a twenty-year-old order processing system. On the third attempt she spent six weeks measuring instead of arguing. Development time tracked over two quarters showed 71 percent going to maintenance, defects, and incidents, up from 52 percent three years earlier according to reconstructed ticket data. Incident hours totalled 940 across the two quarters, 310 of them outside working hours. Licences and a retained specialist contractor added a figure that had never appeared in the same document as the salaries. And a survey of two business departments found roughly 1.5 full-time equivalents spent on manual reconciliation that existed solely because two subsystems did not exchange data correctly. The trend line, extrapolated, showed maintenance consuming all available capacity within four years. The funding was approved in one meeting.

The attribution analysis then changed what the funding was spent on. The team had assumed the effort should go to the oldest subsystem, which was the least pleasant to work in. Crossing incident hours with change frequency showed something different: the order validation module, comparatively modern and unremarkable to work in, accounted for 44 percent of incident hours because of how it interacted with a partner interface. Redirecting the first phase there reduced incident hours by roughly a third within two quarters — a result that, reported against the same measures that had justified the investment, secured the second phase without a business case.
