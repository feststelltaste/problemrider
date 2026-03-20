---
title: On-Site Customer
description: Directly involve customers in development
category:
- Requirements
- Communication
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/on-site-customer
problems:
- stakeholder-developer-communication-gap
- requirements-ambiguity
- inadequate-requirements-gathering
- misaligned-deliverables
- customer-dissatisfaction
- feedback-isolation
- no-continuous-feedback-loop
- implementation-rework
layout: solution
---

## How to Apply ◆

> In legacy modernization projects, having a customer representative embedded with the development team prevents the common failure mode of building technically excellent replacements that miss actual user needs.

- Identify a customer or user representative who has deep knowledge of the legacy system's daily workflows and secure their commitment for at least several hours per week of direct availability to the team.
- Seat the customer representative physically or virtually with the development team so that questions about legacy behavior can be answered in minutes rather than days of email exchanges.
- Have the on-site customer participate in sprint planning and story refinement to clarify requirements that stem from undocumented legacy behavior — they often know why a process works a certain way when no documentation exists.
- Use the on-site customer to validate completed features against real-world usage patterns, catching misunderstandings before they accumulate into major rework.
- Encourage the customer to demonstrate actual legacy system workflows to the team, including workarounds and unofficial processes that will not appear in any specification document.
- Rotate on-site customer representatives periodically to capture different perspectives and avoid single-person bias in requirements interpretation.

## Tradeoffs ⇄

> Having a customer embedded with the team dramatically reduces requirements ambiguity but requires organizational commitment and careful management of the customer's time.

**Benefits:**

- Eliminates the delay between encountering a requirements question and getting an authoritative answer, which is especially valuable when modernizing systems with undocumented business rules.
- Reduces implementation rework by catching misunderstandings early through continuous validation rather than late-stage acceptance testing.
- Builds shared understanding between technical and business stakeholders, reducing the communication gap that frequently derails modernization projects.
- Captures tacit knowledge about legacy system usage that would otherwise be lost when the old system is decommissioned.

**Costs and Risks:**

- Finding a customer representative who has both deep domain knowledge and availability to dedicate significant time to the development team can be difficult.
- A single on-site customer may represent only one perspective, leading to solutions that work for their workflow but not for other user groups.
- The customer representative may become a bottleneck if the team relies on them for every decision rather than building their own domain understanding.
- Organizational politics may prevent the right person from being assigned to the role, resulting in a representative who lacks authority or knowledge.

## Examples

> The following scenarios illustrate the impact of on-site customer involvement in legacy modernization.

A municipal government was replacing a 25-year-old permitting system. Initial attempts based on written requirements documents resulted in a system that technically met specifications but was rejected by permit clerks because it forced them through a rigid linear workflow instead of the flexible, multi-application juggling they actually performed daily. After embedding a senior permit clerk with the development team three days per week, the team discovered dozens of undocumented shortcuts and workarounds that were essential to meeting daily processing targets. The clerk's continuous involvement reduced rework by an estimated 40% and resulted in a system that permit staff actually preferred to the legacy application.
