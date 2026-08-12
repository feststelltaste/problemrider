---
title: Domain Immersion
description: Send developers to observe the actual work the system supports, so that requirements are understood rather than transcribed.
category:
- Requirements
- Team
- Communication
problems:
- complex-domain-model
- inadequate-requirements-gathering
- requirements-ambiguity
- knowledge-gaps
- feedback-isolation
- suboptimal-solutions
- incomplete-knowledge
- reduced-feature-quality
- eager-to-please-stakeholders
- stakeholder-dissatisfaction
- legacy-business-logic-extraction-difficulty
- negative-user-feedback
- declining-business-metrics
- feature-factory
- frequent-changes-to-requirements
- product-direction-chaos
- stakeholder-frustration
layout: solution
---

## Description

Domain immersion means developers spending time where the work the system supports actually happens — sitting with the claims handlers, watching a shift in the warehouse, observing a month-end close — rather than receiving that work as a description. The gap it addresses is specific and reliable: people cannot accurately describe work they do fluently. A practitioner asked to explain their process gives the official version, omits the exceptions that make up a third of their day, and never mentions the workarounds they have stopped noticing. Requirements written from such descriptions are correct and incomplete in the same characteristic ways every time. In legacy contexts immersion has a second use, because the business rules encoded in a decades-old system frequently exist nowhere else, and the people who still apply them by hand where the system falls short are the closest thing to documentation available.

## How to Apply ◆

> The spreadsheet on a user's second monitor is a specification of what the system does not do, and it is invisible from any distance.

- **Observe rather than interview.** Sit with someone doing their actual work, for a substantial block of time — half a day at minimum. The point is to see what they do, not to hear what they say they do, and the difference between the two is the whole value.
- **Watch for the workarounds**: the spreadsheet, the sticky note, the second system opened alongside, the step where they consistently pause. Each is a requirement the current system fails to meet, and none of them will appear in any request.
- **Ask about the exceptions**, because the routine path is the one that gets described. "What happens when it is not straightforward?" reliably opens the part of the domain the specification never covered, and in most domains the exceptions are where the complexity lives.
- **Send the people who will build it**, not only analysts. Understanding transferred through an intermediary loses precisely the details whose significance is not obvious in advance, which are the details that matter.
- **Go at a meaningful time.** A quiet Tuesday shows the routine path; month-end, a peak period, or an incident shows the system under the conditions where its inadequacies actually cost something.
- **Write down what you observed, promptly**, including the things you did not understand. The confusions are as valuable as the observations, and both fade within a day.
- **Learn the vocabulary and use it exactly.** Domain terms carry precise distinctions, and a developer who conflates two of them will build a model that cannot represent the difference — which is how domain models come to fight their domain.
- **Repeat it periodically** rather than once at the start of a project. The work changes, and a team's understanding of the domain drifts toward whatever the system currently does.
- **Feed observations back explicitly** to the people you observed. It builds the relationship, corrects your misunderstandings early, and frequently prompts them to mention something they had not thought worth saying.
- **Include support and operations staff.** They see the system's failures across many users, which is a view neither the developers nor any individual user has.

## Tradeoffs ⇄

> Immersion produces understanding that no amount of written requirement conveys, at the cost of developer and practitioner time and a dependence on people who are busy.

**Benefits:**

- Requirements are understood rather than transcribed, which prevents the correct-but-useless features that specifications reliably produce.
- Workarounds become visible, and each one is both a requirement and a quantifiable ongoing cost the organization is already paying.
- The domain model improves, because the developers building it have seen the distinctions the vocabulary encodes rather than inferring them from names.
- Undocumented business rules surface, and in legacy contexts the people applying them by hand are frequently the last remaining source.
- The relationship between developers and users improves substantially, which changes the quality of every subsequent conversation.

**Costs and Risks:**

- It consumes the time of both developers and the observed practitioners, and the latter are usually busy people whose availability must be negotiated.
- Observation is disruptive and can feel like surveillance, particularly if it is arranged by management rather than agreed with the person.
- What is observed is one person's way of working, which may be idiosyncratic. Observing a single practitioner and generalizing produces a confidently wrong model.
- Physical presence is difficult for distributed teams and impossible for some domains, and remote observation loses much of the value.
- Developers can over-identify with the users they observed and advocate for their particular needs against the broader user population.

## How It Could Be

A team building a replacement for a freight booking system worked from a 40-page specification written by a business analyst. Two developers spent a day in the booking office before starting. The specification described a linear process: enter shipment details, select a carrier, confirm. What they observed was that bookers had three browser tabs open, kept a shared spreadsheet of carrier quirks that the system did not model — which carriers refused certain postcodes, which required 48 hours' notice, which quoted differently by phone — and made roughly one call in four to negotiate rates the system had no field for. None of this was in the specification, because the analyst had asked how the process worked and been told how it was supposed to work. The replacement was redesigned around the actual process, and the shared spreadsheet became a modelled entity rather than a workaround.

A second team used immersion to recover business logic they could not extract from code. A pricing module contained a 600-line conditional that nobody could interpret, and the original developer had left in 2014. Rather than continuing to read it, two developers spent two days with the commercial team, watching them quote and asking why each price was what it was. The rules that emerged — a volume threshold that changed by customer segment, a legacy discount honored for pre-2011 contracts, a regional adjustment applied only to two countries — mapped onto roughly 400 of the 600 lines. The remaining 200 turned out to implement a promotional scheme that had ended in 2016 and which nothing had reached since, a conclusion the team confirmed with logging before deleting it.
