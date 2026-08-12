---
title: Technology Radar
description: Maintain an explicit, periodically reviewed list of which technologies the organization adopts, tolerates, or retires — and hold decisions to it.
category:
- Architecture
- Management
- Dependencies
problems:
- obsolete-technologies
- technology-lock-in
- premature-technology-introduction
- cv-driven-development
- cargo-culting
- dependency-version-conflicts
- technology-isolation
- technology-stack-fragmentation
- rapid-prototyping-becoming-production
- shared-dependencies
- vendor-dependency
- second-system-effect
- dependency-on-supplier
- inappropriate-skillset
- vendor-relationship-strain
layout: solution
---

## Description

A technology radar is a maintained, published classification of the technologies an organization uses: which are the default choice, which are permitted in specific circumstances, which are being trialed, and which are on their way out. It is reviewed on a fixed cadence by a group with the authority to move items between categories. Its value in a legacy landscape is twofold. It constrains proliferation going forward, so that the next decade does not add another five frameworks chosen by whoever happened to start the project. And it makes retirement an explicit, scheduled decision rather than something that happens when a technology stops working — which in practice means never, since a component that still runs is never urgent until it suddenly is.

## How to Apply ◆

> The characteristic legacy technology landscape is not one bad choice but forty reasonable choices made independently over twenty years, each still requiring someone who knows it.

- **Start by inventorying what is actually in use**, including the components nobody has thought about in years. The inventory alone typically finds several technologies that no current employee is qualified to maintain, and that discovery is often more valuable than the radar itself.
- Use **a small number of clear categories** with different meanings. Four works well: default for new work, acceptable in stated circumstances, under evaluation with a decision date, and scheduled for retirement. More categories produce debate about classification rather than about technology.
- **Place every item explicitly**, including the ones everybody is tired of arguing about. An unclassified technology is one that will be introduced again, by someone who did not know there had been a discussion.
- Record **why** each item sits where it does, briefly. The reasoning is what makes the radar useful when someone wants to challenge a placement, and challenging placements should be possible — an unchallengeable radar becomes an obstacle to route around.
- Give **retirement entries a date and an owner**, or they will not move. A technology marked as being phased out for four consecutive reviews is a technology that is not being phased out, and the radar should make that visible rather than conceal it.
- **Review on a fixed cadence**, twice a year for most organizations. Reviewing more often produces churn; less often lets the radar drift from what teams are actually doing, at which point it is ignored.
- Include **the people who will be bound by it** in the review. A radar produced by an architecture group and issued to teams is complied with in documents and ignored in code. One where each team has a voice in the placement is enforced by the teams themselves.
- Define what happens when someone **wants to use something not on the radar**: a stated process, a time-boxed evaluation, and a decision. Without an entry path, the radar becomes an obstacle and the response is to introduce the technology quietly.
- **Connect it to the dependency and end-of-support data.** A technology whose vendor support ends in fourteen months should be moving toward retirement on the radar automatically, not through someone noticing.

## Tradeoffs ⇄

> A radar reduces proliferation and makes retirement a decision rather than an accident, in exchange for governance overhead and some loss of team autonomy.

**Benefits:**

- Technology proliferation slows, which directly reduces the number of skills the organization must sustain and the number of components that can become unmaintainable.
- Retirement becomes a scheduled activity with owners and dates, rather than something forced by an end-of-support announcement or a security advisory.
- New technology introduction is subject to a stated process rather than to who started the project, which is the mechanism behind most résumé-driven choices.
- Recurring arguments about the same technology choices stop consuming design discussions, because the decision and its reasoning are recorded.
- Hiring and training can target a bounded set of technologies, which matters enormously in an organization maintaining systems in six languages.

**Costs and Risks:**

- It reduces team autonomy over technical decisions, which is demotivating and can drive away the developers who most value that autonomy.
- A radar maintained by a group too distant from the code becomes an ignored document, and the effort spent producing it is wasted twice — once in producing it, once in the parallel unofficial reality.
- Overly restrictive classification pushes experimentation underground, where it happens without review rather than not at all.
- Reviews consume time from senior technical people twice a year and generate debate that can become political.
- A radar can entrench an outdated default. The category that says "this is what we use" is the hardest one to change, and it can keep an organization on a technology past the point where the choice made sense.

## How It Could Be

A financial services organization inventoried its production technology as the first step toward a radar and found 34 distinct runtime platforms, including four programming languages for which no current employee claimed proficiency and two message brokers that had been out of vendor support for over three years. The radar that resulted was deliberately unambitious: two default languages, a named set of acceptable exceptions, and eleven items with retirement dates and owners. The first review six months later showed three of the eleven retired, two extended with stated reasons, and six untouched — which prompted a direct conversation about capacity that had not previously been possible, because "we are phasing those out" had been an acceptable answer for years precisely because nobody was counting.

The entry process turned out to matter more than the classifications. A team wanted to introduce a document database for a new reporting feature. Under the previous informal regime this would either have happened silently or been refused by an architect in a meeting. Instead it entered the radar as under evaluation with a three-month decision date and a stated question: does it reduce the reporting query load enough to justify a new operational skill. The evaluation found that it did not — a materialized view in the existing database achieved most of the benefit — and the team reached that conclusion themselves. The radar's contribution was not the refusal but the requirement that the question be asked and answered explicitly.
