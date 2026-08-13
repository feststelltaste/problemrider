---
title: Process-Software Misfit
description: The software was bent to fit a process that grew historically, instead of the process being examined against what the product assumes.
category:
- Business
- Process
- Requirements
related_problems:
solutions:
- fit-to-standard-principle
- domain-immersion
- business-process-modeling
- functional-gap-analysis
- regular-stakeholder-demonstrations
- outcome-based-goal-setting
- executive-sponsorship
- value-stream-mapping
- pilot-projects
- definition-of-ready
layout: problem
---

## Description

Process-software misfit occurs when a commercial software product is adapted to reproduce an organization's existing way of working, without anyone asking whether that way of working is worth preserving. Commercial software encodes a process model, and much of its value comes from that model being coherent and having been refined across many customers. An organization that overrides the model to match its own historically grown practice pays for the product, discards the reasoning inside it, and takes on the cost of maintaining the difference forever. The misfit is rarely a deliberate decision. It follows from a requirements process that records how things are done today and treats that as the specification, and from the fact that no one is accountable for changing how the business works.

## Indicators ⟡

- Requirements were gathered by documenting the current process and were not challenged
- The customization list is dominated by items that reproduce existing steps rather than enabling new outcomes
- Users describe the new system as working "like the old one, but slower"
- Nobody can explain why a step exists beyond the fact that it has always been there
- Standard product training is not used because it does not describe how your installation works
- Process changes are considered out of scope for the software project, by explicit agreement

## Symptoms ▲

- [Excessive Customization](excessive-customization.md)
<br/>  Reproducing an existing process in a product built around a different one requires adaptation at every point where they differ.
- [Inefficient Processes](inefficient-processes.md)
<br/>  Historical inefficiencies are preserved and encoded, and afterwards they are harder to remove than they were before automation.
- [Reimplemented Standard Functionality](reimplemented-standard-functionality.md)
<br/>  Where the product's version of a step differs from the local one, the local version is built instead of the standard being adopted.
- [Increased Cost of Development](increased-cost-of-development.md)
<br/>  Effort is spent recreating what the organization already had, rather than on capability it did not.
- [Difficulty Quantifying Benefits](difficulty-quantifying-benefits.md)
<br/>  A project that reproduces the previous process delivers little measurable improvement, which makes its value hard to demonstrate afterwards.
- [User Frustration](user-frustration.md)
<br/>  Users experience the disruption of a new system without the benefit of a better process, which is the least favourable combination.
- [Upgrade Blocked by Customization](upgrade-blocked-by-customization.md)
<br/>  The accumulated difference between the local process and the product's model has to be carried through every subsequent release.

## Causes ▼

- [Inadequate Requirements Gathering](inadequate-requirements-gathering.md)
<br/>  Documenting the current state is treated as requirements analysis, so the specification is a description of the past.
- [Project Authority Vacuum](project-authority-vacuum.md)
<br/>  Changing how a department works requires authority the project does not have, so the software is changed instead.
- [Eager to Please Stakeholders](eager-to-please-stakeholders.md)
<br/>  Departments asked what they need describe what they do now, and the answers are accepted rather than examined.
- [Legacy Skill Shortage](legacy-skill-shortage.md)
<br/>  Nobody involved knows the product's process model well enough to argue that the standard would serve, so the local process wins by default.
- [Dependency on Supplier](dependency-on-supplier.md)
<br/>  An implementation partner paid for development has no incentive to argue that the customer should change its process instead.
- [Market Pressure](market-pressure.md)
<br/>  Process change takes longer than software change, and where a date is fixed, the faster option is chosen regardless of which is better.

## Detection Methods ○

- Review the requirements from the last implementation and count how many describe current practice versus a desired outcome
- For each significant customization, ask what the standard would have done and why it was rejected; missing answers indicate the question was never asked
- Compare process metrics before and after implementation; a project that reproduced the process will show little movement
- Ask users whether the system works the way the vendor's training describes, and where it diverges
- Look for steps in the process that exist only because a previous system required them
- Check whether the project had a mandate to change business process, and whether it was used

## Examples

A logistics company implemented a warehouse management product and customized 40 areas to match how their sites already worked. Two years later, an external review compared their picking process against the product's standard model and found that six of the customizations preserved steps introduced in the 1990s to compensate for a paper-based system that had been retired in 2004. One step required a supervisor countersignature on a movement that the product would have prevented from being incorrect in the first place. Removing the six customizations and adopting the standard process reduced picking time per order by roughly eleven percent — an improvement that had been available on day one of the implementation and had been customized away.

The authority problem was the underlying cause and it was visible in the project's own records. The implementation charter stated explicitly that the project would not require changes to warehouse operating procedures, on the grounds that operations could not absorb disruption during a peak season. That constraint was set for one season and was never revisited across a two-year implementation. Nobody involved had the standing to reopen it, and by the time the review took place, the customizations had been in production long enough that removing them was itself a change requiring the authority that had been missing at the start.
