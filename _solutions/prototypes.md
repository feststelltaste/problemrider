---
title: Prototypes
description: Validate suitability and usability early through business prototypes
category:
- Requirements
- Process
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/prototypes
problems:
- implementation-rework
- misaligned-deliverables
- requirements-ambiguity
- poor-user-experience-ux-design
- customer-dissatisfaction
- fear-of-change
- modernization-strategy-paralysis
- assumption-based-development
layout: solution
---

## How to Apply ◆

> Business prototypes in legacy modernization let stakeholders see and interact with proposed replacements before committing to full implementation, reducing the risk of building the wrong thing.

- Build low-fidelity prototypes (wireframes, clickable mockups) of the replacement system's key workflows and present them to actual users of the legacy system for feedback.
- Focus prototypes on the workflows where the legacy system is most painful or where the replacement design departs most from existing behavior, since these are the highest-risk areas for user rejection.
- Use prototypes to validate that critical legacy system behaviors are preserved — users often have strong expectations shaped by years of using the old system.
- Iterate on prototypes rapidly based on user feedback, treating each version as a learning tool rather than a commitment to a specific design.
- Create high-fidelity prototypes for the most critical or contentious features to give stakeholders confidence before committing development resources.
- Present prototypes to different user groups separately to capture divergent needs and workflows.

## Tradeoffs ⇄

> Prototypes accelerate alignment between stakeholders and developers but must be clearly positioned as disposable artifacts to avoid scope confusion.

**Benefits:**

- Reduces costly rework by identifying usability issues and requirements gaps before full development begins.
- Helps stakeholders who struggle to articulate requirements in the abstract provide concrete feedback when they can see and interact with a proposed solution.
- Builds stakeholder confidence in the modernization effort by making progress visible early.
- Surfaces hidden requirements and unstated assumptions about legacy system behavior that written specifications miss.

**Costs and Risks:**

- Stakeholders may mistake a prototype for a nearly finished product and underestimate the remaining development effort.
- Prototype code that is rushed into production creates technical debt from the start of the modernization effort.
- Building prototypes requires design skills and tooling that legacy-focused teams may not have readily available.
- Over-prototyping can delay actual development if the team cycles through too many iterations without committing to an implementation.

## How It Could Be

> The following scenario illustrates how prototypes prevent costly mistakes during legacy replacement.

An insurance company was replacing a legacy policy administration system that agents had used for 15 years. The development team built an initial prototype based on requirements documents and presented it to a group of agents. Within 30 minutes, agents identified that the prototype's linear workflow for policy endorsements would triple the time required compared to the legacy system's multi-panel approach that let them view and edit several sections simultaneously. Without the prototype, this fundamental workflow mismatch would not have been discovered until acceptance testing months later. The team redesigned the interface around a tabbed layout that preserved the multi-section editing capability, validated it with a second prototype round, and then proceeded to implementation with high confidence.
