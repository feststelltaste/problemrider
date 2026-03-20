---
title: Evolutionary Requirements Development
description: Detailing and refining requirements incrementally throughout the project
category:
- Requirements
- Process
- Communication
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/evolutionary-requirements-development/
problems:
- inadequate-requirements-gathering
- requirements-ambiguity
- frequent-changes-to-requirements
- implementation-starts-without-design
- scope-creep
- changing-project-scope
- stakeholder-developer-communication-gap
- no-continuous-feedback-loop
- eager-to-please-stakeholders
- missed-deadlines
- planning-dysfunction
- stakeholder-frustration
- feature-creep
layout: solution
---

## How to Apply ◆

> In legacy systems where requirements were often defined years ago and the original stakeholders may no longer be available, evolutionary requirements development replaces the impossible goal of complete upfront specification with a disciplined process of progressive refinement that keeps pace with actual understanding.

- Begin each project increment with a requirements workshop that brings together developers, business stakeholders, and users to collaboratively refine the next slice of requirements. In legacy contexts, these workshops serve double duty: they capture current needs and surface undocumented behaviors of the existing system that must be preserved or deliberately changed.
- Use lightweight specification formats — user stories with acceptance criteria, specification by example, or decision tables — rather than heavyweight requirements documents that become stale before development begins. In legacy modernization, acceptance criteria should explicitly state whether existing behavior is being preserved or intentionally changed.
- Maintain a living requirements backlog that is continuously groomed rather than a requirements specification that is approved once and then treated as immutable. Rank items by business value and technical risk, and accept that lower-ranked items may change significantly by the time the team reaches them.
- Require that every requirement includes at least one concrete example of expected behavior, expressed in terms that both developers and stakeholders can verify. Abstract requirements like "the system should be fast" or "improve the user experience" are not actionable and should be rejected until they can be expressed as testable scenarios.
- Conduct regular backlog refinement sessions where the team reviews upcoming requirements, identifies ambiguities, asks clarifying questions, and estimates effort. These sessions should happen at least one iteration ahead of development to prevent the team from starting implementation with unclear requirements.
- Establish explicit "just enough" design checkpoints before implementation begins: the team discusses architectural implications and component interactions for the upcoming requirements without producing comprehensive design documents. For legacy systems, this includes analyzing how new requirements interact with existing system constraints.
- Create a shared glossary of domain terms that both business and technical stakeholders agree on, and maintain it as a living document. In legacy systems, the same term often means different things to different departments because the system evolved over decades under different teams.
- Track requirements changes as a normal part of the process rather than treating them as failures. Measure the rate of change to identify when instability indicates a deeper problem — such as unclear product vision or conflicting stakeholder interests — rather than healthy evolution.

## Tradeoffs ⇄

> Evolutionary requirements development trades the comfort of a "complete" specification for the ability to adapt as understanding grows, which is particularly valuable in legacy contexts where the true requirements are often discovered rather than specified.

**Benefits:**

- Prevents the costly pattern of building to a specification that proves wrong, which is especially dangerous in legacy modernization where the gap between documented requirements and actual system behavior can be enormous.
- Reduces requirements ambiguity by deferring detailed specification until the team has enough context to write precise, testable criteria, rather than guessing at details months before implementation.
- Naturally accommodates requirement changes by treating them as expected rather than disruptive, reducing the adversarial dynamic between stakeholders who need changes and developers who resist them.
- Creates regular opportunities for stakeholders and developers to build shared understanding, gradually closing the communication gap that plagues many legacy projects.
- Enables earlier detection of conflicting requirements by discussing them in concrete terms before significant development effort is invested.

**Costs and Risks:**

- Requires sustained stakeholder availability throughout the project, which can be difficult when business experts are already stretched thin supporting the legacy system they are also trying to replace.
- Teams accustomed to receiving complete specifications may feel uncomfortable working with requirements that are explicitly incomplete, perceiving it as poor planning rather than intentional incrementalism.
- Without discipline, evolutionary requirements can degenerate into "we'll figure it out as we go," which is not progressive refinement but absence of planning — the team must maintain a clear vision of the overall scope even as details evolve.
- Stakeholders who expect predictable long-term roadmaps may struggle with a process that deliberately defers detailed requirements until closer to implementation, creating planning credibility concerns at the organizational level.
- In fixed-price or regulated contexts, evolutionary requirements conflict with contractual or compliance expectations for upfront specification, requiring careful negotiation of how requirements baselines are managed.

## How It Could Be

> The following scenarios illustrate how evolutionary requirements development addresses the challenges of incomplete or changing requirements in legacy system contexts.

A logistics company was modernizing its order management system that had been in production for twelve years. The original requirements documents were outdated and incomplete, and the three business analysts who understood the system had retired. Rather than attempting to reverse-engineer a complete specification, the team adopted two-week refinement cycles where they worked with warehouse operators and customer service staff to document the requirements for the next two features in detail, using concrete examples from actual orders. Each cycle produced testable acceptance criteria and surfaced undocumented business rules embedded in the legacy code. Over six months, the team built a reliable, growing specification that was always accurate because it was always current, and they avoided the three-month upfront analysis phase that had derailed two previous modernization attempts.

An insurance company embarked on a policy administration replacement where stakeholders initially provided a 200-page requirements document. The development team discovered within the first month that many requirements contradicted the actual behavior of the legacy system, and business users confirmed that the documented rules had been overridden by manual processes years ago. The team shifted to evolutionary requirements development, working with claims adjusters to define requirements one business process at a time, validating each against the legacy system's actual behavior before building. This approach extended the initial timeline by three weeks but eliminated an estimated four months of rework that would have resulted from building to the inaccurate specification. More importantly, it gave the business stakeholders confidence that the replacement would actually handle their real workflows, not just the officially documented ones.
