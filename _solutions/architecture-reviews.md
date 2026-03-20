---
title: Architecture Reviews
description: Regular systematic review of the software architecture
category:
- Architecture
- Process
quality_tactics_url: https://qualitytactics.de/en/maintainability/architecture-reviews/
problems:
- insufficient-design-skills
- misunderstanding-of-oop
- procedural-background
- suboptimal-solutions
- complex-implementation-paths
- uncontrolled-codebase-growth
- single-entry-point-design
- high-coupling-low-cohesion
- gold-plating
- second-system-effect
- cargo-culting
- rapid-prototyping-becoming-production
- convenience-driven-development
layout: solution
---

## How to Apply ◆

> In legacy systems, architecture reviews serve a dual purpose: they catch design problems before they become entrenched, and they build the design skills that the team needs to stop creating those problems in the first place. The review process itself is a teaching mechanism.

- Schedule architecture reviews at two cadences: lightweight reviews before implementation of any feature that touches more than three modules or introduces a new component, and comprehensive reviews quarterly to assess the overall trajectory of the system's structural health.
- Use a structured review format rather than free-form discussion to prevent reviews from becoming subjective debates. The ATAM (Architecture Tradeoff Analysis Method) provides a proven format for evaluating architectural decisions against quality attribute scenarios, but even a simple checklist covering coupling, cohesion, separation of concerns, and appropriate use of patterns is effective.
- Include at least one reviewer external to the team for comprehensive reviews. Internal reviewers share the same assumptions and blind spots as the developers; an external perspective identifies patterns like cargo culting, gold plating, or the second-system effect that the team may not recognize because they are too close to the decisions.
- Review architectural decisions against the actual business requirements they serve, not against abstract "best practices." This directly prevents gold plating and the second-system effect by forcing the team to justify every architectural element in terms of a concrete business need.
- Make architecture reviews a learning opportunity by asking developers to explain their design rationale. When developers with procedural backgrounds or insufficient design skills present their designs, the review discussion itself teaches them about alternative approaches, design principles, and the trade-offs involved.
- Evaluate whether the complexity of the proposed solution matches the complexity of the problem it solves. If a simple CRUD operation is being implemented through three layers of abstraction, an event bus, and a custom framework, the review should challenge the proportionality. This catches cargo culting and CV-driven over-engineering.
- Review prototype-to-production transitions explicitly. When a prototype or proof-of-concept is proposed for production deployment, the review should evaluate it against production requirements including error handling, security, scalability, observability, and operational maintainability.
- Document review outcomes as Architecture Decision Records (ADRs) that capture what was reviewed, what alternatives were considered, what was decided, and why. This creates an institutional memory that prevents the same bad decisions from being repeated and gives new team members insight into the system's design rationale.
- Track architectural metrics over time — coupling between modules, component size, dependency depth, circular dependency count — and review trends rather than just snapshots. A single review shows current state; trend analysis reveals whether the system is improving or degrading.

## Tradeoffs ⇄

> Architecture reviews provide the oversight mechanism that prevents legacy systems from accumulating the design problems that make them legacy in the first place, but they require organizational commitment and must be balanced against delivery velocity.

**Benefits:**

- Catches design problems early when they are inexpensive to fix, preventing the accumulation of structural issues that make legacy systems expensive to maintain — a design flaw caught during review costs hours to fix, while the same flaw caught in production costs months.
- Builds design skills across the team by exposing developers to architectural reasoning, alternative approaches, and trade-off analysis during every review, directly addressing the insufficient design skills and misunderstanding of OOP that create legacy problems.
- Prevents cargo culting and CV-driven development by requiring teams to articulate why a specific technology or pattern is appropriate for their context, filtering out choices that cannot be justified beyond "it's popular" or "I want it on my resume."
- Identifies single-entry-point designs, god objects, and high-coupling patterns before they become entrenched, when the cost of restructuring is still manageable.
- Creates accountability for architectural decisions through documented records, making it visible when convenience-driven shortcuts were chosen over sustainable solutions.

**Costs and Risks:**

- Architecture reviews add time to the development process, and if they are too heavyweight or too frequent, they slow delivery without proportional benefit — the process must be calibrated to the team's cadence and the system's risk profile.
- Reviews conducted by inexperienced reviewers may not catch subtle design problems or may raise false concerns, creating friction without improving quality. Effective reviews require reviewers who have seen enough systems to recognize patterns and anti-patterns.
- If review feedback is delivered as criticism rather than teaching, developers may become defensive and begin designing to pass reviews rather than to solve problems — the same dynamic as defensive coding practices, shifted to the architectural level.
- Reviews can become gatekeeping mechanisms that concentrate architectural decision-making in a small group, creating bottlenecks and disempowering the broader team rather than building their skills.
- In fast-moving environments, the delay introduced by architecture reviews may conflict with market pressures, and teams may bypass reviews for "urgent" work, which is often exactly the work that most needs review.

## Examples

> The following scenarios illustrate how architecture reviews have been used to prevent and correct design problems in legacy system environments.

A retail company's development team proposed replacing their monolithic order management system with a microservices architecture of 15 services, inspired by conference talks from large technology companies. During an architecture review, an external reviewer asked the team to map each proposed service to a specific business capability and explain why it needed to be independently deployable. The exercise revealed that only four of the fifteen services had genuine independence requirements; the remaining eleven were fine-grained decompositions that would create distributed complexity without business benefit. The review guided the team toward a modular monolith with four clearly bounded modules, which delivered the isolation benefits they needed without the operational overhead of a distributed system. The team later acknowledged that without the review, they would have spent eighteen months building infrastructure for services that never needed to be independent.

A government software project conducted quarterly architecture reviews that tracked coupling metrics across releases. Over three reviews, the metrics showed that coupling between the citizen enrollment module and the eligibility determination module was increasing steadily — from 12 cross-module dependencies to 34 over nine months. The review identified that developers were taking convenient shortcuts by directly querying each other's database tables rather than using the defined API. The review resulted in an architectural rule enforced by CI tooling: no module could access another module's database schema directly. The coupling metric decreased to 8 within two releases, and the eligibility module team was subsequently able to replace their database implementation without affecting enrollment.

A healthcare technology company used architecture reviews specifically to address design skill gaps in a team where most developers came from procedural programming backgrounds. Each review included a "design alternatives" segment where the reviewer presented how the same problem could be solved using different design approaches — comparing the team's procedural solution with an object-oriented alternative, explaining trade-offs rather than mandating the OOP approach. Over twelve months and twenty-four reviews, the team's design patterns shifted measurably: static utility classes decreased from 60% of new classes to 15%, and the use of interfaces and polymorphism increased correspondingly. The reviews functioned as continuous architectural mentoring embedded in the delivery process.
