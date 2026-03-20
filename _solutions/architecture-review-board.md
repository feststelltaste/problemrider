---
title: Architecture Review Board
description: Establishment of a committee for monitoring and controlling architecture development
category:
- Architecture
- Management
quality_tactics_url: https://qualitytactics.de/en/maintainability/architecture-review-board
problems:
- stagnant-architecture
- technology-stack-fragmentation
- inconsistent-codebase
- architectural-mismatch
- decision-avoidance
- convenience-driven-development
- high-technical-debt
layout: solution
---

## How to Apply ◆

> In legacy environments, an architecture review board provides the organizational structure needed to make deliberate, coordinated architectural decisions rather than letting the system continue to decay through uncoordinated individual choices.

- Form a board with representatives from each major development team plus senior architects, keeping it small enough to make decisions efficiently (five to eight members typically works well).
- Define a clear charter that specifies which decisions require board review (cross-team changes, new technology introductions, major refactoring) and which are delegated to individual teams.
- Meet regularly on a short cadence (biweekly or monthly) with a structured agenda rather than only convening for major decisions, so that the board stays informed about ongoing architectural evolution.
- Publish all board decisions, including rationale and dissenting opinions, in an accessible decision log so that teams understand not just what was decided but why.
- Use the board to coordinate modernization efforts across teams, ensuring that different teams' changes are architecturally compatible and move toward a shared target state.
- Review the board's effectiveness periodically and adjust its scope and processes to prevent it from becoming either a rubber stamp or a bottleneck.

## Tradeoffs ⇄

> An architecture review board provides coordinated architectural direction but can become a bottleneck or ivory tower if not managed carefully.

**Benefits:**

- Prevents uncoordinated technology proliferation by providing a forum for evaluating and approving new technology introductions.
- Ensures cross-team architectural consistency, which is especially important when multiple teams modify different parts of the same legacy system.
- Creates accountability for architectural decisions, reducing the tendency to defer hard choices indefinitely.
- Provides a venue for sharing architectural knowledge and patterns across teams that might otherwise work in isolation.

**Costs and Risks:**

- A board that requires approval for too many decisions becomes a bottleneck that slows development and frustrates teams.
- Board members who are disconnected from day-to-day development may make decisions that are theoretically sound but practically infeasible.
- Without clear delegation rules, teams may be unsure whether they need board approval, leading to either unnecessary delays or unapproved changes.
- The board may develop a bias toward conservatism, resisting necessary changes because they introduce short-term risk.

## How It Could Be

> The following scenario shows how an architecture review board coordinates legacy modernization across teams.

A government agency with six development teams maintaining a shared legacy platform established an architecture review board after discovering that three teams had independently begun adopting different microservices frameworks. The board evaluated all three options, selected one as the standard, and created migration guidelines that all teams would follow. They also established a "technology radar" that classified technologies into four categories: adopt, trial, assess, and hold. The radar made it clear which technologies were approved for production use and which were still being evaluated. Over two years, the board reviewed 45 significant architectural proposals, approved 38 (often with modifications), and rejected 7 with explanations. The rejected proposals included two cases where teams wanted to introduce technologies already on the "hold" list, preventing further fragmentation.
