---
title: Communities of Practice
description: Create standing cross-team groups around a shared craft — testing, a subsystem, a language — so that knowledge and standards spread horizontally rather than through management.
category:
- Team
- Communication
- Culture
problems:
- team-silos
- knowledge-silos
- inconsistent-execution
- skill-development-gaps
- limited-team-learning
- duplicated-research-effort
- technology-isolation
- duplicated-effort
- inconsistent-knowledge-acquisition
- undefined-code-style-guidelines
- technology-stack-fragmentation
- slow-knowledge-transfer
- team-confusion
- author-frustration
- automated-tooling-ineffectiveness
- code-duplication
- communication-risk-within-project
- convenience-driven-development
- difficult-to-understand-code
- extended-research-time
- fear-of-conflict
- high-turnover
- inappropriate-skillset
- inconsistent-naming-conventions
- individual-recognition-culture
- inexperienced-developers
- insufficient-design-skills
- language-barriers
- legacy-skill-shortage
- mentor-burnout
- new-hire-frustration
- nitpicking-culture
- procedural-programming-in-oop-languages
- reduced-review-participation
- reduced-team-flexibility
- team-churn-impact
- team-members-not-engaged-in-review-process
- unclear-sharing-expectations
- uneven-workload-distribution
- unmotivated-employees
layout: solution
---

## Description

A community of practice is a standing, voluntary group of people from different teams who share a craft or a concern — testing, the database, a programming language, a particular legacy subsystem — and who meet regularly to exchange what they are learning and agree on shared practice. It supplies a channel that team-aligned organizations structurally lack. Once teams are organized around capabilities or products, which is usually correct for delivery, knowledge stops flowing sideways: four teams solve the same testing problem four times, and four different conventions emerge for the same thing. A community of practice is the horizontal connection that restores that flow without reorganizing anyone. In legacy landscapes it has a second use, since the people who know a given decades-old subsystem are frequently scattered across teams, and the community may be the only place they ever talk to each other.

## How to Apply ◆

> In a long-lived system the expertise on any given subsystem is usually distributed across teams by accident of history, and nobody has ever assembled it in one room.

- **Form around a genuine shared concern**, not around an organizational category. A community for "backend developers" has nothing specific to discuss; one for "the batch processing subsystem" or "how we test legacy code" has an agenda immediately.
- Keep membership **voluntary and self-selected**. Attendance mandated by management produces rooms full of people waiting for it to end. A community that nobody wants to attend is telling you the topic is not a shared concern.
- Give it a **named coordinator** with a small amount of protected time. Communities without someone responsible for scheduling and agenda quietly stop meeting after the third session, and the failure is gradual enough that nobody notices.
- Meet on a **predictable cadence** — monthly works for most. Weekly is too demanding for a voluntary group; quarterly is too infrequent to build the relationships that make the exchange work.
- **Ground sessions in real work.** A member walks through a problem they are facing, a solution they built, or an incident they handled. Presentations on general topics attract an audience once and then attendance decays.
- Give the community **authority over shared conventions** in its domain — coding standards for a language, testing approach, the shared library's interface. Recommendations without authority get ignored, and the community becomes a discussion group. This is the difference between one that matters and one that does not.
- Produce **something durable** from each session: a decision, a note in a shared space, an addition to a convention. A community that generates only conversation loses its justification the first time someone questions the time.
- Use it deliberately for **legacy subsystem knowledge**. A community around a specific old system, drawing everyone who touches it regardless of team, is often the fastest way to consolidate a fragmented picture of how it actually works.
- **Let communities end.** When a topic is settled or the shared concern dissolves, closing the group is a success rather than a failure. Zombie communities consume calendar time and discredit the format.

## Tradeoffs ⇄

> Communities restore horizontal knowledge flow cheaply, but they consume time across many teams and decay without an owner and without real authority.

**Benefits:**

- Knowledge crosses team boundaries without requiring reorganization, which is the specific gap that capability-aligned teams create.
- Duplicated investigation declines, because someone in the room has usually already solved the problem or knows why it cannot be solved.
- Conventions converge across teams by agreement among practitioners rather than by architectural decree, which makes them stick.
- Scattered expertise on a legacy subsystem gets consolidated, and the community often becomes the de facto owner of knowledge that had no owner.
- Developers get professional development and a peer group outside their immediate team, which measurably affects retention among specialists.

**Costs and Risks:**

- Meeting time accumulates across many people, and the cost is real while the benefit is diffuse and hard to attribute.
- Communities decay quietly. Without a coordinator with protected time, they stop meeting and nobody makes a decision about it.
- Without authority over conventions they become talking shops, which wastes the time of exactly the people whose time is most contested.
- They can develop into gatekeeping bodies that impose the preferences of their most vocal members on teams that were not represented.
- In organizations under sustained delivery pressure, voluntary cross-team activity is the first thing to disappear from calendars, so the format needs visible management support to survive.

## How It Could Be

An organization with six product teams found that each had independently built its own approach to testing legacy code, and that four of the six had separately concluded it was impossible in their area. A testing community of practice was formed with a coordinator given four hours a month and a monthly session. The second session was one developer demonstrating extract-and-override on a real untestable class from her team's codebase. Three other teams applied the technique within a month. By the sixth session the community had agreed a shared set of conventions for characterization tests and had built a small shared library of test fixtures for the two most commonly stubbed external services — work that no single team would have justified alone and that all six now depend on.

A second community formed around a specific 1990s mainframe subsystem that eleven people across four teams touched occasionally, none of whom understood it fully. The first three sessions were spent collectively reconstructing what it did, and the notes from those sessions became the first documentation the subsystem had ever had. The most valuable output was less expected: two of the eleven discovered they had been maintaining what they each believed were separate interfaces, which turned out to be two entry points into the same code path with subtly different validation. One of them had been the source of an intermittent data problem the other team had been investigating, separately, for four months.
