---
title: Pair Programming
description: Two developers work together on a task at one workstation
category:
- Team
- Code
- Process
quality_tactics_url: https://qualitytactics.de/en/maintainability/pair-programming/
problems:
- knowledge-silos
- tacit-knowledge
- implicit-knowledge
- difficult-developer-onboarding
- lower-code-quality
- reviewer-inexperience
- inadequate-mentoring-structure
- slow-knowledge-transfer
- inappropriate-skillset
- knowledge-dependency
- inexperienced-developers
- skill-development-gaps
- limited-team-learning
- inconsistent-knowledge-acquisition
layout: solution
---

## How to Apply ◆

> In legacy systems where critical knowledge is concentrated in a small number of individuals, pair and mob programming are the most direct interventions for transferring that knowledge before it is lost.

- Prioritize pairing on the parts of the legacy codebase with the highest bus factor — the modules where only one person truly understands the design — and treat those sessions explicitly as knowledge transfer, not just development.
- Use strong-style pairing (navigator dictates intent, driver types) when a senior developer with legacy knowledge works alongside a junior who is new to the system; the senior must articulate every assumption they would otherwise apply silently.
- Apply mob programming to particularly dangerous or complex legacy modules where multiple perspectives are needed and where a mistake could have broad impact — the entire team working through a fragile shared component together is often safer than any individual touching it alone.
- When debugging obscure legacy failures, pair rather than solo investigate; the navigator often spots the relevant pattern (a known quirk of the system, an undocumented dependency) far faster than a single developer tracing through an unfamiliar stack.
- Rotate pairs deliberately across unfamiliar modules rather than defaulting to the person who "already knows" an area — the goal is to eliminate single points of knowledge, not to optimize for short-term efficiency.
- Timebox pairing sessions to two or three hours at a stretch; legacy code exploration is cognitively intensive and extended pairing without breaks produces diminishing returns.
- Use mob programming sessions for onboarding new developers into the legacy codebase; a group walkthrough of a key module in mob style, with the newcomer as driver, surfaces implicit knowledge faster than any written documentation could.
- Track which team members have paired on which modules using a simple matrix; if the same pairs keep forming, the knowledge distribution is not improving and rotation needs to be enforced.

## Tradeoffs ⇄

> Pair and mob programming require investing two or more developers' time in one task, which is especially contentious in legacy teams that are already under pressure — but the cost of a knowledge silo becoming permanent is typically far higher.

**Benefits:**

- Distributes knowledge of notoriously siloed legacy modules across more team members, directly reducing the risk of critical knowledge walking out the door when a long-tenured developer leaves.
- Catches legacy-specific errors — incorrect assumptions about external system behavior, misunderstood shared state, implicit ordering constraints — at the moment of change rather than in production.
- Produces better designs for legacy modifications, because the navigator maintains the broader context while the driver focuses on implementation, reducing the tunnel vision that individual developers develop when working alone in unfamiliar code.
- Accelerates onboarding into the legacy system dramatically compared to solo exploration, because the pairing partner provides real-time explanation of the system's quirks and history.
- Reduces the length and complexity of subsequent code reviews, since code written by a pair has already been reviewed continuously as it was written.

**Costs and Risks:**

- In teams with only one or two people who understand a critical legacy module, pairing that expert for knowledge transfer means they produce less individually in the short term, which managers focused on throughput may resist.
- Personality mismatches, experience gaps, and differing working styles are magnified in pair settings when dealing with frustrating legacy code; the stress of working in difficult code can spill into the collaboration.
- Pairing on legacy code with no tests or documentation is mentally exhausting faster than pairing on well-structured greenfield code; sessions need to be shorter and breaks more frequent.
- In organizations that measure individual developer productivity, pair programming is invisible at the individual level — two developers closing one ticket looks like half the productivity even when quality outcomes are better.
- Remote pairing introduces additional friction when dealing with legacy code that requires local environment setup, proprietary tooling, or access to institutional systems that are difficult to share over screen-sharing tools.

## Examples

> The following scenarios illustrate how pair and mob programming address knowledge concentration and quality problems in legacy system contexts.

An insurance company's actuarial calculation engine had been maintained by a single developer for eleven years. When that developer announced their retirement, the team organized a series of mob programming sessions in which the retiring developer navigated while junior team members drove implementation of planned features. Over four months, three developers acquired enough understanding of the calculation logic to work on it independently. The mob sessions also produced the most complete documentation the module had ever had, captured directly in the code and in the written records of the sessions themselves.

A government agency running a legacy tax processing system needed to integrate a new digital identity provider. The integration touched a module that two developers had built years earlier using a bespoke protocol that nobody else understood. Rather than assigning the integration to the original authors alone, the team ran a week of mob programming sessions with all four available developers, including the two who had no prior experience with the module. The mob approach forced the original authors to explain their protocol decisions aloud, surfaced three undocumented edge cases that would have caused failures in production, and left the team with four people who could maintain the integration going forward.

A fintech startup had grown its payment processing service through years of expedient additions until it was both critical and fragile. When the team began a controlled modernization effort, they adopted a policy of pairing on every change to the payment service, regardless of how small. The navigator role was rotated among the team's most experienced developers. Over six months, this policy produced two outcomes: the accumulation of new debt in the payment service slowed markedly because the navigator consistently challenged quick fixes, and the knowledge of the service's quirks spread from the two original authors to five additional developers who could now respond to production incidents without escalating.
