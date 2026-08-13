---
title: Internal Technical Coaching
description: Give someone explicit, protected time to raise the technical practice
  of the team by working alongside people rather than by teaching at them.
category:
- Team
- Process
- Culture
problems:
- inexperienced-developers
- skill-development-gaps
- limited-team-learning
- slow-knowledge-transfer
- misunderstanding-of-oop
- procedural-programming-in-oop-languages
- cargo-culting
- inappropriate-skillset
- inadequate-mentoring-structure
- reviewer-inexperience
- clever-code
- inconsistent-execution
- defensive-coding-practices
- incomplete-knowledge
- inconsistent-knowledge-acquisition
- knowledge-dependency
- author-frustration
- difficult-to-understand-code
- extended-research-time
- high-turnover
- insufficient-design-skills
- legacy-skill-shortage
- mentor-burnout
- new-hire-frustration
- reduced-team-flexibility
- reviewer-anxiety
- implementation-partner-dependency
- low-code-customization-sprawl
layout: solution
related_solutions:
- slug: pair-and-mob-programming
  similarity: 0.75
- slug: technical-skills-development
  similarity: 0.7
- slug: code-reading-sessions
  similarity: 0.7
- slug: communities-of-practice
  similarity: 0.7
- slug: structured-onboarding-program
  similarity: 0.65
- slug: code-review-process-reform
  similarity: 0.65
---

## Description

Internal technical coaching is the deliberate assignment of someone — with protected time and an explicit mandate — to raise the technical practice of a team by working alongside its members on their real work. It differs from training in that it happens in the codebase rather than in a classroom, and from mentoring in that it targets the team's practice rather than an individual's career. The distinction that makes it work is protected time: an experienced developer who is also expected to deliver features will always deliver features, because that is what is measured. In legacy contexts the need is acute and the usual remedies do not fit. External courses teach patterns that assume greenfield conditions, and the skills that actually matter — reading unfamiliar code, breaking dependencies safely, deciding when not to refactor — are learned by doing them next to someone who already can.

## How to Apply ◆

> The techniques that make legacy work tractable are rarely taught anywhere and are almost never written down; they propagate by working alongside someone who has them.

- **Protect the time explicitly** — a stated share of the coach's capacity, typically twenty to fifty percent, removed from delivery commitments. Coaching that is expected to happen alongside a full delivery load does not happen, and the failure is invisible because everyone is busy.
- **Coach on the team's actual work**, not on exercises. The value is in showing how to approach this legacy module, this untestable class, this unclear requirement — the transfer does not survive abstraction into a toy example.
- **Work with people, not at them.** The coach pairs on real tasks, takes the keyboard occasionally and gives it back, and lets the other person struggle productively. A coach who solves the problem has produced a solution rather than a capability.
- **Pick a small number of practices** and pursue them until they stick. A coach promoting testing, refactoring, domain modeling, and review quality simultaneously achieves surface familiarity with all four. One practice adopted properly is worth four attempted.
- **Choose the coach for teaching ability, not seniority.** The best developer on a team is frequently a poor coach, because their expertise has become tacit and they cannot decompose it. Willingness to work at someone else's pace matters more than raw skill.
- Combine formats deliberately: **pairing for depth, code reading sessions for breadth, and short focused workshops** for a specific technique. Different knowledge transfers through different channels, and pairing alone reaches too few people.
- Give the coach a **mandate to change practice, not just to advise**. A coach who can only recommend is ignored the moment there is deadline pressure. The mandate should be explicit and known to the team.
- **Agree observable goals** at the outset — more people able to work in a given subsystem, tests accompanying changes in a given area, review comments shifting from style to substance. Coaching without goals becomes an indefinite role whose value nobody can assess and which is cut in the first budget review.
- **Rotate who is coached** rather than concentrating on the newest joiners. Long-tenured developers in a legacy system often carry the most entrenched habits, and they are also the ones whose practice most shapes everyone else's.
- **Plan for the coach to become unnecessary.** The measure of success is that the practice persists without them, which means deliberately handing over — having coached developers coach the next group.

## Tradeoffs ⇄

> Coaching raises the practice of a whole team durably, at the cost of a significant share of an experienced person's capacity and with results that are slow and hard to attribute.

**Benefits:**

- Skills transfer in the context where they will be used, which is far more durable than classroom training that must be translated to the real codebase by the learner.
- The specific competencies legacy work demands — safe dependency breaking, incremental refactoring, reading unfamiliar code — are taught, and these are essentially unavailable from external training.
- Practice becomes more consistent across the team, which reduces the divergence in how the same problems get solved in different corners of the system.
- Review quality improves as more people become capable of substantive review, relieving the small group that currently carries it.
- Cargo-culted practices get examined, because a coach's job includes asking why something is done this way — a question nobody inside the team asks any more.

**Costs and Risks:**

- A substantial fraction of an experienced developer's capacity leaves delivery, which is felt immediately while the benefits arrive over quarters.
- The role is difficult to evaluate, and coaching is therefore vulnerable in any budget or headcount review despite being effective.
- A poorly chosen coach can entrench their own preferences as team standards, which is worse than no coaching if those preferences are dated or dogmatic.
- Coaching pushed onto a team that did not ask for it produces polite resistance, and the coach spends their time on people who are complying rather than learning.
- The coach can become a dependency in their own right if the handover step is skipped, leaving the team's practice reliant on one person again.

## How It Could Be

A team of eleven maintaining an energy trading platform had two developers who could write tests for the legacy pricing modules and nine who could not, with the result that most changes shipped untested. Formal training had been tried: a two-day course on unit testing, after which nothing changed, because the techniques presented assumed injectable dependencies and the actual code had none. The team instead assigned one of the two capable developers 40 percent of their time as a coach for six months, with one goal: every change to the pricing subsystem ships with a test. He paired with each developer on their own tickets, teaching extract-and-override and characterization testing on the actual classes they were changing. After six months, nine of eleven were writing tests for that subsystem unprompted, and the test count in the area had gone from 12 to over 400.

The same coaching arrangement surfaced something the team had not been looking for. Pairing across the whole team revealed that four developers were independently reimplementing a date-handling routine because each believed the shared utility was broken. It was not broken; it was undocumented and its parameter order was surprising. The coach's cross-team view was what made the pattern visible, since each developer had encountered it alone and worked around it alone. Naming the parameters and writing six lines of documentation removed a source of duplicated logic that had been quietly spreading for two years.
