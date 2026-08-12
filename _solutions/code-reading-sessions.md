---
title: Code Reading Sessions
description: Read existing code together, out loud, as a scheduled group activity — the fastest way to spread understanding of a system nobody fully understands.
category:
- Team
- Communication
- Code
problems:
- difficult-to-understand-code
- clever-code
- knowledge-silos
- slow-knowledge-transfer
- limited-team-learning
- inexperienced-developers
- misunderstanding-of-oop
- procedural-programming-in-oop-languages
- knowledge-gaps
- incomplete-knowledge
- difficult-developer-onboarding
- legacy-system-documentation-archaeology
- cargo-culting
- skill-development-gaps
- bloated-class
- copy-paste-programming
- global-state-and-side-effects
- inconsistent-knowledge-acquisition
- knowledge-dependency
- maintenance-bottlenecks
- superficial-code-reviews
- code-duplication
- complex-implementation-paths
- defensive-coding-practices
- extended-research-time
- hidden-side-effects
- inappropriate-skillset
- information-decay
- insufficient-design-skills
- legacy-skill-shortage
- mentor-burnout
- monolithic-functions-and-classes
- new-hire-frustration
- over-reliance-on-utility-classes
- poor-encapsulation
- procrastination-on-complex-tasks
- reduced-team-flexibility
- reviewer-anxiety
- team-churn-impact
- team-members-not-engaged-in-review-process
layout: solution
---

## Description

A code reading session is a scheduled meeting in which a group reads a piece of existing code together, out loud, and works out what it does. It inverts the usual code review: nothing is being proposed, nothing is being judged, and the code is typically years old. The purpose is comprehension, and the reason it works better than individual reading is that understanding legacy code is a process of forming and discarding hypotheses, which happens far faster when several people do it aloud. Teams underuse this because reading code has no visible output and feels unproductive next to writing it. But in a legacy system the limiting factor on almost every task is understanding rather than typing, and the understanding is currently distributed such that each person holds a different fragment. A reading session is the cheapest available mechanism for merging those fragments.

## How to Apply ◆

> The code most worth reading together is the code everyone avoids — which is exactly the code nobody will ever read alone.

- **Choose the target deliberately**: a module that several people will need to touch soon, an area with concentrated knowledge, or code that keeps producing defects. Hotspot data — frequently changed, frequently implicated in bugs — picks good candidates when intuition is uncertain.
- Keep sessions **short and bounded**: sixty to ninety minutes, and one clearly delimited piece of code. Attempting to cover a whole subsystem produces a tour rather than an understanding.
- **Read the code, do not summarize it.** Project it, step through it line by line, and let people ask about anything. The moment it becomes a prepared presentation by whoever knows it already, the mechanism stops working — the value is in the group's questions, not the expert's narration.
- **Encourage the naive question explicitly.** "Why is this here?" and "what happens if this is null?" from someone unfamiliar with the code regularly uncover genuine defects and assumptions the familiar readers had stopped seeing.
- Have someone **take notes and commit them**, as comments, as a short document, or as a diagram. A session that produces nothing durable has to be repeated for every new person. The notes are also the closest thing to documentation the module is likely to get.
- **Record the questions nobody could answer.** These are the highest-value items in the session: they mark the parts of the system where knowledge has actually been lost, and they form the agenda for investigation or for the next session.
- **Rotate who chooses the code**, so that sessions cover what each person finds impenetrable rather than what one person finds interesting.
- Use it deliberately for **onboarding**: a new joiner attending four or five reading sessions across the main subsystems acquires a working map far faster than by reading documentation or by being told to explore the codebase.
- Keep it **separate from review and from criticism.** The code being read is usually poor — that is often why it was chosen. A session that turns into a critique of absent authors becomes unsafe for whoever wrote the code being read next week.

## Tradeoffs ⇄

> Reading together spreads understanding quickly and finds real defects, at the cost of several people's time on an activity with no immediate deliverable.

**Benefits:**

- Understanding spreads across the team quickly, which directly addresses the concentration of knowledge that makes legacy maintenance fragile.
- Defects and unsafe assumptions surface during reading, particularly through questions from people unfamiliar with the code, at a cost far lower than finding them in production.
- Documentation is produced as a byproduct, written by people who just discovered what was unclear and aimed at exactly that audience.
- Onboarding accelerates substantially, since a new developer gains context on the real system rather than on its documented abstraction.
- Less experienced developers observe how experienced ones form hypotheses about unfamiliar code, which is a skill that is otherwise almost never taught explicitly.

**Costs and Risks:**

- Several people spend an hour or more with no deliverable, which is difficult to justify to anyone measuring output and is the first thing dropped under pressure.
- Sessions can drift into critique of past developers, which is unproductive and makes people defensive about code they wrote themselves.
- Without notes, the understanding evaporates and the session has to be repeated for the next person, which quickly makes the practice feel wasteful.
- A dominant expert can turn the session into a lecture, which conveys structure but not the reasoning that makes the knowledge usable.
- Understanding decays if nobody works in the code afterward, so sessions are best scheduled shortly before the area is actually touched.

## How It Could Be

A team of six maintaining a subscription billing platform had one developer who understood the proration logic — roughly 1,200 lines that had accumulated over eleven years — and every change touching it queued behind him. They scheduled four ninety-minute reading sessions over two weeks. In the second session, a developer who had joined four months earlier asked why a particular branch subtracted a day, and nobody could answer. Investigation found it compensated for a timezone handling error elsewhere in the system, and that the compensation was wrong for two of the eleven supported countries — a defect that had been quietly producing incorrect invoices for an estimated three years. By the end of the four sessions, three developers could work in the module, and the notes taken during the sessions became the first documentation it had ever had.

The same team adopted reading sessions for onboarding. Their previous approach had been a written architecture document and an instruction to explore the codebase, after which new developers took roughly four months to make an unsupervised change to a core module. New joiners now attend one reading session per week for their first six weeks, covering the five main subsystems. The time to first unsupervised change in a core module fell to about six weeks. An unanticipated effect was that the long-tenured developers found the sessions valuable too: in the session on the payment gateway integration, two people who had each worked on it for years discovered they held incompatible beliefs about how retries interacted with idempotency keys, and one of them was wrong.
