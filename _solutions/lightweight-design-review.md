---
title: Lightweight Design Review
description: Discuss the intended approach for non-trivial changes before implementation, in a short session with a written sketch, so that design problems surface before code exists.
category:
- Architecture
- Code
- Process
problems:
- suboptimal-solutions
- complex-implementation-paths
- insufficient-design-skills
- second-system-effect
- rapid-prototyping-becoming-production
- quality-compromises
- large-pull-requests
- procedural-programming-in-oop-languages
- misunderstanding-of-oop
- process-design-flaws
- large-feature-scope
- convenience-driven-development
- god-object-anti-pattern
- accumulated-decision-debt
- analysis-paralysis
- communication-risk-within-project
- defensive-coding-practices
- delayed-decision-making
- inadequate-initial-reviews
- inexperienced-developers
- over-reliance-on-utility-classes
- poor-encapsulation
- tangled-cross-cutting-concerns
- unproductive-meetings
- reimplemented-standard-functionality
layout: solution
---

## Description

A lightweight design review is a short discussion of how a change will be approached, held before the change is built, based on a written sketch of one page or less. It fills a gap that most teams have without noticing: code review examines a decision already made and expensive to reverse, while formal architecture review is too heavy to invoke for the ordinary changes where most design decisions actually get made. The consequence is that the majority of a legacy system's design accumulates through individual choices that nobody discussed. The mechanism is cheap because a design is cheap to change — moving a boundary in a sketch costs minutes, and moving it in three weeks of implemented code costs a negotiation. Its secondary effect matters as much: it is one of the few settings in which design reasoning is made explicit and can therefore be learned.

## How to Apply ◆

> In legacy work the most consequential design decision is usually where to put the new code relative to the old — and that decision is made in the first hour, alone, and never discussed.

- Define a **trigger** so it is clear when a review is expected: a change beyond a certain size, one that adds a new component or interface, one that touches more than one subsystem, or one that introduces a dependency. Without a trigger it happens for the changes people already feel uncertain about, which are not the risky ones.
- Require **a written sketch, kept to one page**: what the change needs to do, the approach proposed, what was considered and rejected, and what will be affected. The act of writing this catches a meaningful share of the problems before anyone else reads it.
- Keep the session **short — thirty minutes — and small**, two or three people including someone who knows the affected area. A large review turns into a design-by-committee session and produces compromise architectures.
- Focus on a **small set of questions**: does this fit how the system is already organized, what happens when the parts it depends on fail, what will it be like to change in two years, and is there a simpler approach. Anything more detailed belongs in code review.
- **Include the option of not building it this way.** The most valuable outcome of a design review is often the discovery that an existing mechanism already does most of this, and that outcome is only available before implementation.
- **Record the decision and the reasoning**, briefly, where it will be found later — ideally as an architecture decision record for anything consequential. The sketch plus the outcome is often the only design documentation the change will ever have.
- Use it deliberately as **teaching**. Less experienced developers presenting their approach and hearing the questions experienced reviewers ask is how design judgment transfers; it is essentially never taught explicitly otherwise.
- **Do not let it become a gate.** A review that blocks work while waiting for a senior reviewer's calendar will be bypassed, and correctly. Same-day or next-day turnaround is the requirement, and a stated fallback for when it cannot happen.
- **Keep it out of the trivial cases.** Applied to every change, the overhead becomes resented and the practice is abandoned along with the cases where it was valuable.

## Tradeoffs ⇄

> Reviewing the design before implementation catches problems while they are cheap, at the price of a delay before coding starts and a practice that easily becomes bureaucratic.

**Benefits:**

- Design problems are caught when changing them costs minutes rather than weeks, which is the entire economic argument.
- Duplicated mechanisms are avoided, because someone in the room usually knows that the system already does this somewhere.
- Pull requests get smaller and more reviewable, since the approach is settled and the code review can focus on correctness.
- Design reasoning becomes explicit and observable, which is how it spreads to developers who have not previously been taught it.
- Consistency across the system improves, because independent changes are checked against the existing organization rather than each inventing their own.

**Costs and Risks:**

- Implementation starts later, and for changes that would have been fine the delay is pure cost.
- The practice drifts toward a formal approval gate, at which point it becomes an obstacle and gets routed around by whoever is in a hurry.
- Design discussion can produce over-engineering, particularly when the reviewers are more experienced than the problem requires and enjoy the discussion.
- Reviews with too many participants converge on the design that offends nobody, which is frequently worse than either alternative.
- The sketch can become a heavyweight document if standards creep, and a one-page requirement needs active defense.

## How It Could Be

A team maintaining a document management system consistently produced pull requests of 800 to 2,000 lines that reviewers could only approve rather than assess, and roughly a quarter of them prompted significant rework after review. They introduced a trigger — anything expected to take more than three days, or touching more than one subsystem — requiring a one-page sketch and a thirty-minute discussion. The first sketch reviewed proposed a new background worker for thumbnail generation. Within ten minutes a colleague pointed out that the existing batch framework already handled scheduling, retries, and failure alerting, none of which the proposal had accounted for. The change went from an estimated two weeks to four days. Over the following six months the proportion of pull requests requiring significant rework after review dropped from about a quarter to under five percent.

The teaching effect turned out to be the more durable one. Two developers with backgrounds in procedural codebases had been consistently producing designs that placed logic in static helper classes, which reviewers then objected to at code review — after the code was written, which made the conversation adversarial. In design review the same objection arrived as a question about where the behavior belonged, before any code existed, and could be discussed as a genuine question. Within a few months both developers were asking that question themselves in their sketches. The team's reviewers noted that they were now explaining reasoning rather than requesting changes, which was a materially different conversation.
