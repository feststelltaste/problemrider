---
title: Work in Progress Limits
description: Cap how many items the team may have started but not finished, so that work is completed rather than accumulated.
category:
- Process
- Team
- Management
problems:
- context-switching-overhead
- priority-thrashing
- uneven-work-flow
- uneven-workload-distribution
- work-blocking
- extended-cycle-times
- delayed-project-timelines
- constant-firefighting
- maintenance-bottlenecks
- reduced-team-productivity
- mental-fatigue
- cascade-delays
- incomplete-projects
- delayed-issue-resolution
- avoidance-behaviors
- competing-priorities
- developer-frustration-and-burnout
- extended-review-cycles
- increased-stress-and-burnout
- increased-time-to-market
- procrastination-on-complex-tasks
- reduced-individual-productivity
- reduced-predictability
- reduced-review-participation
- resource-waste
- review-bottlenecks
- review-process-breakdown
- rushed-approvals
- team-demoralization
- team-members-not-engaged-in-review-process
layout: solution
---

## Description

A work in progress limit is an agreed maximum number of items that may be in an unfinished state at once — per person, per workflow stage, or per team. It is the most direct available intervention against the pattern where everything is started, nothing is finished, and everyone is busy. The mechanism is uncomfortable by design: when the limit is reached, no new work may be pulled, so the team must either finish something or resolve whatever is blocking it. That forced confrontation with blockers is the actual value; the limit itself is only the trigger. In legacy maintenance the effect is pronounced, because unfinished work in a fragile system is not merely idle — half-migrated data structures, partially applied refactorings, and abandoned branches actively increase the risk and cost of everything else the team touches.

## How to Apply ◆

> Legacy teams are pulled in many directions at once — production incidents, migration work, feature requests, and support escalations — so limits have to account for interrupt-driven work rather than pretending it does not exist.

- Make current work in progress **visible before limiting it**. Put every started-but-unfinished item on one board, including the invisible ones: open branches, waiting reviews, half-finished investigations, and support tickets someone is quietly carrying. Teams are routinely shocked by the count, and the visibility alone changes behavior before any limit is set.
- Set the first limit **just below the current average**, not at a theoretically ideal number. If the team currently carries eighteen items, start at fourteen. A limit that is immediately violated by reality is discarded within a week; a limit that bites gently is kept.
- Apply the limit **per workflow stage, not just overall**, so that bottlenecks become visible. A cap on "in review" is usually the highest-value first limit for teams with review bottlenecks, because it forces reviewing to compete with starting new work rather than always losing to it.
- Define the **stop-the-line rule** explicitly: when a stage is at its limit, people who would have pulled new work instead help finish or unblock existing work. Without this rule the limit merely creates idle time and gets abandoned as wasteful.
- Reserve **explicit capacity for interrupt work** rather than letting it break the limit. A dedicated slot — one person on support rotation, or two reserved WIP slots for incidents — keeps unplanned work from silently expanding the limit to infinity, and makes the true cost of the interrupt load visible in planning.
- Track **blocked items separately** and review them daily. The value of a WIP limit appears only if blockers get escalated; if blocked items simply sit inside the limit, the team is capped but not flowing. A blocker older than one day should have a named owner and an escalation path.
- Measure **cycle time and completion rate**, not utilization. The expected outcome is that people are individually less busy and the team finishes more, which looks like a regression on any utilization metric and needs to be explained to management in advance.
- Revisit the limit every few weeks and **lower it while flow continues to improve**. The limit is a tuning parameter, not a policy; when lowering it stops improving cycle time or starts causing genuine idleness, the previous value was right.

## Tradeoffs ⇄

> Limiting work in progress improves throughput and predictability but requires accepting visible idleness and saying no to work that has already been promised.

**Benefits:**

- Cycle time drops, often dramatically, because items stop queueing behind each other while nominally in progress. This is arithmetic rather than motivation: less concurrent work means less time waiting per item.
- Context switching falls, which recovers a large amount of effective capacity in legacy work where reloading the context of a complex module is expensive.
- Blockers surface immediately and get escalated, instead of being quietly absorbed by starting something else.
- Bottlenecks become visible and locatable. A stage that is constantly at its limit identifies exactly where the team's capacity constraint lies — usually review or testing.
- Half-finished work in the codebase decreases, reducing the risk that partially applied changes interact badly with each other.

**Costs and Risks:**

- The limit is only as strong as management's willingness to honor it. If new work is pushed in regardless, the team is left with the ceremony and none of the benefit, and trust in the practice is spent.
- Visible idleness is politically difficult. An engineer with nothing to pull looks like waste to an observer who measures busyness, and this needs to be actively defended.
- Poorly chosen limits cause real stalls, particularly in small teams where one blocked item can consume a large share of the cap.
- Interrupt-heavy environments can render limits meaningless unless interrupt capacity is explicitly reserved, and reserving it means admitting how much capacity the interrupt load actually consumes.
- The practice exposes uncomfortable facts — that the team is the bottleneck at a particular stage, or that a dependency team never responds — which some organizations would rather not have documented.

## How It Could Be

A five-person team maintaining a hospital scheduling system tracked their in-flight work for the first time and counted twenty-three started items: six branches over a month old, five reviews waiting on someone, four investigations with no clear next step, and eight tickets in active development. They set a team limit of ten and a review-stage limit of three. In the first week almost nothing new started; the team spent four days closing out stale branches, two of which were abandoned outright and one of which turned out to conflict with work someone else had just completed. Over the following quarter median cycle time fell from nineteen days to six, and the number of items completed per month rose by roughly forty percent despite no change in team size.

A platform team suffering from constant firefighting used a different variant: two of their eight WIP slots were permanently reserved for incidents and urgent support, and planned work could never exceed six. The reservation made the interrupt load measurable for the first time — it consistently consumed both slots and often demanded a third. That data, presented as "a quarter of our capacity goes to unplanned work in this subsystem," was what finally justified a dedicated stabilization effort on the two modules generating most of the incidents. Six months later the interrupt load fit comfortably in one slot, and the freed capacity went back to planned work.
