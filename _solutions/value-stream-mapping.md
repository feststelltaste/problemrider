---
title: Value Stream Mapping
description: Map every step from request to production with its working time and waiting time, so that the waiting — which is nearly always the majority — becomes visible and addressable.
category:
- Process
- Management
problems:
- inefficient-processes
- increased-manual-work
- wasted-development-effort
- extended-cycle-times
- long-release-cycles
- delayed-value-delivery
- operational-overhead
- increased-time-to-market
- approval-dependencies
- work-blocking
- immature-delivery-strategy
- resource-waste
- delayed-project-timelines
- budget-overruns
- cascade-delays
- constantly-shifting-deadlines
- context-switching-overhead
- delayed-issue-resolution
- maintenance-cost-increase
- missed-deadlines
- organizational-structure-mismatch
- project-resource-constraints
- reduced-team-productivity
- team-coordination-issues
- unrealistic-deadlines
- bottleneck-formation
- competing-priorities
- extended-review-cycles
- increased-stress-and-burnout
- mental-fatigue
- planning-credibility-issues
- planning-dysfunction
- priority-thrashing
- process-design-flaws
- team-demoralization
- uneven-work-flow
- uneven-workload-distribution
layout: solution
---

## Description

Value stream mapping records every step a piece of work passes through from request to production, and for each step captures two numbers: how long the work is actively being worked on, and how long it waits. The ratio between them is the finding. Teams consistently estimate that most of their cycle time is development effort; the map consistently shows that eighty to ninety-five percent is waiting — for review, for approval, for a test environment, for a release window, for another team. This matters because improvement effort is usually aimed at the working time, where the potential gains are small and the disruption is high, when the waiting is where the time actually goes. In legacy organizations the effect is pronounced, since decades of accumulated process controls, handoffs, and sign-offs each added a queue that nobody has since revisited.

## How to Apply ◆

> The steps that consume the most time in a legacy delivery process are usually the ones nobody thinks of as steps: the wait for the shared test environment, the change advisory board that meets on Thursdays, the release window every third week.

- **Map one real, recent item end to end** rather than the process as documented. Pick a specific change that reached production and reconstruct what happened to it, with dates. The documented process and the actual process diverge, and the divergence is often where the delay lives.
- Include **the full span from request to production**, not just the development portion. Most of the waste sits before development starts and after it finishes, so a map that begins at "ticket assigned" and ends at "code merged" will find nothing.
- For each step record **process time and elapsed time separately**. A code review with fifteen minutes of process time and three days of elapsed time is a queueing problem, not a review problem, and the two have completely different fixes.
- **Record the handoffs** and who is on each side. Every handoff is a queue, a context loss, and a potential rework loop. Counting them is often more revealing than the times themselves.
- Note **rework loops**: how often work goes backward, and why. Work returning from testing to development twice on average is a defect-prevention problem masquerading as a delivery problem.
- **Do the mapping with the people who do the work**, in one room, on a wall. The map is not the deliverable — the shared realization is. A map produced by a consultant and presented to the team persuades nobody.
- **Attack the largest wait first**, not the most annoying step. This is counterintuitive; the steps people complain about are usually short and irritating, while the multi-day waits are so normal that nobody mentions them.
- Distinguish **waits that protect something from waits that protect nothing**. A change approval board that has rejected two changes in three years is a queue with no yield; a security review that catches real issues is a queue worth keeping and worth making faster.
- **Re-map after changes** to verify the improvement moved the total, not just one step. Local optimizations frequently push the queue somewhere else, and only the end-to-end number shows whether anything actually improved.

## Tradeoffs ⇄

> Mapping is cheap and frequently produces the single highest-value insight available to a delivery organization, but the improvements it identifies often lie outside the team's authority.

**Benefits:**

- Waiting time becomes visible, and since it usually dominates cycle time, this is where the largest available improvements are.
- Improvement effort gets directed by evidence rather than by which step is most irritating to the people who complain most.
- Handoffs and approval steps that have outlived their purpose are identified concretely, with the cost of each attached.
- The map is a persuasive artifact for management, because a diagram showing three days of work and thirty-one days of waiting makes an argument that no verbal complaint can.
- It creates a shared understanding across roles who each see only their own segment, which frequently resolves long-standing mutual blame between development, testing, and operations.

**Costs and Risks:**

- The workshop consumes several hours of many people's time, and produces no output until something changes as a result.
- The biggest waits are often owned by other departments — change boards, security, procurement — so the team can measure the problem without being able to fix it.
- A single mapped item may not be representative. Mapping one unusually smooth or unusually troubled change leads to conclusions that do not generalize.
- Maps become obsolete as the process changes, and a stale map used for decisions is worse than none.
- Removing a control step that appears wasteful can remove a protection whose value was invisible precisely because it was working.

## How It Could Be

A team supporting an insurance claims platform believed their delivery was slow because the codebase was difficult. They mapped one recent change from request to production: 4.5 days of actual work spread across 47 calendar days. The map showed 9 days waiting for business sign-off on the requirement, 6 days waiting for a shared integration environment, 11 days waiting for the fortnightly change advisory board, and 14 days waiting for the monthly release window. The code itself was never the constraint. Over the next two quarters they addressed the two largest queues — moving to containerized per-branch environments and negotiating a fast path through the change board for pre-approved low-risk change types — and median lead time fell from 47 days to 16 without anyone touching the codebase.

A second organization used the map to defend a control rather than to remove it. Mapping showed that their security review added 4 days of waiting per change, and there was pressure to eliminate it. The same exercise recorded that the review had caught 14 genuine issues in 18 months, three of which were serious. Rather than removing the step, they moved it earlier — reviewing designs rather than finished changes — and added automated checks for the recurring categories. The wait dropped to under a day, and the catch rate stayed the same. The distinction between a queue with yield and a queue without it was what made the discussion productive instead of positional.
