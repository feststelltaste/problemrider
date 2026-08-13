---
title: No-Regret Moves
description: Identify the modernization steps that pay off under every plausible future, and do those first while the destination is still undecided.
category:
- Architecture
- Management
- Process
problems:
- modernization-roi-justification-failure
- modernization-strategy-paralysis
- difficulty-quantifying-benefits
- analysis-paralysis
- decision-paralysis
- system-stagnation
- delayed-decision-making
- inability-to-innovate
- accumulated-decision-debt
- second-system-effect
- increasing-brittleness
- technology-lock-in
- legacy-system-documentation-archaeology
- market-pressure
- technology-stack-fragmentation
- upgrade-blocked-by-customization
layout: solution
---

## Description

A no-regret move is a piece of work that is worth doing regardless of which strategic option the organization eventually chooses. It exists as a named practice because legacy modernization decisions are routinely blocked on a question that cannot yet be answered — replace or rewrite, cloud or on-premises, buy or build — and while that question is open, nothing happens. This is a false constraint. A substantial part of what any of those futures requires is common to all of them: knowing what the system does, having tests around it, separating the parts that are entangled, removing what nothing uses. That work can start immediately, needs no strategic decision, and improves the organization's position under every branch. It also usually makes the blocked decision easier, because the reason it cannot be answered is typically that nobody understands the system well enough.

*The framing of no-regret moves as a distinct class of decision comes from strategy under uncertainty, and appears as a pattern in the Cloud Native transformation community.*

## How to Apply ◆

> The paralysing question in legacy modernization is nearly always about the destination, and a surprising share of the journey is identical whichever destination is chosen.

- **Enumerate the strategic options honestly**, including doing nothing. Three or four is enough. The point is not to choose between them yet but to have something concrete to test candidate work against.
- **Test each candidate against every option**: would this still have been worth doing if we go this way? Work that survives all of them is a no-regret move. Work that survives most of them is a low-regret move and belongs in a second tier.
- **Look first at the recurring categories.** Characterization tests around the behavior being preserved, removal of code and features that nothing uses, breaking dependencies that make anything hard to move, documenting what the system actually does, and establishing measurement — these are needed for replacement, rewrite, encapsulation, and continued maintenance alike.
- **Include the measurement work explicitly.** Baselines, cost data, and usage instrumentation are pure no-regret: they are cheap, they are required to justify anything, and they are needed under every option including doing nothing.
- **Start immediately and separately from the strategic decision.** The whole value is that this work does not wait. Attaching it to a programme that requires approval reintroduces the block it was meant to bypass.
- **Report it as progress against the strategic question**, not as unrelated maintenance. Six months of no-regret work materially changes what the eventual decision costs, and framing it that way keeps it funded and keeps the decision alive.
- **Feed what you learn back into the options.** The most reliable effect of this work is that the strategic question becomes answerable — usually because the system turns out to be different from what everyone assumed. Re-test the options as evidence arrives.
- **Watch for pseudo-no-regret work.** Anything requiring commitment to a specific target technology, framework, or vendor is not a no-regret move regardless of how it is framed, and this is the most common way the concept gets abused to smuggle in a preferred direction.
- **Set a limit.** No-regret work cannot substitute for the strategic decision indefinitely. Agree in advance roughly how long it runs before the decision must be forced, or it becomes a comfortable way of never deciding.

## Tradeoffs ⇄

> Doing what pays off under every future breaks modernization paralysis and needs no approval of a destination, but it can also become a way of postponing the decision forever.

**Benefits:**

- Progress starts without the strategic decision, which is frequently the difference between a system being improved and being discussed for another two years.
- Every option becomes cheaper, so the work is not wasted whichever way the eventual decision goes.
- The decision itself usually becomes answerable, because the blocking uncertainty is normally ignorance about the system rather than genuine strategic ambiguity.
- The work is individually easy to justify, since each piece is defensible on its own terms without reference to a contested programme.
- Optionality is preserved. Untangling dependencies and removing dead code widen the set of futures that remain available.

**Costs and Risks:**

- It can become a substitute for deciding, letting an organization feel productive for years while the strategic question stays open.
- Genuinely no-regret work is a smaller category than it first appears, and the boundary is easy to blur in favour of what someone already wanted to do.
- Some of it turns out to be wasted anyway — tests around a module that is ultimately deleted, documentation of a system that is replaced wholesale.
- It produces no visible business outcome for an extended period, which makes it vulnerable in any funding review that asks what has been delivered.
- Doing the easy common work first can leave the hardest, most option-specific problems entirely untouched, so the remaining decision is no less daunting than before.

## How It Could Be

An insurer spent two years unable to decide between replacing its policy administration system with a package, rewriting it, or encapsulating and keeping it. Three consultancies had produced three recommendations. Meanwhile nothing changed. A new architect proposed setting the question aside for six months and testing candidate work against all three options. Four categories survived: characterization tests around the premium calculation, deleting product variants that usage data showed had not been sold since 2015, separating the policy data from the reporting extracts that read it directly, and instrumenting the system to establish maintenance cost and incident baselines. None of it needed the strategic decision. All of it was needed under every option.

The decision answered itself in month five. The dead-product cleanup removed 40 percent of the premium calculation's branching, and the characterization work established that the remaining rules were far more standard than anyone had believed — the complexity everyone had cited as the reason a package could not fit had largely been variants nobody sold. The package option, previously dismissed as unworkable, became the recommendation, and the business case was built on the measurement work done during those five months. The architect's later assessment was that the two years of deadlock had been caused not by a hard strategic choice but by nobody knowing enough about the system to make it, and that this is the normal case.
