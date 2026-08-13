---
title: Quality Ratchet
description: Require that quality measures never get worse than they are today, instead of setting absolute thresholds a legacy codebase can never meet.
category:
- Code
- Process
- Testing
problems:
- high-technical-debt
- quality-degradation
- increasing-brittleness
- increased-technical-shortcuts
- poor-test-coverage
- accumulation-of-workarounds
- mixed-coding-styles
- undefined-code-style-guidelines
- quality-compromises
- inconsistent-execution
- copy-paste-programming
- refactoring-avoidance
- maintenance-cost-increase
- brittle-codebase
- convenience-driven-development
layout: solution
---

## Description

A quality ratchet enforces that a measure may not get worse than its current value, rather than requiring it to reach an absolute standard. Test coverage may not fall below what it is today; the number of static analysis warnings may not increase; the count of dependencies past end of support may not rise. This solves the specific reason quality gates fail on legacy codebases. An absolute threshold — eighty percent coverage, zero warnings — is unreachable on a system with twenty years of history, so it is either not adopted, or adopted and immediately suspended, or adopted with so many exemptions that it measures nothing. A ratchet is achievable from day one regardless of the starting point, because it asks only that today's change not make things worse. Over time it converts every improvement into a new floor, so progress accumulates instead of eroding.

## How to Apply ◆

> The reason legacy quality initiatives fail is almost never that the team disagrees with the standard; it is that the standard is unreachable, and unreachable standards are ignored.

- **Set the initial threshold at the current measured value**, whatever it is, and record it. A ratchet starting at an aspirational number is an absolute gate wearing different clothes and will fail the same way.
- **Ratchet on the measures that matter and can be measured reliably**: test coverage, static analysis findings, build duration, dependency currency, and the count of files exceeding a complexity threshold. Two or three ratchets are enough; a dozen produces friction without a corresponding benefit.
- **Apply it to changed code where the measure allows.** Coverage on modified lines is a far more useful ratchet than coverage across the whole codebase, since it improves the areas actually being worked on rather than encouraging tests for dormant code.
- **Enforce it in the pipeline**, not by convention. A ratchet that depends on people remembering will erode, quietly, and the erosion is invisible until someone measures.
- **Update the floor automatically when a change improves the measure.** This is the mechanism: an improvement made for its own reasons becomes permanent without anyone deciding to protect it.
- **Provide an explicit override with a name and a reason attached**, and review the overrides periodically. A ratchet with no escape hatch will be circumvented or disabled at the first genuine emergency; one whose overrides are recorded stays honest and shows where the friction is.
- **Do not ratchet on measures that are easy to game.** Line coverage invites tests that execute code without asserting anything. Where a measure can be satisfied without the underlying improvement, it will be, and the ratchet then enforces a ritual.
- **Introduce them one at a time**, with a period of reporting before enforcing. A ratchet that starts failing builds on the day it is introduced will be blamed for whatever else goes wrong that week.
- **Report the floor's movement** quarterly. The trend — coverage floor rising from 31 to 44 percent over a year without any dedicated testing project — is the evidence that the mechanism is working, and it is unusually persuasive because nobody had to be persuaded to produce it.

## Tradeoffs ⇄

> Ratchets are achievable on any codebase and make improvement permanent, but they only prevent decline — they do not by themselves produce progress, and they add friction at inconvenient moments.

**Benefits:**

- It is adoptable from day one on any codebase, however bad, which is exactly what absolute thresholds are not.
- Improvements become permanent. Without a ratchet, gains made during a focused effort erode over the following year and the effort has to be repeated.
- Degradation stops being invisible. Legacy quality decays gradually, each individual change being defensible, and the ratchet is what makes the aggregate visible.
- The floor's movement over time is an evidence-based progress measure that costs nothing extra to produce.
- It applies pressure at the point of change, where the developer has the context, rather than through periodic cleanup campaigns.

**Costs and Risks:**

- A ratchet prevents decline but does not drive improvement. A team can sit at its initial floor indefinitely and be fully compliant.
- It adds friction exactly when people are in a hurry, which is when overrides get used and when the practice is most likely to be abandoned.
- Gameable measures get gamed, and a ratchet on a weak proxy enforces the appearance of quality rather than quality.
- Whole-codebase measures can be satisfied in unhelpful ways, such as adding tests to trivial dormant code to offset a decline in an important area.
- Legitimate work sometimes makes a measure worse — deleting well-tested code can lower overall coverage — and a ratchet without judgement will block improvements.

## How It Could Be

A team had attempted a coverage gate twice. The first set 70 percent, against an actual figure of 23 percent, and was abandoned within a week. The second set 30 percent, which was met by adding tests to a utility package nobody used, after which coverage in the areas that mattered continued to fall. The third attempt was a ratchet: coverage on modified lines could not fall below the current value for that file, and overall coverage could not fall below 23 percent. It failed no builds in the first month, because it asked nothing new of anyone — only that changes not make their file worse. Over fourteen months overall coverage rose to 44 percent, without any dedicated testing project, entirely as a byproduct of ordinary work in the areas that were being worked on.

The static analysis ratchet produced a different lesson. The codebase had roughly 4,100 warnings, and the team's previous attempt to reduce them had stalled after a week of tedium. A ratchet on the total meant a change adding a warning had to remove one, which developers generally satisfied by fixing something adjacent to what they were already touching. The count fell to about 2,600 over a year. The override log turned out to be the more valuable artifact: 31 overrides, of which 19 were the same warning class in the same subsystem — a code generation step producing output that the analyser objected to and that nobody could change. That was fixed by configuring an exclusion for generated code, which had been quietly poisoning the team's relationship with static analysis for years.
