---
title: Team Working Agreements
description: "Make the team's unwritten expectations explicit \u2014 how decisions\
  \ are made, how disagreement is handled, what gets shared, what a meeting is for\
  \ \u2014 and revise them when they fail."
category:
- Team
- Culture
- Process
problems:
- poor-teamwork
- team-dysfunction
- unclear-sharing-expectations
- nitpicking-culture
- style-arguments-in-code-reviews
- bikeshedding
- unproductive-meetings
- individual-recognition-culture
- inconsistent-execution
- poor-communication
- team-confusion
- language-barriers
- fear-of-conflict
- conflicting-reviewer-opinions
- author-frustration
- avoidance-behaviors
- blame-culture
- communication-risk-within-project
- inadequate-mentoring-structure
- inconsistent-onboarding-experience
- knowledge-sharing-breakdown
- mentor-burnout
- micromanagement-culture
- new-hire-frustration
- perfectionist-review-culture
- rapid-team-growth
- reduced-review-participation
- review-process-avoidance
- review-process-breakdown
- team-members-not-engaged-in-review-process
- unclear-documentation-ownership
- communication-breakdown
- insufficient-code-review
- merge-conflicts
- power-struggles
- team-coordination-issues
- extended-review-cycles
- inadequate-code-reviews
- inadequate-initial-reviews
- lack-of-ownership-and-accountability
layout: solution
related_solutions:
- slug: code-review-guidelines
  similarity: 0.7
- slug: psychological-safety-practices
  similarity: 0.7
- slug: written-first-communication
  similarity: 0.65
- slug: decision-rights-and-escalation
  similarity: 0.65
- slug: team-retrospectives
  similarity: 0.65
- slug: code-conventions
  similarity: 0.6
---

## Description

A working agreement is a short, explicit statement of how a team operates: how decisions get made, what response times people can expect from each other, how disagreements are resolved, what is shared and where, and what behavior is out of bounds. Every team already has these norms; the question is only whether they are stated or inferred. Unstated norms fail in predictable ways — they are learned by newcomers through mistakes, they differ between members who each believe theirs is obvious, and they cannot be appealed to when someone violates them, because nobody can point at anything. Writing them down does not make a team functional, and a team with real conflicts will not resolve them with a document. What it does is remove an entire class of friction that comes from people operating on incompatible assumptions while believing they share them.

## How to Apply ◆

> Long-lived maintenance teams accumulate unwritten conventions over years, and the resulting expectations are usually invisible to everyone except the newest member, who keeps violating them.

- **Write the agreement together**, in one session of an hour or two. An agreement handed down by a lead is a policy, and policies are complied with rather than owned. The discussion is at least as valuable as the artifact, because it surfaces the assumptions members did not know they disagreed about.
- Start from **real friction, not from a template**. Ask what has gone wrong recently and what each person wishes were different. Agreements assembled from generic best practices cover situations the team does not have and omit the ones it does.
- Cover the areas that actually generate conflict: **how decisions are made and by whom, expected response times for reviews and questions, what gets written down and where, how disagreements are escalated, meeting norms, and availability expectations** across time zones or working hours.
- Be **specific enough to be checkable**. "We communicate openly" is unfalsifiable and therefore useless. "Reviews are picked up within one working day; if you cannot, say so in the channel" can be observed, appealed to, and violated.
- Address **disagreement explicitly**. State how the team handles it — who decides, within what time, and what happens to the dissenting position. Teams that avoid conflict do so partly because they have no agreed procedure for it, so every disagreement threatens to become personal.
- Include **norms for review behavior** if reviews are a friction point: what blocks a merge, what is a suggestion, and that mechanically checkable style is the tool's job. This is where nitpicking and bikeshedding are most cheaply addressed, because both are usually the absence of an agreed scope rather than a character trait.
- For **distributed or multilingual teams**, state the language and channel conventions plainly: which language is used for code, comments, tickets, and meetings; that asking someone to repeat or rephrase is expected rather than rude; and that written summaries follow verbal decisions. This removes a large and rarely discussed source of exclusion.
- **Keep it to one page** and put it where new members meet it in their first week. An agreement longer than a page is not read, and an agreement that is not read is worse than none because people believe it is doing something.
- **Revise it when it fails.** Every recurring friction is a gap in the agreement or a rule that is not working. A standing retrospective question — "did we follow our agreement, and where did it not help?" — keeps it alive; without that, it becomes an artifact from an onboarding folder.

## Tradeoffs ⇄

> Explicit agreements resolve the friction caused by mismatched assumptions, but they cannot resolve genuine conflicts of interest and can become a bureaucratic weapon if written poorly.

**Benefits:**

- New members become effective faster, because the norms they would otherwise learn by transgressing are stated up front.
- Recurring low-grade conflicts — review nitpicking, response-time expectations, who decides what — get settled once instead of being renegotiated in every instance.
- Problematic behavior can be addressed by referring to a shared agreement rather than by one person confronting another, which lowers the personal cost of raising it dramatically.
- Meetings improve measurably when their purpose and norms are stated, since most unproductive meetings are unproductive because nobody agreed what they were for.
- Distributed and multilingual teams gain the most, as these are exactly the settings where unstated norms diverge most widely and are hardest to infer.

**Costs and Risks:**

- Agreements decay into unread documents unless they are revisited. A team with a two-year-old agreement that nobody has looked at has the same problem it started with, plus a false sense of having addressed it.
- They cannot fix a team with genuine trust problems, an abusive member, or structural conflicts of interest, and attempting to use them that way delays the real intervention.
- Written rules can be weaponized. A member who invokes the agreement selectively to win arguments turns a coordination tool into a compliance instrument.
- The initial session surfaces disagreements that were previously suppressed, which is the point but is uncomfortable and requires enough psychological safety to be productive rather than damaging.
- Overly detailed agreements become bureaucracy and are resented, particularly by experienced members who read them as a statement that they are not trusted to behave reasonably.

## How It Could Be

A distributed team of seven maintaining a European logistics platform spanned four countries and three time zones. Code reviews sat for three to four days, meetings ran in English with two members contributing almost nothing, and a recurring argument about commit message format had consumed part of every retrospective for months. In a two-hour session they wrote a one-page agreement: reviews picked up within one working day or declined explicitly in the channel; decisions made in meetings summarized in writing within the hour; commit format enforced by a hook rather than by people; and an explicit statement that asking for a rephrase in a meeting was expected. Review latency fell to under a day within three weeks. The two quiet members began contributing in writing after meetings, which the team had not anticipated but which turned out to be their preferred mode, and two significant design objections surfaced that way in the first month.

A different team used their agreement to end a long-running review conflict. Two senior developers had incompatible views on error handling, and every pull request touching that area became a standoff that the author had to broker. The agreement added two lines: mechanically checkable rules belong in the linter, and where two reviewers disagree substantively, the module owner decides within one working day and the dissenting position is recorded in the pull request. The first invocation was uncomfortable. The fourth was routine, and the recorded dissents later became the input for an architecture decision record that settled the underlying question properly.
