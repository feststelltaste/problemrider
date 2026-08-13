---
title: Automated Code Migration
description: Express a repetitive code change as an executable recipe that rewrites
  the syntax tree, so a migration across thousands of call sites becomes reviewable
  and repeatable.
category:
- Code
- Dependencies
- Process
problems:
- dependency-version-conflicts
- obsolete-technologies
- technology-lock-in
- api-versioning-conflicts
- legacy-api-versioning-nightmare
- code-duplication
- copy-paste-programming
- inconsistent-naming-conventions
- mixed-coding-styles
- high-technical-debt
- large-estimates-for-small-changes
- maintenance-paralysis
- increasing-brittleness
- vendor-dependency-entrapment
- fear-of-breaking-changes
- inconsistent-execution
- maintenance-cost-increase
- monolithic-functions-and-classes
- over-reliance-on-utility-classes
- refactoring-avoidance
- technology-stack-fragmentation
- undefined-code-style-guidelines
layout: solution
related_solutions:
- slug: large-scale-refactoring
  similarity: 0.75
- slug: continuous-dependency-updates
  similarity: 0.7
- slug: code-review-process-reform
  similarity: 0.7
- slug: code-generation
  similarity: 0.7
- slug: static-analysis-and-linting
  similarity: 0.7
- slug: small-change-batches
  similarity: 0.7
---

## Description

Automated code migration expresses a repetitive source change as an executable recipe operating on the parsed syntax tree rather than on text, and applies it across an entire codebase. Tools of this kind — OpenRewrite for JVM languages, Rector for PHP, jscodeshift and ts-morph for JavaScript and TypeScript, and the refactoring engines built into most IDEs — understand types, imports, and scope, so they can rename a method across 4,000 call sites, replace a deprecated API with its successor, or migrate a framework's configuration idiom without the false matches that search-and-replace produces. The practice matters for legacy work because the dominant cost of a library or framework upgrade is rarely the upgrade itself. It is the mechanical adaptation of thousands of call sites to a changed API, which is too large to do by hand, too error-prone to do with regular expressions, and is therefore not done at all — which is how a codebase ends up five major versions behind.

This is a different thing from a dependency update bot. Renovate and Dependabot raise the version number and open the pull request; they do not touch your source, so the build then fails on every call site the new version changed. Automated code migration is what fixes those call sites. The two are complements: the bot surfaces that an upgrade is available, and the recipe makes it applicable.

## How to Apply ◆

> The upgrades that never happen are usually not the difficult ones; they are the ones that are individually trivial and repeated four thousand times.

- **Check for an existing recipe before writing one.** The major migrations — framework major versions, JUnit 4 to 5, Java language level upgrades, common library successions — are already published, and running someone else's tested recipe is a completely different proposition from authoring one.
- **Work on the syntax tree, not on text.** A regular expression cannot distinguish a method call from a string containing the same name, and the false positives in a large codebase will consume more time than the migration saves. This is the entire reason these tools exist.
- **Run it on one module first** and review the diff by hand, line by line. The first application is where you discover that the recipe does not handle a pattern your codebase uses, and discovering that on 40 files is very different from discovering it on 4,000.
- **Land the migration as its own change**, containing no behavior modification. A mechanical change of 4,000 lines is reviewable when the reviewer knows it is behavior-preserving and can spot-check; the same diff with one functional change hidden in it is not reviewable at all.
- **Verify with the test suite you have, and be honest about what it does not cover.** Where coverage is thin, characterization tests around the affected area are the prerequisite, and skipping this is how a mechanical migration silently changes behavior.
- **Write your own recipe when the change is repeated enough to justify it** — an internal API being retired, a logging convention being standardized, a deprecated utility being replaced. The threshold is lower than people assume, roughly a few hundred call sites.
- **Keep recipes in version control alongside the code** and re-run them periodically. A recipe that enforces a convention becomes a way of preventing the pattern from reappearing, not just of removing it once.
- **Handle the residue explicitly.** No recipe reaches 100 percent; there will be call sites using reflection, dynamic dispatch, or an idiom the recipe does not recognize. List them, fix them by hand, and do not let the unfinished remainder block landing the ninety-odd percent.
- **Combine with a quality ratchet** so the old pattern cannot return: once the migration lands, a rule that the retired API may not be reintroduced keeps the codebase from drifting back.

## Tradeoffs ⇄

> Recipe-based migration makes changes possible that are otherwise simply not attempted, but the tooling has a real learning cost and mechanical changes at scale carry their own risks.

**Benefits:**

- Migrations that are impractical by hand become routine, which directly addresses the reason legacy codebases fall many versions behind on their dependencies.
- The change is consistent everywhere, so the codebase does not end up with the new idiom in the files someone got to and the old one in the rest.
- It is far safer than search-and-replace, because the tool understands types and scope and does not match text that merely looks similar.
- Recipes are repeatable and shareable, so the same migration can be applied to other services and re-run to prevent regression.
- Estimates for large mechanical changes become credible, since the work is largely the recipe rather than the call sites.

**Costs and Risks:**

- The tooling has a genuine learning curve, and authoring a non-trivial recipe is a skill that takes time to acquire.
- Enormous mechanical diffs are difficult to review meaningfully, and reviewers tend to approve them on trust — which is reasonable only if the behavior-preservation claim is verified another way.
- Without adequate test coverage, a mechanical change can alter behavior invisibly, and legacy codebases are exactly where coverage is thin.
- Support is uneven across languages and ecosystems; some legacy stacks have no usable tooling of this kind at all.
- Large migrations create merge conflicts with everything in flight, so they need coordinating with the team's other work rather than being landed opportunistically.

## How It Could Be

A team maintaining a Java platform was three major versions behind on their framework and had estimated the upgrade at four months, almost entirely for adapting roughly 5,800 call sites to changed APIs. The estimate had been rejected twice. They ran a published migration recipe against one of their eleven modules: 94 percent of the call sites were transformed automatically in under a minute, and the remaining 6 percent were listed. Reviewing that module's diff by hand took a day and found two patterns the recipe handled in a way they did not want, which they overrode. Applying it across the remaining ten modules took a week, the manual residue took a further week, and the whole upgrade landed in under a month against a four-month estimate. The decisive change was not the tool's speed but that the work had become reviewable — the framework diff was mechanical and separate, and the eleven behavioral adaptations were a small change that could actually be read.

The team subsequently wrote two recipes of their own. The first replaced an internal date utility whose parameter order had been causing recurring defects, across about 900 call sites. The second enforced their logging convention — correlation identifier present, no string concatenation in log calls — and was run in the pipeline rather than once, so the pattern could not reappear. The second one turned out to be the more valuable: their previous attempt to establish that convention had been a written guideline, followed for about two months, after which compliance decayed to roughly where it had started.
