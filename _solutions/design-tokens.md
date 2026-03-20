---
title: Design Tokens and Theming
description: Encoding visual design decisions platform-agnostically for theming and cross-platform consistency
category:
- Architecture
- Code
quality_tactics_url: https://qualitytactics.de/en/usability/design-tokens/
problems:
- inconsistent-codebase
- inconsistent-behavior
- poor-user-experience-ux-design
- high-maintenance-costs
- maintenance-overhead
- technology-stack-fragmentation
- difficult-code-reuse
layout: solution
---

## How to Apply ◆

> Legacy systems often have visual design decisions scattered throughout the codebase as hardcoded values. Design tokens centralize these decisions into a single source of truth that can be applied across platforms.

- Extract hardcoded color values, font sizes, spacing units, and border radii from legacy CSS, stylesheets, and inline styles into a centralized token file using a format such as JSON or YAML.
- Define a token naming hierarchy that separates primitive tokens (raw values like specific hex colors) from semantic tokens (purpose-based names like "color-error" or "spacing-section") to make the system maintainable and meaningful.
- Implement a build pipeline that transforms design tokens into platform-specific formats: CSS custom properties for web, resource files for mobile, and constants for desktop applications.
- Apply tokens incrementally during legacy maintenance. When touching a file that contains hardcoded visual values, replace them with token references rather than attempting a complete system-wide replacement at once.
- Support theming by mapping semantic tokens to different value sets for light mode, dark mode, high-contrast mode, and brand-specific variants.
- Document the token system with visual examples so developers can look up the correct token to use rather than guessing or introducing new hardcoded values.

## Tradeoffs ⇄

> Design tokens create a powerful abstraction for visual consistency but require discipline and tooling to manage effectively.

**Benefits:**

- Enables system-wide visual changes through a single token update rather than searching through thousands of lines of legacy CSS for hardcoded values.
- Supports theming and accessibility modes without duplicating stylesheets or maintaining parallel visual codebases.
- Ensures visual consistency across different technology stacks within the legacy system, even when different modules use different frontend frameworks.
- Reduces maintenance overhead by eliminating redundant color and spacing definitions scattered across the codebase.

**Costs and Risks:**

- Setting up the token infrastructure and build pipeline requires initial investment before any visual benefit is realized.
- Migrating a large legacy codebase from hardcoded values to tokens is tedious and must be done incrementally to avoid regressions.
- Over-engineering the token hierarchy with too many abstraction layers can make the system harder to understand than the hardcoded values it replaced.
- Teams must adopt the discipline of using tokens for all new work, or the system will gradually return to inconsistency.

## Examples

> Legacy systems with long histories accumulate dozens of slightly different shades of the same color and inconsistent spacing, creating visual noise that undermines professionalism.

A legacy enterprise application built over fifteen years has its UI spread across three frontend technologies: JSP pages, a React-based module added five years ago, and an Angular dashboard added recently. Each uses its own color definitions, resulting in three different shades of the company's primary blue and inconsistent spacing throughout. The team extracts all color and spacing values from all three codebases into a shared design token file. A build step generates CSS custom properties for the JSP pages, a JavaScript module for React, and a TypeScript constants file for Angular. After three months of incremental migration during routine maintenance, the application achieves visual consistency across all modules for the first time, and implementing a dark mode becomes a matter of creating an alternate token value set rather than rewriting hundreds of CSS rules.
