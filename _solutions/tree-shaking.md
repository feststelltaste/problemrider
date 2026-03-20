---
title: Tree Shaking
description: Eliminating unused code while building
category:
- Code
- Performance
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/tree-shaking
problems:
- high-client-side-resource-consumption
- slow-application-performance
- uncontrolled-codebase-growth
- feature-bloat
- inefficient-frontend-code
- gradual-performance-degradation
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Configure module bundlers (Webpack, Rollup, esbuild) to perform dead code elimination during the build process
- Convert legacy CommonJS modules to ES modules to enable static analysis of import/export dependencies
- Mark packages and modules as side-effect-free in package.json to allow more aggressive tree shaking
- Audit bundle contents using visualization tools (webpack-bundle-analyzer) to identify large unused dependencies
- Replace monolithic utility libraries with modular alternatives that support per-function imports
- Refactor barrel files (index.js re-exports) that prevent tree shaking from identifying unused exports
- Add bundle size checks to the CI pipeline to prevent regression

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces JavaScript bundle size, directly improving page load times
- Removes dead code that obscures the actually used codebase
- Decreases client-side memory consumption and parsing time
- Can be implemented incrementally alongside other modernization efforts

**Costs and Risks:**
- Legacy code with side effects in module initialization can break when tree-shaken
- Dynamic imports and require() calls cannot be statically analyzed and may be incorrectly removed
- Requires migration from CommonJS to ES modules, which can be disruptive in large codebases
- Build configuration complexity increases with tree shaking rules and exceptions
- Some libraries are not tree-shakeable, requiring replacement or manual exclusion

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A legacy single-page application built with Angular.js had accumulated a 4.2 MB JavaScript bundle over five years. Bundle analysis revealed that a full lodash import contributed 600 KB despite using only 12 functions, and several feature modules that had been disabled in configuration were still included. The team switched to lodash-es with per-function imports, converted key modules to ES module syntax, and enabled Webpack's tree shaking. The production bundle dropped to 1.8 MB, cutting initial page load time from 6 seconds to 2.5 seconds on typical connections.
