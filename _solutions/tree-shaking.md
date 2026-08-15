---
title: Tree Shaking
description: Eliminating unused code while building
category:
- Code
- Performance
problems:
- high-client-side-resource-consumption
- slow-application-performance
- uncontrolled-codebase-growth
- feature-bloat
- inefficient-frontend-code
- gradual-performance-degradation
layout: solution
related_solutions:
- slug: code-splitting
  similarity: 0.8
- slug: image-and-asset-optimization
  similarity: 0.8
- slug: strategic-code-deletion
  similarity: 0.75
- slug: lazy-loading
  similarity: 0.75
- slug: lazy-evaluation
  similarity: 0.75
- slug: compression
  similarity: 0.7
---

## Description

Tree shaking is a build-time optimization performed by module bundlers that statically analyzes a codebase's import and export graph and strips out any code that is never actually referenced, so the shipped bundle contains only what the application uses rather than everything a dependency happens to provide. It relies on the static, analyzable structure of ES modules to determine reachability at build time, which is why legacy CommonJS code — with its dynamic `require()` calls that cannot always be resolved without running the program — frequently defeats it and needs conversion before the optimization can take effect. In legacy frontend codebases this matters because bundle size tends to grow monotonically over years: whole utility libraries get imported for a handful of functions, disabled features stay bundled because nobody removed their imports, and barrel files re-export everything indiscriminately, none of which shows up as a functional bug but all of which quietly taxes every page load. Tree shaking addresses this without requiring anyone to manually hunt down and delete dead code path by path; instead, the build process itself removes what static analysis proves is unreachable, given enough structural cleanup (ES modules, side-effect-free package markers, avoidance of overly broad barrel exports) to make the analysis effective. Because it operates entirely within the build pipeline, it can be adopted incrementally alongside other modernization work, delivering measurable page-load improvements without requiring a rewrite of the application's runtime behavior.

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
