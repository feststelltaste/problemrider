---
title: Dynamic Connascence
description: A name, file path, or string literal encodes a runtime reference between
  a producer and a consumer that the compiler and automated rename or move refactoring
  cannot see.
category:
- Architecture
- Code
related_problems:
- slug: poor-naming-conventions
  similarity: 0.5
- slug: hidden-dependencies
  similarity: 0.5
solutions:
- dependency-injection
- dependency-injection-container
- static-analysis-and-linting
- characterization-tests
- change-impact-analysis
- consumer-driven-contracts
- typed-schema-extraction
- explicit-extension-points
- automated-code-migration
- schema-registry
layout: problem
---

## Description

Dynamic connascence occurs when two parts of a system must agree on a name, but that agreement is enforced only at runtime rather than by the compiler or an IDE's static analysis. A reflective class lookup built from a computed string, a UI component paired with its template through an identical file path, or a persisted JSON field that a deserializer expects by exact key are all instances of this. Because the reference exists only as a string or a naming convention, an ordinary symbol rename or file move slips through untouched: automated refactoring tools, compilers, and simple text search all fail to recognize it as the same reference. The failure then surfaces far away from the change, often only when the affected code path actually executes, and frequently as a silent fallback or a confusing runtime error rather than a build failure. The concept comes from Meilir Page-Jones' connascence taxonomy, which distinguishes static connascence, detectable by reading the source, from dynamic connascence, detectable only by running the program.

## Indicators ⟡

- A reflective or dynamic lookup (`Class.forName`, `loadClass`, dynamic `import()`, string-based service lookup) constructs a name from a string that is built or configured elsewhere.
- Two files are paired only because they share the same relative path and base name, with no explicit reference between them.
- Grepping for a class, field, or file name misses real usages because some are constructed at runtime through prefixes, suffixes, or concatenation.
- A rename that an IDE's "Find Usages" or automated refactor reported as fully covered still breaks something once deployed.
- Persisted data such as database rows, configuration files, or exported bookmarks contains class names, keys, or IDs that must match code exactly.

## Symptoms ▲

- [Hidden Dependencies](hidden-dependencies.md)
<br/>  The link between producer and consumer exists only as a shared string, so the dependency is invisible in the code's interfaces or structure.
- [Regression Bugs](regression-bugs.md)
<br/>  A rename or move that passes every automated check still breaks the runtime lookup, producing a regression no test suite anticipated.
- [Debugging Difficulties](debugging-difficulties.md)
<br/>  When the failure surfaces, there is no stack trace pointing back to the rename; developers must reconstruct the naming contract by hand.
- [Fear of Breaking Changes](fear-of-breaking-changes.md)
<br/>  Once a team has been burned by an invisible naming contract, they become reluctant to rename or move anything near reflective or convention-based code.

## Causes ▼

- [Stringly Typed Code](stringly-typed-code.md)
<br/>  Domain values represented as raw strings are exactly what dynamic lookups, prefixes, and query paths are built from, turning a value problem into a coupling problem.
- [Deadline Pressure](deadline-pressure.md)
<br/>  Reflection- or convention-based wiring is faster to write than explicit, typed wiring, so pressed developers reach for it without weighing the refactoring risk.
- [Inexperienced Developers](inexperienced-developers.md)
<br/>  Developers unfamiliar with a framework's convention-over-configuration magic don't recognize that they are creating an implicit runtime contract at all.

## Detection Methods ○

- **Static Pattern Search:** Grep the codebase for reflective or dynamic-loading APIs (`Class.forName`, `loadClass`, dynamic `import()`) and string concatenation immediately before them.
- **Structural Pairing Audit:** Compare source and resource trees for files connected only by identical relative paths and base names.
- **Constant Name Scan:** Search for constants named `PREFIX`, `SUFFIX`, `SEPARATOR`, `SERVICE_NAME`, or similar that hint at a hand-rolled naming protocol.
- **Rename Rehearsal:** Before a real rename, do a trial rename in a branch and run the full test and integration suite, not just the compiler, to see what breaks only at runtime.
- **Persisted Data Sampling:** Check production data, configuration, and exports for stored class names, keys, or IDs that a rename would silently invalidate.

## Examples

A CMS resolves a domain type's list view by taking `type.getSimpleName()`, appending `"ListPage"`, and loading that class reflectively to render the admin UI. A developer renames the `Invoice` model to `Bill` using their IDE's automated rename, which dutifully updates every statically-typed reference; the build is green and the tests pass. In production, however, the admin list view for bills throws a `ClassNotFoundException`, because the reflective lookup now searches for `BillListPage`, a class nobody renamed since the IDE never saw it as a reference to `Invoice`. A second example: a single-page application's backend renders a template attribute `module="frontend/components/InvoiceTable"`, and a bridge resolves that string into a dynamic `import()` on the frontend. When the frontend team reorganizes their component folder during a cleanup, TypeScript compiles cleanly and the frontend's own tests pass, but the backend's hardcoded template string still points at the old path, and the admin dashboard silently renders a blank panel in production.
