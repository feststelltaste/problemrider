---
title: Emulation
description: Reproduce a foreign platform's behavior so existing software runs without modification
category:
- Operations
- Architecture
quality_tactics_url: https://qualitytactics.de/en/compatibility/emulation
problems:
- obsolete-technologies
- technology-lock-in
- vendor-lock-in
- stagnant-architecture
- deployment-environment-inconsistencies
- legacy-skill-shortage
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Identify legacy applications that depend on obsolete hardware or operating systems no longer available
- Evaluate emulation solutions (hardware emulators, OS compatibility layers, runtime emulators) for the target platform
- Test the legacy application thoroughly under emulation to verify behavioral fidelity
- Use emulation as a bridge strategy while planning a proper migration or rewrite
- Document the emulation setup so it can be reproduced if the emulation environment needs rebuilding
- Monitor performance under emulation and establish acceptable performance thresholds

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Extends the life of legacy software without any code changes
- Buys time for planning and executing a proper migration strategy
- Can preserve business-critical functionality that would be expensive to rewrite

**Costs and Risks:**
- Emulation typically incurs performance overhead compared to native execution
- Emulated environments may have subtle behavioral differences that surface as rare bugs
- Relying on emulation indefinitely increases technical debt and operational risk
- Emulation tools themselves may become unsupported or obsolete

## Examples

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A defense contractor ran mission-critical simulation software on a Solaris SPARC platform that was approaching end of vendor support. Rather than rewriting the simulation, which contained decades of validated physics models, the team deployed it under a SPARC emulator on modern x86 hardware. While performance was 30% slower, the simulation results were identical. This bought the organization three years to plan and fund a proper migration to a modern platform while maintaining uninterrupted access to the simulation.
