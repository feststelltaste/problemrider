---
title: Long-Lived Feature Branches
description: Code is not being reviewed and merged in a timely manner, leading to
  integration problems and increased risk.
category:
- Code
- Process
related_problems:
- slug: large-feature-scope
  similarity: 0.7
- slug: slow-feature-development
  similarity: 0.6
- slug: merge-conflicts
  similarity: 0.55
- slug: inconsistent-codebase
  similarity: 0.55
- slug: fear-of-breaking-changes
  similarity: 0.55
- slug: feature-creep
  similarity: 0.55
solutions:
- feature-flags
layout: problem
---

## Description
Long-lived feature branches are a common problem in teams that use a branching model for development. When a feature branch is kept separate from the main branch for an extended period of time, it can become difficult and risky to merge back in. The longer a branch lives, the more it diverges from the main branch, increasing the likelihood of merge conflicts and making it harder to integrate the changes. This can lead to a "merge hell" scenario, where a significant amount of time is spent resolving conflicts instead of delivering value.

## Indicators ⟡
- Feature branches are often days or weeks old.
- Merging a feature branch is a major event that requires a lot of coordination.
- The team is constantly dealing with merge conflicts.
- The team is afraid to merge feature branches for fear of breaking something.

## Symptoms ▲

- [Merge Conflicts](merge-conflicts.md)
<br/>  The longer a branch diverges from mainline, the more likely conflicting changes accumulate, creating painful merge conflicts.
- [Integration Difficulties](integration-difficulties.md)
<br/>  Code developed in isolation for extended periods becomes structurally incompatible with mainline changes, making integration costly.
- [Regression Bugs](regression-bugs.md)
<br/>  Large merges from long-lived branches introduce many changes at once, increasing the chance of subtle regressions.
- [Implementation Rework](implementation-rework.md)
<br/>  When parallel development on mainline makes a branch's approach incompatible, significant rework is needed before merging.
- [Large Pull Requests](large-pull-requests.md)
<br/>  Long-lived branches accumulate many changes, resulting in large pull requests that are difficult to review effectively.
## Causes ▼

- [Large Feature Scope](large-feature-scope.md)
<br/>  Features with overly broad scope take longer to implement, naturally extending branch lifetimes.
- [Long Build and Test Times](long-build-and-test-times.md)
<br/>  Slow CI pipelines discourage frequent integration, as developers avoid the long feedback cycles of merging often.
- [Review Bottlenecks](review-bottlenecks.md)
<br/>  Bottlenecks in the code review process delay merges, forcing branches to live longer than intended.
- [Fear of Breaking Changes](fear-of-breaking-changes.md)
<br/>  Teams delay merging because they fear introducing breaking changes, keeping branches separate longer.
## Detection Methods ○

- **Version Control System Analysis:** Monitor the age and size of feature branches in your Git repository.
- **Code Review Metrics:** Track the time it takes for pull requests to be reviewed and merged.
- **Build/Deployment Frequency:** Observe how often the main branch is built and deployed.
- **Developer Feedback:** Ask developers about their experiences with merge conflicts and integration challenges.

## Examples
A team is developing a major new module for an application. The development takes three months on a single feature branch. When it's time to merge, there are hundreds of conflicts with the main branch, and the team spends weeks resolving them, delaying the release. In another case, a developer works on a new feature for several weeks without pushing their changes or creating a pull request. Meanwhile, another developer makes a related change on the main branch. When the first developer finally tries to merge, their changes are incompatible, requiring significant rework. This problem is often a symptom of a team that has not fully embraced continuous integration or agile development practices. It can lead to significant technical debt and slow down the overall development process.
