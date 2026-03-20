---
title: CV Driven Development
description: Choosing technologies or practices primarily to enhance personal resumes
  rather than solve business problems
category:
- Code
- Communication
- Process
related_problems:
- slug: convenience-driven-development
  similarity: 0.7
- slug: increased-technical-shortcuts
  similarity: 0.55
- slug: cargo-culting
  similarity: 0.55
- slug: decision-avoidance
  similarity: 0.55
- slug: premature-technology-introduction
  similarity: 0.55
- slug: technology-isolation
  similarity: 0.5
solutions:
- boring-technologies
- technical-skills-development
layout: problem
---

## Description

CV Driven Development occurs when technical decisions are made primarily to add impressive technologies, frameworks, or methodologies to developers' resumes rather than to address actual business needs or technical requirements. This leads to the adoption of trendy, complex, or cutting-edge solutions that may be inappropriate for the project's scale, team expertise, or long-term maintenance needs. The practice prioritizes individual career advancement over project success and sustainable software development.

## Indicators ⟡

- Technology proposals focus heavily on novelty or trendiness rather than business value
- Developers emphasize learning opportunities over project requirements during technical discussions
- Frequent requests to use the latest versions of frameworks or experimental technologies
- Resistance to using proven, stable, but "boring" technologies for appropriate use cases
- Technical decisions coincide with developers' job search periods or performance reviews
- Team members explicitly mention resume building when proposing new technologies
- Disproportionate interest in conference-worthy or blog-worthy technical solutions
- Preference for complex architectures when simpler solutions would suffice

## Symptoms ▲

- [Premature Technology Introduction](premature-technology-introduction.md)
<br/>  Technologies are adopted based on resume value rather than project fit, introducing immature or inappropriate tools into the stack.
- [High Maintenance Costs](high-maintenance-costs.md)
<br/>  Unnecessarily complex technology choices driven by resume building create systems that are expensive to maintain after the original developer leaves.
- [Knowledge Gaps](knowledge-gaps.md)
<br/>  When technologies are chosen for resume value rather than team expertise, most team members lack the knowledge to work with them effectively.
- [Technology Stack Fragmentation](technology-stack-fragmentation.md)
<br/>  Different developers introducing their preferred resume-building technologies creates a fragmented stack with many incompatible tools.
## Causes ▼

- [Individual Recognition Culture](individual-recognition-culture.md)
<br/>  A culture that values individual accomplishments over team success incentivizes developers to prioritize personal career advancement in technical decisions.
- [Short-Term Focus](short-term-focus.md)
<br/>  When management does not consider long-term maintainability, developers face no pushback against choosing technologies for personal benefit.
- [Market Pressure](market-pressure.md)
<br/>  Competitive job market pressure motivates developers to build their resumes with trendy technologies to improve their career prospects.
- [High Turnover](high-turnover.md)
<br/>  Developers who chose technologies for their resumes leave once they have gained enough experience to list them, taking critical knowledge with them.
## Detection Methods ○

- Review technical decision documentation for business justification versus technology benefits
- Monitor correlation between individual developers' technology proposals and their career goals
- Assess whether technology choices align with team expertise and project timelines
- Track maintenance effort and bug rates in areas using newer versus established technologies
- Evaluate if technical complexity matches the actual problem complexity being solved
- Interview departing team members about their motivation for technical choices
- Compare project technology stack against industry standards for similar projects
- Monitor recruitment difficulty for roles requiring the chosen technology stack
- Analyze whether technology adoption follows industry hype cycles rather than project needs
- Review retrospectives for mentions of technology-related challenges or regrets

## Examples

A senior developer on a small e-commerce project insists on implementing the backend using a cutting-edge functional programming language, a complex event-sourcing architecture, and the latest NoSQL database, despite the team having no prior experience with these technologies. The project requirements are straightforward: user authentication, product catalog, and order processing. When questioned, the developer mentions wanting to speak about these technologies at conferences and update their LinkedIn profile. The result is a six-month delay in delivery, significant cost overruns, and a system that only one person can maintain. After the developer leaves for a new role highlighting their "modern architecture experience," the remaining team struggles to fix bugs or add features, ultimately requiring a complete rewrite using more conventional technologies.
