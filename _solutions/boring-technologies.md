---
title: Boring Technologies
description: Use proven and mature technologies
category:
- Architecture
- Process
quality_tactics_url: https://qualitytactics.de/en/reliability/boring-technologies/
problems:
- cv-driven-development
- cargo-culting
- second-system-effect
- gold-plating
- rapid-prototyping-becoming-production
- assumption-based-development
- suboptimal-solutions
- insufficient-design-skills
- implementation-rework
layout: solution
---

## How to Apply ◆

> In legacy system contexts, "boring technology" does not mean obsolete technology — it means choosing tools and approaches where the failure modes are well understood, the team has genuine expertise, and the operational costs are predictable. This directly counteracts the tendency to adopt flashy solutions that create new problems while trying to solve old ones.

- Establish a technology radar for your organization that explicitly categorizes technologies into "adopt," "trial," "assess," and "hold." Make "adopt" the default for production systems and require a written justification — tied to a specific business need, not developer interest — before introducing anything from the other categories.
- Require a "boring alternative analysis" for every technology proposal: before adopting a new framework, database, or architectural pattern, the proposer must document what the boring, well-understood alternative would be and why it is insufficient. This forces honest evaluation of whether novelty is being chosen for its own sake.
- Apply the "innovation tokens" concept: each team or project has a limited budget of complexity that can be spent on novel technologies or approaches. Once the budget is spent, every remaining technical decision must use proven, well-understood tools. This prevents the cumulative complexity explosion that results from adopting multiple unfamiliar technologies simultaneously.
- Evaluate technology choices against team expertise, not industry trends. A technology that works brilliantly at a company with 500 engineers and a dedicated platform team may be a disaster for a team of eight maintaining a legacy system. Ask "can our team debug this at 3 AM?" rather than "is this what modern companies use?"
- Prevent prototype code from reaching production by establishing a clear prototype-to-production gate. Prototypes built with experimental technologies serve their purpose — demonstrating feasibility — but the production implementation should use the team's most reliable stack unless there is a compelling technical reason otherwise.
- Combat the second-system effect by defining a minimum viable replacement scope before design begins and enforcing it through regular scope reviews. When replacing a legacy system, the tendency to add every feature the old system lacked must be actively resisted by requiring validated user demand for each proposed capability.
- Make technology decisions as a team rather than allowing individual developers to unilaterally introduce new tools. Group decision-making naturally filters out CV-driven choices because the proposer must convince colleagues who will also have to maintain the technology.
- Document all technology decisions in Architecture Decision Records (ADRs) that include the context, the options considered, the decision rationale, and the expected consequences. This creates accountability and makes it visible when decisions were driven by resume building rather than project needs.

## Tradeoffs ⇄

> Boring technologies reduce surprise and operational risk, but they require the discipline to resist the appeal of novelty and the maturity to accept that solving problems with well-understood tools is more valuable than solving them with impressive ones.

**Benefits:**

- Eliminates the knowledge gap that occurs when CV-driven technology choices leave the team unable to maintain systems after the original developer moves on, because the entire team is already proficient with the chosen stack.
- Reduces implementation rework by avoiding technologies whose limitations only become apparent after significant development effort, since boring technologies have well-documented limitations and workarounds.
- Prevents the cargo cult anti-pattern by requiring teams to understand why a technology is appropriate for their context rather than adopting it because successful companies use it.
- Lowers operational costs because the team can troubleshoot production issues using their existing expertise rather than learning debugging techniques for unfamiliar tools under incident pressure.
- Counteracts gold plating and the second-system effect by constraining the solution space to proven approaches, making it harder to justify unnecessary complexity or speculative features.

**Costs and Risks:**

- Taken to an extreme, "boring technologies" can become an excuse for technological stagnation, where teams never adopt genuinely beneficial innovations because any change is perceived as risky.
- Teams that only use familiar tools may miss significant productivity or reliability improvements offered by newer technologies that have matured enough to be considered "boring" at other organizations.
- Talented developers who value learning opportunities may become frustrated if the policy is perceived as stifling innovation, potentially increasing turnover — the policy must allow controlled experimentation in non-critical contexts.
- The definition of "boring" is subjective and context-dependent: what is boring and well-understood for one team may be novel and risky for another, so the policy must be calibrated to each team's actual expertise.
- Legacy systems already built on now-obsolete technologies may need migrations to more modern (but still boring) alternatives, and the boring technology principle should not be used to justify staying on unsupported platforms.

## Examples

> The following scenarios illustrate how the boring technology principle has been applied to prevent common anti-patterns in legacy system modernization.

A fintech startup replacing a legacy payment processing system considered building the replacement with an event-sourcing architecture backed by Apache Kafka, a GraphQL API layer, and a distributed NoSQL database — technologies that several developers wanted on their resumes. A senior architect applied the boring technology analysis: the team had deep expertise in PostgreSQL, REST APIs, and a standard message queue. The boring alternative could handle the company's transaction volume with headroom to spare. The team built the replacement with proven technologies and delivered it in four months instead of the estimated twelve that the "modern" stack would have required. Two years later, the system handles ten times the original volume without architectural changes, and any team member can debug production issues independently.

A government agency rebuilding its citizen portal suffered from the second-system effect: the design included AI-powered form assistance, blockchain-based document verification, a microservices architecture with 23 planned services, and real-time analytics — all features intended to address frustrations with the old system. After eighteen months and significant budget overrun, the team had delivered only three of the planned services. A project reset applied the boring technology principle: the team identified the five most-used citizen interactions, built them as a standard web application with a relational database, and deployed within four months. The remaining functionality was prioritized based on actual usage data rather than speculative requirements, and most of the originally planned "innovative" features were never requested by actual users.

A logistics company discovered that three different teams had independently introduced three different message queue technologies — RabbitMQ, Apache Kafka, and Amazon SQS — because each team's lead developer wanted experience with a different tool. No single team member understood all three, and cross-team debugging of message flow issues required assembling experts from multiple teams. Applying the boring technology principle, the engineering leadership mandated consolidation to RabbitMQ, which the largest number of developers already understood and which met the technical requirements of all three use cases. The consolidation took six weeks but eliminated an entire category of production incidents related to misconfigured queue consumers, and on-call engineers could now troubleshoot any message flow without needing specialist knowledge.
