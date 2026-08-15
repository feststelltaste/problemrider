---
title: Raising User Awareness
description: Sensitizing and training employees and users on security topics
category:
- Security
- Culture
problems:
- knowledge-gaps
- inadequate-onboarding
- implicit-knowledge
- fear-of-change
- resistance-to-change
- workaround-culture
- password-security-weaknesses
layout: solution
related_solutions:
- slug: security-training
  similarity: 0.85
- slug: security-policies-for-users
  similarity: 0.8
- slug: security-community
  similarity: 0.8
- slug: security-certification
  similarity: 0.75
- slug: security-incident-handling
  similarity: 0.75
- slug: vulnerability-scans
  similarity: 0.75
---

## Description

Raising user awareness is a set of ongoing educational activities — training sessions, simulated phishing campaigns, role-specific modules, security bulletins — intended to build a human layer of defense against threats like social engineering and credential theft that technical controls alone cannot fully address, since many attacks are designed specifically to exploit a person's judgment rather than a system's code. Applied to legacy environments, it directly targets a common and specific weakness: legacy systems are disproportionately likely to be operated with shared accounts, weak passwords, and other informal access practices that accumulated over a long operational history, precisely because the humans using them were never given a structured reason to change those habits. Awareness programs also function as an unplanned discovery mechanism — asking employees to think critically about their own access and credentials tends to surface undocumented shared accounts and other artifacts of that same informal history, findings that then justify a broader access review. The approach is complementary rather than a substitute for technical hardening, since improved awareness reduces the likelihood and impact of social engineering but does nothing on its own to close a technical vulnerability elsewhere in the legacy stack. Its main costs are the need for continuous content refresh as threats evolve, the reputational risk of overly aggressive simulated attacks damaging trust, and the general difficulty of proving a clear return on investment for a program whose success is measured by incidents that did not happen.

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Conduct regular security awareness training sessions covering common threats like phishing, social engineering, and credential theft
- Create role-specific training modules that address the security concerns relevant to each user group
- Run simulated phishing campaigns to measure awareness levels and identify areas needing improvement
- Establish clear reporting channels for users to flag suspicious activities or potential security incidents
- Integrate security awareness into onboarding programs for new employees and contractors
- Distribute regular security bulletins highlighting recent threats and best practices
- Gamify security awareness through quizzes, competitions, and recognition programs

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Reduces the likelihood of successful social engineering attacks
- Creates a human layer of defense that complements technical security controls
- Improves incident reporting speed and quality
- Builds a security-conscious culture that persists beyond individual training events

**Costs and Risks:**
- Training programs require ongoing investment and regular content updates
- Overly aggressive simulated attacks can damage employee trust and morale
- Awareness alone does not prevent all attacks; technical controls remain essential
- Measuring the ROI of awareness programs is inherently difficult

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A manufacturing company running legacy ERP systems experienced repeated credential compromises because employees used simple passwords and shared accounts. The security team introduced quarterly awareness sessions combined with monthly simulated phishing emails. Within six months, phishing click rates dropped from 32% to 8%, and employees began proactively reporting suspicious emails. The initiative also led to the discovery of several shared service accounts in the legacy system that had never been documented, prompting a broader access review.
