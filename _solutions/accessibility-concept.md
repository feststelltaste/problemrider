---
title: Accessibility Concept
description: Design of software to make it accessible and usable for people with disabilities
category:
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/accessibility-concept
problems:
- poor-user-experience-ux-design
- customer-dissatisfaction
- regulatory-compliance-drift
- negative-user-feedback
- user-frustration
- feature-gaps
layout: solution
---

## How to Apply ◆

> Concrete steps, approaches, or practices to implement this solution in a legacy system context.

- Audit the legacy application against WCAG (Web Content Accessibility Guidelines) standards to identify barriers
- Prioritize remediation of the most impactful accessibility issues: keyboard navigation, screen reader support, and color contrast
- Add semantic HTML and ARIA attributes to legacy UI components that lack proper accessibility markup
- Implement focus management and keyboard navigation for legacy interactive elements
- Ensure all images, icons, and non-text content have appropriate alternative text
- Test with assistive technologies including screen readers, magnifiers, and voice control software
- Include users with disabilities in usability testing to validate accessibility improvements
- Create an accessibility style guide for ongoing development to prevent regression

## Tradeoffs ⇄

> What you gain and what you give up by applying this solution.

**Benefits:**
- Expands the user base to include people with disabilities who were previously excluded
- Satisfies legal and regulatory requirements for digital accessibility
- Improves usability for all users, as accessible design often enhances overall user experience
- Reduces legal risk from accessibility-related complaints and lawsuits

**Costs and Risks:**
- Retrofitting accessibility into legacy UIs built without semantic markup can be labor-intensive
- Legacy frameworks or custom UI components may have fundamental accessibility limitations
- Comprehensive accessibility compliance requires ongoing testing and maintenance
- Some legacy visual designs may need significant rework to meet contrast and layout requirements

## How It Could Be

> Concrete examples or scenarios from legacy system contexts that illustrate this solution in practice.

A government services portal built in 2009 received multiple complaints from citizens unable to complete forms using screen readers. An accessibility audit revealed that the legacy application used non-semantic HTML tables for layout, lacked form labels, and had custom JavaScript controls that were completely invisible to assistive technology. The team prioritized the five most-used forms, replacing table-based layouts with semantic HTML, adding ARIA labels, and implementing keyboard navigation. These changes enabled screen reader users to complete forms independently for the first time. The improvements also benefited sighted users by creating a cleaner, more logical form flow, reducing overall form abandonment rates by 15%.
