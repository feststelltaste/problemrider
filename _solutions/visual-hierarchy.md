---
title: Visual Hierarchy
description: Highlight important elements on the user interface and create a clear visual structure
category:
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/visual-hierarchy/
problems:
- poor-user-experience-ux-design
- user-confusion
- cognitive-overload
- increased-cognitive-load
- user-frustration
- negative-user-feedback
- increased-error-rates
layout: solution
---

## How to Apply ◆

> Legacy systems often present all information with equal visual weight, making it difficult for users to identify what is important. Visual hierarchy uses size, color, contrast, and spacing to guide attention.

- Establish a clear heading hierarchy using distinct font sizes and weights for page titles, section headers, and sub-sections. Legacy systems often use the same font size for everything, making structure invisible.
- Use size and prominence to indicate importance. Primary actions like "Save" or "Submit" should be visually dominant, while secondary actions like "Cancel" and tertiary actions like "Advanced Settings" should be progressively less prominent.
- Apply whitespace strategically to separate logical groups of content. Legacy interfaces that pack elements tightly together create visual noise that makes scanning difficult.
- Use color purposefully and consistently: a limited palette where each color carries meaning, such as red for errors, green for success, and blue for interactive elements. Avoid decorative color that adds visual noise without conveying information.
- Differentiate between data and labels. Field labels should be visually distinct from the data they describe, and required fields should be visually distinguishable from optional ones.
- De-emphasize secondary information through smaller font sizes, lighter colors, or collapsible sections, rather than removing it entirely.

## Tradeoffs ⇄

> Visual hierarchy makes interfaces scannable and intuitive, but requires design expertise and consistent application across the system.

**Benefits:**

- Enables users to scan screens quickly and find the information they need without reading every element, significantly improving productivity.
- Reduces errors by making primary actions visually dominant and secondary or dangerous actions less prominent.
- Makes the system feel more modern and professional, improving user confidence and satisfaction.
- Reduces cognitive overload by organizing information in a way that the eye can process naturally.

**Costs and Risks:**

- Establishing visual hierarchy requires design expertise that legacy development teams may lack, as it involves understanding typography, color theory, and layout principles.
- Retrofitting visual hierarchy into legacy CSS built with table-based layouts or inline styles can require significant refactoring.
- Inconsistent application of visual hierarchy across different parts of the legacy system can create a jarring experience as users move between modules.
- Cultural and accessibility considerations affect color meaning and contrast requirements, adding complexity to visual design decisions.

## Examples

> Legacy systems that treat all content as equally important end up making nothing feel important, overwhelming users with visual noise.

A legacy case management system displays case details on a single screen with twenty-eight fields arranged in a grid. All fields use the same font size, the same label style, and the same spacing. Case workers scanning the screen to find a case's current status must read through fields like creation date, internal case code, assigned office, and several rarely needed administrative fields before finding the status, which looks identical to everything else. The team redesigns the screen with a visual hierarchy: a prominent header area shows the case title, status, and priority with large, bold text and status-specific color coding. Contact information and key dates appear in a secondary section below. Administrative fields are collapsed into an expandable panel. Case workers report that they can now glance at a case and immediately understand its status and priority, something that previously required careful reading. The average time to triage a new case drops because the most important information is the most visible.
