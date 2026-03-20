---
title: Localization
description: Adapting software to different languages, regions, and cultural conventions
category:
- Requirements
- Code
quality_tactics_url: https://qualitytactics.de/en/usability/localization/
problems:
- poor-user-experience-ux-design
- user-confusion
- user-frustration
- competitive-disadvantage
- feature-gaps
- negative-user-feedback
- hardcoded-values
- customer-dissatisfaction
layout: solution
---

## How to Apply ◆

> Legacy systems were often built for a single language and region. As organizations expand, the lack of localization becomes a significant barrier to adoption in new markets.

- Extract all hardcoded strings from the legacy codebase into resource files or a localization framework. This includes UI labels, error messages, help text, email templates, and report headers.
- Implement locale-aware formatting for dates, times, numbers, currencies, and addresses. Legacy systems that display dates as "MM/DD/YYYY" confuse users in regions that expect "DD.MM.YYYY" or "YYYY-MM-DD."
- Support Unicode throughout the stack, including the database, APIs, and UI rendering. Legacy systems built with ASCII-only assumptions break when handling characters from non-Latin scripts.
- Design UI layouts to accommodate text expansion. German and French translations are typically thirty to forty percent longer than English, and right-to-left languages like Arabic require mirrored layouts.
- Externalize locale-specific business rules such as tax calculations, address formats, and regulatory requirements so they can be configured per region without code changes.
- Establish a translation workflow with professional translators who understand the domain, rather than relying on developer-level machine translation.

## Tradeoffs ⇄

> Localization opens new markets and improves user experience for non-English speakers, but adds significant complexity to development and testing.

**Benefits:**

- Enables expansion into new markets by removing language and cultural barriers that prevent adoption.
- Reduces user confusion and errors caused by unfamiliar date formats, number conventions, and terminology.
- Demonstrates respect for users' languages and cultures, improving satisfaction and trust.
- Eliminates hardcoded values throughout the codebase, which improves maintainability as a side effect.

**Costs and Risks:**

- Extracting strings from a legacy codebase with years of hardcoded text is labor-intensive and error-prone, as strings may be embedded in unexpected places.
- Testing the application in every supported locale multiplies the QA effort, and automated tests must account for variable string lengths and formats.
- Translation is an ongoing cost: every new feature, error message, and UI change requires translation into all supported languages.
- Right-to-left language support may require significant layout changes that are difficult to retrofit into legacy CSS and component structures.
- Cultural localization goes beyond translation and includes considerations like icon meanings, color associations, and content appropriateness that require domain expertise.

## How It Could Be

> Localization failures in legacy systems range from comical to costly, and they become urgent when the organization expands internationally.

A legacy accounting system built for the US market is deployed to the company's new European offices without localization. European accountants immediately encounter problems: dates are ambiguous because the system displays them in MM/DD format without indication, currency amounts use periods as decimal separators instead of the commas expected in Germany and France, and all error messages are in English. The team undertakes a phased localization effort, starting with date and number formatting using the browser's locale settings, then extracting UI strings into resource bundles for translation. After six months of incremental work, the system supports English, German, and French with locale-appropriate formatting. European users report that the system is finally usable for their daily work, and data entry errors caused by date format confusion are eliminated.
