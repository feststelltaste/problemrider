---
title: Angst vor Breaking Changes
description: Das Team scheut sich, Änderungen an der Codebasis vorzunehmen, aus
  Angst, bestehende Funktionalität zu brechen, was zu einem stagnierenden und veralteten
  System führen kann.
category:
- Code
- Culture
- Process
related_problems:
- slug: fear-of-change
  similarity: 0.8
- slug: resistance-to-change
  similarity: 0.75
- slug: maintenance-paralysis
  similarity: 0.7
- slug: brittle-codebase
  similarity: 0.7
- slug: refactoring-avoidance
  similarity: 0.7
- slug: fear-of-failure
  similarity: 0.7
solutions:
- blue-green-canary-deployments
- feature-flags
- strangler-fig-pattern
- backward-compatibility
- backward-compatible-apis
- code-coverage-analysis
- compatibility-as-error
- compatibility-requirements
- consumer-driven-contracts
- functional-spike
- functional-tests
- regression-tests
- characterization-tests
- mikado-method
- small-change-batches
- change-impact-analysis
- parallel-run
- continuous-dependency-updates
- automated-code-migration
layout: problem
lang: de
en_slug: fear-of-breaking-changes
---

## Description
Angst vor Breaking Changes ist ein verbreitetes Problem in der Softwareentwicklung. Es ist die Angst, dass eine Änderung an der Codebasis unbeabsichtigte Konsequenzen haben und bestehende Funktionalität brechen wird. Diese Angst kann lähmend sein und ein Team davon abhalten, notwendige Änderungen am System vorzunehmen. Wenn ein Team Angst hat, Änderungen vorzunehmen, kann das System stagnieren und veralten. Dies kann zu einer Reihe von Problemen führen, einschließlich eines Rückgangs der Nutzerzufriedenheit, eines Verlusts des Wettbewerbsvorteils und erheblicher Frustration für das Entwicklungsteam.

## Indicators ⟡
- Das Team zögert, Änderungen an der Codebasis vorzunehmen.
- Das Team refaktoriert den Code nicht.
- Das Team hält nicht mit den neuesten Technologien Schritt.
- Das Team innoviert nicht.

## Symptoms ▲

- [Refactoring-Vermeidung](refactoring-vermeidung.md)
<br/>  Wenn das Team Breaking Changes fürchtet, vermeidet es aktiv Refactoring, selbst wenn es weiß, dass es notwendig ist.
- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Statt bestehenden Code zu ändern, schaffen Entwickler Workarounds, um riskante Bereiche nicht anfassen zu müssen, was Komplexität hinzufügt.
- [Stagnierende Architektur](stagnierende-architektur.md)
<br/>  Die Systemarchitektur entwickelt sich nicht weiter, weil das Team zu viel Angst hat, die für Verbesserungen nötigen strukturellen Änderungen vorzunehmen.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Das Vermeiden notwendiger Änderungen führt dazu, dass sich technische Schulden anhäufen, während die Codebasis zunehmend veraltet.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Angst vor Breaking Changes verlangsamt die Entwicklung, während Teams übermäßige Vorsichtsmaßnahmen treffen oder Features auf Umwegen umsetzen.
- [Systemstagnation](systemstagnation.md)
<br/>  Das System bleibt unverändert und entwickelt sich nicht weiter, weil das Team es vermeidet, Änderungen vorzunehmen.

## Causes ▼

- [Legacy-Code ohne Tests](legacy-code-ohne-tests.md)
<br/>  Ohne automatisierte Tests gibt es kein Sicherheitsnetz, um zu verifizieren, dass Änderungen bestehende Funktionalität nicht brechen, was die Angst rational macht.
- [Geschichte fehlgeschlagener Änderungen](geschichte-fehlgeschlagener-aenderungen.md)
<br/>  Vergangene Erfahrungen, bei denen Änderungen Produktionsausfälle verursachten, schaffen anhaltende Angst und Zurückhaltung bei künftigen Änderungen.
- [Brüchige Codebasis](bruechige-codebasis.md)
<br/>  Eine brüchige Codebasis, bei der kleine Änderungen häufig unerwartete Ausfälle verursachen, gibt dem Team legitime Gründe, Änderungen zu fürchten.
- [Hohe Kopplung und geringe Kohäsion](hohe-kopplung-und-geringe-kohaesion.md)
<br/>  Eng gekoppelter Code bedeutet, dass Änderungen in einem Bereich häufig andere Bereiche betreffen, was es tatsächlich riskant macht, das System zu ändern.
- [Schlechte Testabdeckung](schlechte-testabdeckung.md)
<br/>  Ohne ausreichende Testabdeckung können Entwickler nicht verifizieren, dass ihre Änderungen sicher sind, was die Angst vor Änderungen verstärkt.

## Detection Methods ○
- **Code-Churn:** Analyse der Historie der Codebasis, um zu sehen, wie oft der Code geändert wird.
- **Technische Schulden:** Nachverfolgung der Menge technischer Schulden im System.
- **Entwickler-Umfragen:** Befragung von Entwicklern zu ihren Gefühlen bezüglich Änderungen am System.
- **Experimentierbereitschaft:** Ist das Team bereit, mit neuen Ideen und Technologien zu experimentieren?

## Examples
Ein Unternehmen hat ein Legacy-System, das für sein Geschäft kritisch ist. Das System ist alt und brüchig, und das Team hat Angst, Änderungen daran vorzunehmen. Infolgedessen wird das System nicht aktualisiert und fällt hinter die Konkurrenz zurück. Das Unternehmen verliert Marktanteile und läuft Gefahr, aus dem Geschäft ausscheiden zu müssen. Das Team weiß, dass es Änderungen am System vornehmen muss, ist aber von Angst gelähmt.
