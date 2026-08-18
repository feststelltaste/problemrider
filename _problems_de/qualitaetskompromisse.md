---
title: Qualitätskompromisse
description: Qualitätsstandards werden absichtlich gesenkt oder Abkürzungen genommen,
  um Termine, Budgets oder andere Einschränkungen zu erfüllen, was langfristige
  Probleme schafft.
category:
- Code
- Process
related_problems:
- slug: lower-code-quality
  similarity: 0.7
- slug: reduced-feature-quality
  similarity: 0.65
- slug: increased-technical-shortcuts
  similarity: 0.65
- slug: deadline-pressure
  similarity: 0.65
- slug: quality-degradation
  similarity: 0.65
- slug: test-debt
  similarity: 0.6
solutions:
- definition-of-done
- secure-software-development
- security-culture
- error-budgets
- production-readiness-criteria
- workaround-registry
- defect-triage-process
- lightweight-design-review
- quality-ratchet
- debt-accrual-analysis
- debt-classification
layout: problem
lang: de
en_slug: quality-compromises
---

## Description

Qualitätskompromisse treten auf, wenn Teams oder Organisationen absichtlich niedrigere Qualitätsstandards akzeptieren, Qualitätspraktiken überspringen oder Abkürzungen nehmen, um unmittelbare Einschränkungen wie Termine, Budgets oder Ressourcenlimits zu erfüllen. Während diese Kompromisse kurzfristige Vorteile bieten können, schaffen sie typischerweise langfristige Probleme, einschließlich technischer Schulden, erhöhter Wartungskosten und verringerter Systemzuverlässigkeit.

## Indicators ⟡

- Qualitätspraktiken werden übersprungen oder reduziert, um Termine einzuhalten
- Testabdeckung wird absichtlich reduziert, um die Lieferung zu beschleunigen
- Code-Reviews werden übereilt oder für dringende Änderungen umgangen
- Design- und Architekturentscheidungen priorisieren Geschwindigkeit über Wartbarkeit
- Bekannte Qualitätsprobleme werden akzeptiert statt angegangen

## Symptoms ▲

- [Qualitätsverschlechterung](qualitaetsverschlechterung.md)
<br/>  Wiederholte Abkürzungen und übersprungene Qualitätspraktiken verursachen kumulativen Rückgang der Systemqualität über die Zeit.
- [Geringere Codequalität](geringere-codequalitaet.md)
<br/>  Übersprungene Code-Reviews und Tests produzieren Code, der schwerer zu warten und fehleranfälliger ist.
- [Zunehmende technische Abkürzungen](zunehmende-technische-abkuerzungen.md)
<br/>  Sobald Qualitätsabkürzungen akzeptabel werden, folgen weitere Abkürzungen, während der Präzedenzfall das Abkürzen normalisiert.
- [Zunehmende Brüchigkeit](zunehmende-bruechigkeit.md)
<br/>  Ungetesteter und schlecht gereviewter Code führt versteckte Fragilitäten ein, die sich über die Zeit verstärken.
- [Inkonsistente Qualität](inkonsistente-qualitaet.md)
<br/>  Manche Teile des Systems sind gut gebaut, während unter Druck entwickelte Bereiche merklich geringere Qualität haben.
- [Qualitäts-blinde Flecken](qualitaets-blinde-flecken.md)
<br/>  Das absichtliche Überspringen von Tests schafft systematische Lücken in der Qualitätsverifikation.

## Causes ▼

- [Zeitdruck](zeitdruck.md)
<br/>  Enge Termine zwingen Teams, zwischen Zeitplaneinhaltung und Aufrechterhaltung von Qualitätsstandards zu wählen.
- [Projekt-Ressourcenbeschränkungen](projekt-ressourcenbeschraenkungen.md)
<br/>  Unzureichende Ressourcen machen es unmöglich, Qualitätsstandards innerhalb gegebener Einschränkungen aufrechtzuerhalten.
- [Ständiges Feuerlöschen](staendiges-feuerloeschen.md)
<br/>  Kontinuierliche dringende Arbeit lässt keine Zeit für Qualitätspraktiken, was Teams zu Abkürzungen zwingt.

## Detection Methods ○

- **Nachverfolgung von Qualitätsmetriken:** Überwachung von Trends bei Codequalität, Testabdeckung und Defektraten
- **Prozess-Compliance-Überwachung:** Nachverfolgung, wie oft Qualitätsprozesse übersprungen oder abgekürzt werden
- **Bewertung technischer Schulden:** Messung der Anhäufung technischer Schulden über die Zeit
- **Team-Zufriedenheitsbefragungen:** Bewertung der Team-Zufriedenheit mit Qualitätsstandards und -praktiken
- **Nach-Release-Qualitätsanalyse:** Bewertung von Qualitätsproblemen, die nach dem Deployment entdeckt wurden

## Examples

Ein Entwicklungsteam, das einem kritischen Termin gegenübersteht, entscheidet, Unit-Tests für neue Features zu überspringen und reduziert Code-Review-Anforderungen auf Einzel-Reviewer-Genehmigung statt der üblichen zwei Reviewer. Während dies ihnen erlaubt, den Termin einzuhalten, enthält die veröffentlichte Software mehrere Fehler, die Notfall-Hotfixes erfordern, und die Codebasis wird aufgrund ungetesteten Codes schwerer zu warten. Ein weiteres Beispiel betrifft ein Projekt, bei dem architektonische Abkürzungen genommen werden, um schnell mit einem Drittanbietersystem zu integrieren, was enge Kopplung und komplexe Workarounds schafft, die zukünftige Änderungen extrem schwierig und teuer machen.
