---
title: Angst vor Veränderung
description: Entwickler zögern, bestehenden Code zu ändern, aufgrund des hohen Risikos,
  etwas zu brechen.
category:
- Code
- Process
related_problems:
- slug: fear-of-breaking-changes
  similarity: 0.8
- slug: resistance-to-change
  similarity: 0.8
- slug: history-of-failed-changes
  similarity: 0.75
- slug: maintenance-paralysis
  similarity: 0.75
- slug: refactoring-avoidance
  similarity: 0.75
- slug: fear-of-failure
  similarity: 0.7
solutions:
- blameless-postmortems
- feature-flags
- strangler-fig-pattern
- acceptance-tests
- automated-tests
- bubble-context
- canary-releases
- chaos-engineering
- dark-launches
- feature-toggles
- forward-compatibility
- functional-spike
- integration-tests
- privacy-by-design
- prototypes
- prototyping
- raising-user-awareness
- refactoring-katas
- resilience
- restore-points
- risk-analysis
- rollback-mechanisms
- security-culture
- simulation-environments
- smoke-testing
- technical-spike
- test-driven-development-tdd
- tracer-bullets
- patch-management
- undo-and-redo
- characterization-tests
- mikado-method
- pilot-projects
- technical-debt-assessment
- debt-remediation-estimation
layout: problem
lang: de
en_slug: fear-of-change
---

## Description

Angst vor Veränderung ist eine psychologische und praktische Barriere, die Entwickler davon abhält, bestehenden Code zu ändern. Diese Angst entspringt legitimen Bedenken, Fehler einzuführen, Funktionalität zu brechen oder Systeminstabilität zu verursachen. Wenn Entwickler durchgängig notwendige Änderungen oder Verbesserungen aufgrund dieser Bedenken vermeiden, deutet dies auf tiefere systemische Probleme mit Codequalität, Testpraktiken und Systemarchitektur hin. Diese Angst kann sich selbst verstärken, während vermiedene Änderungen technische Schulden anhäufen, was künftige Modifikationen noch riskanter macht.

## Indicators ⟡
- Entwickler äußern Zurückhaltung oder Angst, wenn sie gebeten werden, bestimmte Teile des Systems zu ändern
- Schätzungen für scheinbar einfache Änderungen sind aufgrund wahrgenommenen Risikos aufgebläht
- Das Team wählt häufig Workarounds, statt Grundursachen anzugehen
- Diskussionen über Codeänderungen konzentrieren sich mehr darauf, was brechen könnte, als auf die Vorteile der Änderung
- Neue Features werden als Ergänzungen umgesetzt statt als Verbesserungen an bestehendem Code

## Symptoms ▲

- [Refactoring-Vermeidung](refactoring-vermeidung.md)
<br/>  Entwickler, die Veränderung fürchten, vermeiden aktiv Refactoring, selbst wenn die Codequalität es erfordert.
- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Statt bestehenden Code zu ändern, setzen Entwickler Workarounds um, die Komplexität hinzufügen, ohne Grundprobleme anzugehen.
- [Große Schätzungen für kleine Änderungen](grosse-schaetzungen-fuer-kleine-aenderungen.md)
<br/>  Angst bläht Schätzungen auf, während Entwickler wahrgenommenes Risiko berücksichtigen, was einfache Änderungen unverhältnismäßig teuer erscheinen lässt.
- [Code-Duplizierung](code-duplizierung.md)
<br/>  Entwickler kopieren bestehenden Code, statt gemeinsam genutzte Komponenten zu ändern, was zu duplizierter Logik in der Codebasis führt.
- [Wartungslähmung](wartungslaehmung.md)
<br/>  Das Team wird gelähmt und kann notwendige Wartung nicht durchführen, weil es nicht verifizieren kann, dass Änderungen sicher sind.
- [Stagnierende Architektur](stagnierende-architektur.md)
<br/>  Die Architektur entwickelt sich nicht mehr weiter, weil das Team die für Verbesserungen nötigen strukturellen Änderungen vermeidet.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Angst vor Veränderung verlangsamt direkt die Feature-Entwicklung, während Teams übermäßige Vorsichtsmaßnahmen treffen oder Workarounds umsetzen.

## Causes ▼

- [Brüchige Codebasis](bruechige-codebasis.md)
<br/>  Eine brüchige Codebasis, bei der Änderungen häufig Fehler einführen, gibt Entwicklern legitime Gründe, Änderungen zu fürchten.
- [Legacy-Code ohne Tests](legacy-code-ohne-tests.md)
<br/>  Ohne automatisierte Tests, die das Verhalten nach Änderungen verifizieren, birgt jede Änderung ein nicht quantifizierbares Risiko.
- [Hohe Kopplung und geringe Kohäsion](hohe-kopplung-und-geringe-kohaesion.md)
<br/>  Eng gekoppelte Komponenten bedeuten, dass Änderungen unvorhersehbare Wellenwirkungen haben, was die Zurückhaltung der Entwickler rechtfertigt.
- [Geschichte fehlgeschlagener Änderungen](geschichte-fehlgeschlagener-aenderungen.md)
<br/>  Eine Vorgeschichte von Änderungen, die Produktionsvorfälle verursacht haben, schafft eine Kultur der Vorsicht und Angst rund um Modifikationen.
- [Schuldzuweisungskultur](schuldzuweisungskultur.md)
<br/>  Wenn Fehler bestraft statt als Lerngelegenheiten behandelt werden, werden Entwickler risikoscheu und vermeiden es, Änderungen vorzunehmen.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Eine mit angehäuften Abkürzungen und Komplexität belastete Codebasis macht die Konsequenzen jeder Änderung schwerer vorhersehbar, was die Zurückhaltung der Entwickler verstärkt, sie zu ändern.

## Detection Methods ○
- **Entwickler-Umfragen:** Befragung von Teammitgliedern zu ihrem Vertrauensniveau bei Änderungen an unterschiedlichen Teilen des Systems
- **Änderungshäufigkeitsanalyse:** Beobachtung, wie oft unterschiedliche Module geändert werden; durchgängig vermiedene Bereiche können auf Angst hindeuten
- **Schätzungsmuster:** Suche nach Mustern, bei denen ähnliche Änderungen stark unterschiedliche Schätzungen haben, abhängig vom betroffenen Codebereich
- **Code-Review-Kommentare:** Beobachtung übermäßiger Vorsicht oder langwieriger Diskussionen über potenzielle Risiken während Code-Reviews
- **Retrospektiven-Feedback:** Achten auf Bedenken über Code-Stabilität und Änderungsschwierigkeit während Team-Retrospektiven

## Examples

Ein Team muss eine Geschäftsregel in seinem Auftragsverarbeitungssystem aktualisieren. Die Änderung selbst ist konzeptionell einfach – die Anpassung einer Rabattberechnung –, aber die Funktion, die Rabatte handhabt, verwaltet auch Bestandsaktualisierungen, sendet E-Mail-Benachrichtigungen und protokolliert Analyseereignisse. Der Entwickler, der die Änderung vornehmen soll, schätzt zwei Wochen statt zwei Stunden, weil er befürchtet, dass die Änderung der Rabattlogik unbeabsichtigt das E-Mail-System brechen oder Bestandsunstimmigkeiten verursachen könnte. Diese Angst ist angesichts der engen Kopplung gerechtfertigt, verhindert aber, dass das Team notwendige Geschäftsänderungen effizient vornimmt. Letztlich setzt es die Rabattänderung als separate Funktion mit duplizierter Logik um, statt die ursprüngliche problematische Funktion zu beheben.
