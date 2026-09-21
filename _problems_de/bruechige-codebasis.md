---
title: Brüchige Codebasis
description: Der bestehende Code lässt sich nur schwer ändern, ohne neue Fehler
  einzuführen, was Wartung und Feature-Entwicklung riskant macht.
category:
- Architecture
- Code
related_problems:
- slug: increasing-brittleness
  similarity: 0.75
- slug: fear-of-change
  similarity: 0.7
- slug: fear-of-breaking-changes
  similarity: 0.7
- slug: refactoring-avoidance
  similarity: 0.65
- slug: large-estimates-for-small-changes
  similarity: 0.65
- slug: high-bug-introduction-rate
  similarity: 0.65
solutions:
- technical-debt-backlog
- bubble-context
- fault-tolerant-data-structures
- feature-detection
- plausibility-checks
- resilience
- tolerant-reader
- defect-triage-process
- technical-debt-assessment
- debt-remediation-estimation
- code-hotspot-analysis
- characterization-tests
- dependency-breaking-techniques
- improvement-budget
- quality-ratchet
- debt-classification
- duplication-detection
layout: problem
lang: de
en_slug: brittle-codebase
---

## Description
Eine brüchige Codebasis ist eine, die schwer und riskant zu ändern ist. Wenn eine kleine Änderung in einem Teil der Codebasis zu unerwarteten Ausfällen in anderen Teilen führt, ist das ein Zeichen für eine brüchige Codebasis. Dies wird oft durch das Fehlen automatisierter Tests, einen hohen Grad an Kopplung zwischen Komponenten und einen allgemeinen Mangel an guten Design-Prinzipien verursacht. Eine brüchige Codebasis ist eine bedeutende Quelle technischer Schulden und kann das Entwicklungstempo erheblich verlangsamen.

## Indicators ⟡
- Entwickler äußern Angst oder Zögern, wenn sie gebeten werden, bestimmte Teile des Systems zu ändern.
- Schätzungen für kleine Änderungen sind durchgängig viel größer als erwartet.
- Das Team vermeidet Refactoring, selbst wenn es weiß, dass es nötig ist.
- Das Onboarding neuer Entwickler dauert ungewöhnlich lange, weil die Codebasis so schwer zu verstehen ist.

## Symptoms ▲

- [Regressionsfehler](regressionsfehler.md)
<br/>  Kleine Codeänderungen führen aufgrund versteckter Kopplung häufig zu Fehlern in scheinbar unzusammenhängenden Teilen des Systems.
- [Angst vor Breaking Changes](angst-vor-breaking-changes.md)
<br/>  Entwickler bekommen Angst davor, die Codebasis zu ändern, weil selbst kleinere Änderungen oft unerwartete Ausfälle verursachen.
- [Vermeidungsverhalten](vermeidungsverhalten.md)
<br/>  Entwickler vermeiden es, brüchige Codebereiche anzufassen, was zu aufgeschobener Wartung und wachsenden technischen Schulden führt.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Die Entwicklungsgeschwindigkeit sinkt, da Entwickler übermäßig viel Zeit damit verbringen, brüchigen Code zu umgehen, um Fehlschläge zu vermeiden.
- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Statt brüchigen Code direkt zu ändern, fügen Entwickler Workarounds hinzu, die die Komplexität weiter erhöhen.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Die Wartung einer brüchigen Codebasis erfordert unverhältnismäßigen Aufwand, da kleine Änderungen umfangreiches Testen und Beheben erfordern.

## Causes ▼

- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Hohe Kopplung zwischen Komponenten bedeutet, dass sich Änderungen unvorhersehbar fortpflanzen, was die Codebasis brüchig macht.
- [Unzureichendes Testen](unzureichendes-testen.md)
<br/>  Ohne ausreichende Testabdeckung gibt es kein Sicherheitsnetz, um durch Änderungen eingeführte Regressionen abzufangen.
- [Spaghetticode](spaghetticode.md)
<br/>  Verworrener, unstrukturierter Code mit unklarem Kontrollfluss macht Änderungen riskant und unvorhersehbar.
- [Refactoring-Vermeidung](refactoring-vermeidung.md)
<br/>  Langfristige Vermeidung von Refactoring lässt strukturelle Probleme sich anhäufen, was die Codebasis zunehmend brüchig macht.
- [Legacy-Code ohne Tests](legacy-code-ohne-tests.md)
<br/>  Legacy-Code ohne Tests trägt erheblich zu brüchigen Codebasen bei, da es kein Sicherheitsnetz gibt, um Regressionen abzufangen.

## Detection Methods ○

- **Code-Coverage-Werkzeuge:** Nutzung von Werkzeugen zur Messung der Testabdeckung. Niedrige Abdeckung ist ein starker Indikator für Brüchigkeit.
- **Statische Analysewerkzeuge:** Werkzeuge, die Code-Komplexität (z. B. zyklomatische Komplexität), Kopplung und andere Metriken messen, können problematische Bereiche aufzeigen.
- **Fehler-Tracking-Metriken:** Beobachtung der Rate an Regressionsfehlern, die nach neuen Features oder Änderungen eingeführt werden.
- **Entwickler-Umfragen/-Interviews:** Befragung von Entwicklern zu ihrer Erfahrung mit der Codebasis und ihrem Vertrauen beim Vornehmen von Änderungen.
- **Code-Review-Feedback:** Suche nach wiederkehrenden Kommentaren darüber, dass Code schwer zu verstehen oder riskant zu ändern ist.

## Examples

Ein Team muss ein kleines Stück Geschäftslogik in einem Legacy-System aktualisieren. Die Änderung wird auf wenige Stunden geschätzt, aber weil der Code so eng gekoppelt ist und keine Tests hat, verbringt das Team zwei Wochen damit, die Änderung umzusetzen und alle neuen Fehler zu beheben, die sie verursacht. Zum Beispiel aktualisiert eine Funktion, die den Rabatt eines Nutzers berechnet, auch dessen Treuepunkte und versendet eine E-Mail. Das Ändern der Rabattberechnungslogik bricht unerwartet die E-Mail-Versandfunktion, weil die Funktion zu viele Verantwortlichkeiten hat. Dieses Problem ist ein Kennzeichen alternder, unzureichend gewarteter Softwaresysteme. Es entsteht oft aus einem Mangel an Disziplin bei Software-Engineering-Praktiken, besonders bei Test- und Design-Prinzipien, über einen langen Zeitraum.
