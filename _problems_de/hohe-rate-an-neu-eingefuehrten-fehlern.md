---
title: Hohe Rate an neu eingeführten Fehlern
description: Mit jeder Änderung an der Codebasis wird eine hohe Anzahl neuer Fehler
  eingeführt, was auf zugrunde liegende Qualitätsprobleme hindeutet.
category:
- Code
related_problems:
- slug: increased-bug-count
  similarity: 0.8
- slug: increased-risk-of-bugs
  similarity: 0.75
- slug: high-defect-rate-in-production
  similarity: 0.7
- slug: large-estimates-for-small-changes
  similarity: 0.7
- slug: increased-error-rates
  similarity: 0.7
- slug: increased-cost-of-development
  similarity: 0.65
solutions:
- definition-of-done
- test-coverage-strategy
- automated-tests
- code-reviews
- continuous-integration
- functional-tests
- regression-tests
- secure-software-development
- static-code-analysis
- test-driven-development-tdd
- code-quality-gates
layout: problem
lang: de
en_slug: high-bug-introduction-rate
---

## Description
Eine hohe Rate an neu eingeführten Fehlern bedeutet, dass mit jedem neuen Feature oder Fix eine erhebliche Anzahl neuer Fehler entsteht. Dies ist ein starker Indikator für eine brüchige und ungesunde Codebasis. Es verlangsamt die Entwicklung, untergräbt das Vertrauen in die Software und erhöht die Wartungskosten. Dieses Problem ist oft ein Symptom tieferer Probleme im Entwicklungsprozess und der Codequalität.

## Indicators ⟡
- Die Anzahl der Fehlerberichte steigt nach jedem Release.
- Entwickler verbringen mehr Zeit mit dem Beheben neuer Fehler als mit dem Bauen neuer Features.
- Die "Bugs"-Spalte auf dem Kanban-Board des Teams ist immer voll.
- Es gibt ein Gefühl von "ein Schritt vorwärts, zwei Schritte zurück" im Entwicklungsprozess.

## Symptoms ▲

- [Ständiges Feuerlöschen](staendiges-feuerloeschen.md)
<br/>  Das Team verbringt den Großteil seiner Zeit damit, den Strom neu eingeführter Fehler zu beheben, statt an geplanten Features zu arbeiten.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Der kontinuierliche Kreislauf aus Einführen und Beheben von Fehlern verringert die Netto-Produktivität des Teams erheblich.
- [Hohe Fehlerrate in Produktion](hohe-fehlerrate-in-produktion.md)
<br/>  Eine hohe Rate an während der Entwicklung eingeführten Fehlern führt natürlich dazu, dass mehr Defekte in der Produktionsumgebung landen.
- [Kundenunzufriedenheit](kundenunzufriedenheit.md)
<br/>  Nutzer erleben häufige Fehler in Releases, was ihr Vertrauen in die Zuverlässigkeit des Produkts untergräbt.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Das ständige Beheben von Fehlern, die sie oder Kollegen eingeführt haben, demoralisiert Entwickler und führt zu Burnout.
- [Anstieg der Wartungskosten](anstieg-der-wartungskosten.md)
<br/>  Jeder eingeführte Fehler erfordert Untersuchung, Behebung, Testen und Deployment, was die Gesamtwartungskosten erhöht.

## Causes ▼

- [Brüchige Codebasis](bruechige-codebasis.md)
<br/>  Brüchiger Code, der leicht bei kleinen Änderungen bricht, ist die Hauptursache dafür, dass bei jeder Modifikation neue Fehler eingeführt werden.
- [Schlechte Testabdeckung](schlechte-testabdeckung.md)
<br/>  Ohne ausreichende Testabdeckung bleiben durch Änderungen eingeführte Fehler unentdeckt, bis sie die Produktion erreichen.
- [Hohe Kopplung und geringe Kohäsion](hohe-kopplung-und-geringe-kohaesion.md)
<br/>  Eng gekoppelter Code bedeutet, dass Änderungen in einem Bereich unerwartete Auswirkungen in anderen Bereichen haben, was Fehler in scheinbar unzusammenhängenden Teilen einführt.
- [Versteckte Abhängigkeiten](versteckte-abhaengigkeiten.md)
<br/>  Undokumentierte Abhängigkeiten zwischen Komponenten führen dazu, dass Entwickler unwissentlich Funktionalität brechen, wenn sie Änderungen vornehmen.
- [Legacy-Code ohne Tests](legacy-code-ohne-tests.md)
<br/>  Die Modifikation von ungetestetem Legacy-Code ist inhärent riskant und führt häufig zu Regressionen, die von Tests erfasst worden wären.

## Detection Methods ○
- **Fehlerverfolgungsmetriken:** Beobachtung der Anzahl neuer nach jedem Release gemeldeter Fehler.
- **Code-Churn-Analyse:** Analyse, wie oft eine Datei geändert wird. Hoher Churn kann auf problematische Bereiche hindeuten.
- **Entwicklerfeedback:** Regelmäßiges Einholen von Feedback vom Entwicklungsteam zur Qualität der Codebasis und des Entwicklungsprozesses.

## Examples
Ein Team veröffentlicht eine neue Version seiner Software mit einigen neuen Features. Innerhalb einer Woche hat sich die Anzahl der Fehlerberichte von Nutzern verdoppelt. Das Team verbringt die nächsten zwei Sprints damit, diese neuen Fehler zu beheben, was den Start der nächsten geplanten Features verzögert. Dieser Kreislauf wiederholt sich mit jedem Release, was zu einem langsamen und unvorhersehbaren Entwicklungsprozess führt.
