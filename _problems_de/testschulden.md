---
title: Testschulden
description: Das angehäufte Risiko aus unzureichender oder vernachlässigter Qualitätssicherung,
  was zu einem fragilen Produkt und langsamer Entwicklungsgeschwindigkeit führt.
category:
- Code
- Process
related_problems:
- slug: high-technical-debt
  similarity: 0.7
- slug: quality-blind-spots
  similarity: 0.65
- slug: insufficient-testing
  similarity: 0.65
- slug: poor-test-coverage
  similarity: 0.65
- slug: testing-environment-fragility
  similarity: 0.65
- slug: invisible-nature-of-technical-debt
  similarity: 0.65
solutions:
- test-coverage-strategy
- automated-tests
- code-coverage-analysis
- regression-tests
- characterization-tests
- dependency-breaking-techniques
- code-quality-gates
- improvement-budget
- production-like-test-data
layout: problem
lang: de
en_slug: test-debt
---

## Description

Testschulden sind das angehäufte Risiko, das aus unzureichenden oder vernachlässigten Qualitätssicherungsaktivitäten resultiert. Sie gehen weit über fehlende Unit-Tests hinaus und umfassen unzureichende Integrationstests, oberflächliche End-to-End-Tests, ignorierte nicht-funktionale Tests (Performance, Sicherheit) und das Fehlen strukturierten manuellen oder explorativen Testens. Diese Schulden werden oft aufgenommen, um Features schneller zu veröffentlichen, indem bei der Qualität Abstriche gemacht werden, was ein fragiles Produkt schafft, bei dem Änderungen riskant sind und die tatsächliche Qualität unbekannt ist.

## Indicators ⟡

- Das Team hat kein klares, gemeinsames Verständnis der aktuellen Teststrategie.
- Manuelles Regressionstesten vor einem Release ist ein langwieriges und stressiges Ereignis.
- Entwickler zögern, Code zu refaktorieren, weil sie Angst haben, unerwartet etwas zu brechen.
- Bugs, die intern hätten erfasst werden sollen, werden häufig von Nutzern gemeldet.
- Die Phrase „Die Tester werden es schon fangen" wird genutzt, um das Fortfahren mit unverifiziertem Code zu rechtfertigen.

## Symptoms ▲

- [Hohe Fehlerrate in Produktion](hohe-fehlerrate-in-produktion.md)
<br/>  Unzureichende Testabdeckung erlaubt es Bugs, in Produktion zu gelangen, die während der Qualitätssicherung hätten erfasst werden sollen.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Ohne angemessene Tests haben Entwickler Angst, Code zu refaktorieren oder zu modifizieren, weil sie nicht verifizieren können, dass sie nichts gebrochen haben.
- [Verzögerte Wertlieferung](verzoegerte-wertlieferung.md)
<br/>  Langwieriges manuelles Regressionstesten und mangelnde automatisierte Verifikation verlangsamen den Release-Zyklus.
- [Qualitätsverschlechterung](qualitaetsverschlechterung.md)
<br/>  Das angehäufte Fehlen von Testen führt zu progressivem Qualitätsverfall, während unentdeckte Probleme sich über die Zeit verstärken.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Ohne angemessene Tests führen Code-Änderungen häufig zu Regressionsfehlern, die unentdeckt bleiben, was eine direkte und offensichtliche Folge von Testschulden ist.

## Causes ▼

- [Zeitdruck](zeitdruck.md)
<br/>  Der Druck, Features schnell zu liefern, führt dazu, dass Teams Testaktivitäten überspringen oder aufschieben, um Termine einzuhalten.
- [Unzureichendes Testen](unzureichendes-testen.md)
<br/>  Ein Muster, keine angemessenen Tests zu schreiben, häuft sich über die Zeit zu Testschulden an.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Die Priorisierung unmittelbarer Lieferung über langfristige Qualität führt zu konsequenten Abstrichen beim Testen.

## Detection Methods ○

- **Testabdeckungsanalyse:** Nutzung von Werkzeugen zur Messung von Zeilen-, Zweig- und Funktionsabdeckung, aber kritische Interpretation der Ergebnisse.
- **Verfolgung des Bug-Ursprungs:** Analyse, wo Bugs gefunden werden. Ein hoher Prozentsatz von in Produktion gefundenen Bugs ist ein klares Zeichen für Testschulden.
- **Zykluszeitmessung:** Verfolgung der Zeit vom Code-Commit bis zum Produktions-Deployment. Lange, unvorhersehbare Testphasen deuten auf Probleme hin.
- **Team-Vertrauensbefragungen:** Anonyme Befragung des Teams zu ihrem Vertrauensniveau für das bevorstehende Release.
- **Explorative Testsitzungen:** Widmen von Zeit für strukturiertes, nicht skriptbasiertes Testen zur Aufdeckung unerwarteter Probleme.

## Examples

Ein Team steht unter Druck, einen neuen E-Commerce-Checkout-Flow zu veröffentlichen. Um den Termin einzuhalten, schreiben sie einige grundlegende Unit-Tests, überspringen aber die Erstellung von Integrationstests für die Zahlungsgateway- und Versandanbieter-APIs. Sie verschieben auch Performance-Tests in der Annahme, dass das System die Last handhaben wird. Das Feature wird „pünktlich" veröffentlicht, aber bald darauf melden Kunden, dass ein bestimmter Kreditkartenanbieter fehlschlägt — ein Problem, das ein Integrationstest erfasst hätte. Während eines Verkaufsereignisses wird das System quälend langsam und stürzt ab, was erheblichen Umsatz kostet. Das Team muss nun alle neue Feature-Arbeit fallen lassen, um dringend Produktionsprobleme zu beheben und rückwirkend die Tests zu bauen, die sie übersprungen haben, wobei sie ihre Testschulden mit hohen Zinsen zurückzahlen.
