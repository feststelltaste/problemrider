---
title: Schlechte Testabdeckung
description: Kritische Teile der Codebasis sind nicht von Tests abgedeckt, was blinde
  Flecken in der Qualitätssicherung schafft.
category:
- Code
- Process
- Testing
related_problems:
- slug: quality-blind-spots
  similarity: 0.75
- slug: insufficient-testing
  similarity: 0.7
- slug: inadequate-code-reviews
  similarity: 0.65
- slug: legacy-code-without-tests
  similarity: 0.65
- slug: insufficient-code-review
  similarity: 0.65
- slug: test-debt
  similarity: 0.65
solutions:
- definition-of-done
- test-coverage-strategy
- acceptance-tests
- automated-tests
- behavior-driven-development-bdd
- business-test-cases
- code-coverage-analysis
- functional-tests
- integration-tests
- mutation-testing
- platform-independent-test-frameworks
- property-based-testing
- regression-tests
- requirements-traceability-matrix
- security-tests
- test-driven-development-tdd
- characterization-tests
- production-like-test-data
- exploratory-testing
- quality-ratchet
- debt-remediation-estimation
layout: problem
lang: de
en_slug: poor-test-coverage
---

## Description

Schlechte Testabdeckung tritt auf, wenn erhebliche Teile der Codebasis, besonders kritische Funktionalität, angemessene automatisierte Tests vermissen lassen. Dies schafft Lücken in der Qualitätssicherung, in denen sich Fehler unentdeckt verstecken können, bis sie Produktion erreichen. Schlechte Abdeckung bedeutet nicht nur niedrige Prozentzahlen – sie bezieht sich speziell auf das Fehlen von Tests für wichtige Geschäftslogik, Fehlerbehandlungspfade, Randfälle und Integrationspunkte, die für Systemzuverlässigkeit entscheidend sind.

## Indicators ⟡
- Code-Abdeckungsberichte zeigen niedrige Prozentzahlen, besonders in kritischen Geschäftslogikbereichen
- Produktionsfehler treten häufig in Bereichen mit wenig oder keiner Testabdeckung auf
- Entwickler sind unsicher, ob Änderungen bestehende Funktionalität brechen werden
- Kritische Systemkomponenten haben keine automatisierten Tests
- Fehlerbehandlung und Randfälle werden selten getestet

## Symptoms ▲

- [Hohe Fehlerrate in Produktion](hohe-fehlerrate-in-produktion.md)
<br/>  Ungetestete Codepfade erlauben es Fehlern, unentdeckt Produktion zu erreichen, was die Produktionsdefektrate erhöht.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Ohne Testabdeckung als Sicherheitsnetz fürchten Entwickler, Änderungen vorzunehmen, die ungetestete Funktionalität brechen könnten.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Das Fehlen automatisierter Tests bedeutet, dass Regressionen während der Entwicklung nicht erfasst werden und später in Produktion auftauchen.
- [Erhöhter manueller Testaufwand](erhoehter-manueller-testaufwand.md)
<br/>  Lücken in der automatisierten Testabdeckung müssen durch umfangreiches manuelles Testen kompensiert werden, was langsam und fehleranfällig ist.
- [Refactoring-Vermeidung](refactoring-vermeidung.md)
<br/>  Ohne Tests zur Verifikation der Korrektheit vermeiden Entwickler Refactoring aus Angst, unentdeckte Fehler einzuführen.
- [Ständiges Feuerlöschen](staendiges-feuerloeschen.md)
<br/>  Schlechte Testabdeckung erlaubt es Fehlern, Produktion zu erreichen, was ständiges Feuerlöschen verursacht.
- [Qualitäts-blinde Flecken](qualitaets-blinde-flecken.md)
<br/>  Lücken in der Testabdeckung schaffen direkt Bereiche, in denen Defekte unentdeckt bleiben.

## Causes ▼

- [Termindruck](termindruck.md)
<br/>  Zeitdruck führt dazu, dass Teams das Schreiben von Tests überspringen, um Features schneller zu liefern.
- [Schwer testbarer Code](schwer-testbarer-code.md)
<br/>  Code, der eng gekoppelt ist oder versteckte Abhängigkeiten hat, ist von Natur aus schwer zu testen, was die Testerstellung entmutigt.
- [Legacy-Code ohne Tests](legacy-code-ohne-tests.md)
<br/>  Übernommene Legacy-Codebasen ohne Tests machen es sehr schwierig, Abdeckung inkrementell hinzuzufügen.
- [Unzureichende Testinfrastruktur](unzureichende-testinfrastruktur.md)
<br/>  Das Fehlen ordentlicher Testwerkzeuge und -infrastruktur macht das Schreiben und Ausführen von Tests prohibitiv schwierig.

## Detection Methods ○
- **Code-Abdeckungsanalyse:** Nutzung von Werkzeugen zur Messung, welcher Prozentsatz des Codes von Tests ausgeführt wird
- **Identifikation kritischer Pfade:** Kartierung geschäftskritischer Funktionalität und Bewertung ihrer Testabdeckung
- **Fehlerquellenanalyse:** Nachverfolgung, ob Produktionsfehler in getesteten vs. ungetesteten Codebereichen auftreten
- **Überwachung des Abdeckungstrends:** Nachverfolgung, ob sich die Testabdeckung über die Zeit verbessert, verschlechtert oder stagniert
- **Abhängigkeit von manuellem Testen:** Identifikation von Bereichen, die aufgrund fehlender Automatisierung stark auf manuelles Testen angewiesen sind

## Examples

Eine Finanzhandelsanwendung hat 40 % Gesamttestabdeckung, aber Analyse offenbart, dass die Kern-Risikoberechnungsalgorithmen – verantwortlich für die Verhinderung katastrophaler Handelsverluste – nur 15 % Testabdeckung haben. Die bestehenden Tests decken nur grundlegende Szenarien mit kleinen Handelsbeträgen ab, aber die komplexe Logik, die große Trades, Margin-Anforderungen und Risikolimits während Marktvolatilität handhabt, ist völlig ungetestet. Wenn sich Marktbedingungen unerwartet ändern, versagt der ungetestete Risikoberechnungscode dabei, das Exposure ordentlich zu begrenzen, was zu erheblichen finanziellen Verlusten führt, die durch umfassendes Testen von Randfällen und Stressszenarien hätten verhindert werden können. Ein weiteres Beispiel betrifft eine E-Commerce-Plattform, bei der das Zahlungsverarbeitungsmodul 80 % Zeilenabdeckung hat, aber 0 % Abdeckung der Fehlerbehandlungspfade. Während normale Zahlungsflüsse gut getestet sind, wird der Code, der abgelehnte Karten, Netzwerk-Timeouts, Teilzahlungen und Rückerstattungsszenarien handhabt, nie von Tests ausgeführt. Wenn Probleme mit dem Zahlungsgateway auftreten, erleben Kunden verlorene Transaktionen, Doppelbelastungen und fehlgeschlagene Rückerstattungen, weil der Fehlerbehandlungscode Fehler enthält, die während des Testens nie erfasst wurden.
