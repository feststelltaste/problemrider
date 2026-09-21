---
title: Regressionsfehler
description: Neue Features oder Korrekturen brechen unbeabsichtigt bestehende Funktionalität,
  die zuvor korrekt funktionierte.
category:
- Code
- Process
- Testing
related_problems:
- slug: high-bug-introduction-rate
  similarity: 0.65
- slug: increased-bug-count
  similarity: 0.65
- slug: delayed-bug-fixes
  similarity: 0.6
- slug: breaking-changes
  similarity: 0.6
- slug: increased-risk-of-bugs
  similarity: 0.6
- slug: frequent-hotfixes-and-rollbacks
  similarity: 0.6
solutions:
- test-coverage-strategy
- acceptance-tests
- automated-tests
- backward-compatibility
- behavior-driven-development-bdd
- business-test-cases
- code-coverage-analysis
- compatibility-as-error
- compatibility-testing
- continuous-integration
- cross-version-testing
- environment-parity
- functional-tests
- integration-tests
- mutation-testing
- property-based-testing
- regression-tests
- root-cause-analysis
- smoke-testing
- test-driven-development-tdd
- value-range-definition
- code-quality-gates
- characterization-tests
- change-impact-analysis
- parallel-run
- production-like-test-data
- defect-triage-process
- exploratory-testing
- duplication-detection
- explicit-extension-points
- customization-under-version-control
layout: problem
lang: de
en_slug: regression-bugs
---

## Description

Regressionsfehler sind Defekte, die auftreten, wenn zuvor funktionierende Funktionalität aufgrund neuer Codeänderungen, Feature-Ergänzungen oder Fehlerbehebungen bricht. Diese Fehler stellen eine erhebliche Bedrohung für die Softwarequalität dar, weil sie Nutzervertrauen untergraben und Probleme wieder einführen können, die als gelöst galten. Regressionsfehler sind besonders problematisch, weil sie oft unentdeckt bleiben, bis Nutzer auf sie in Produktion stoßen, und sie deuten auf fundamentale Probleme mit Testpraktiken und Codewartbarkeit hin.

## Indicators ⟡
- Nutzer berichten, dass Features, die früher funktionierten, jetzt defekt sind
- Zuvor bestandene Tests beginnen nach neuen Deployments zu scheitern
- Der Kundensupport erhält Beschwerden über Funktionalität, die in vorherigen Versionen funktionierte
- Qualitätssicherung entdeckt häufig, dass die Behebung eines Fehlers einen anderen einführt
- Das Team diskutiert regelmäßig, ob Änderungen „etwas anderes brechen" könnten

## Symptoms ▲

- [Erhöhte Fehleranzahl](erhoehte-fehleranzahl.md)
<br/>  Regressionsfehler tragen zur Gesamtfehleranzahl bei, während zuvor behobene Probleme neben neuen Defekten wieder auftauchen.
- [Kundenunzufriedenheit](kundenunzufriedenheit.md)
<br/>  Nutzer verlieren Vertrauen, wenn Features, auf die sie sich verließen, nach Updates brechen, was zu Frustration und Unzufriedenheit führt.
- [Refactoring-Vermeidung](refactoring-vermeidung.md)
<br/>  Häufige Regressionen verstärken die Angst, Änderungen vorzunehmen, was Teams dazu bringt, Refactoring oder Modifikation von Code zu vermeiden.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Wiederholte Erfahrungen, dass Änderungen bestehende Funktionalität brechen, schaffen eine Kultur der Angst rund um Codemodifikationen.
- [Ständiges Feuerlöschen](staendiges-feuerloeschen.md)
<br/>  Regressionen in Produktion erfordern dringende Korrekturen, was das Team in reaktiven Feuerlösch-Modus zieht.

## Causes ▼

- [Testschulden](testschulden.md)
<br/>  Unzureichende Testabdeckung versäumt es, Regressionen vor dem Deployment zu erfassen, was ihnen erlaubt, Produktion zu erreichen.
- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Eng gekoppelte Komponenten bedeuten, dass Änderungen in einem Bereich unerwartet scheinbar nicht zusammenhängende Funktionalität beeinflussen.
- [Brüchige Codebasis](bruechige-codebasis.md)
<br/>  Eine fragile Codebasis mit schlechter Struktur macht es leicht, dass Änderungen unbeabsichtigt bestehende Funktionalität brechen.
- [Unzureichende Code-Reviews](unzureichende-code-reviews.md)
<br/>  Schlechte Code-Reviews versäumen es, Änderungen zu identifizieren, die bestehende Funktionalität brechen könnten, bevor sie gemerged werden.
- [Teilweise Fehlerbehebungen](teilweise-fehlerbehebungen.md)
<br/>  Teilweise Fehlerbehebungen, die Grundursachen nicht angehen, sind eine direkte Ursache für Regressionsfehler, da das zugrunde liegende Problem wieder auftaucht.

## Detection Methods ○
- **Automatisierte Regressionstestsuiten:** Umfassende automatisierte Tests, die bestehende Funktionalität nach jeder Änderung verifizieren
- **User Acceptance Testing:** Systematisches Testen wichtiger Nutzer-Workflows vor Releases
- **Produktionsüberwachung:** Echtzeitüberwachung des Systemverhaltens, um Regressionen schnell zu erfassen
- **A/B-Testing:** Schrittweise Rollouts, die Regressionen vor vollständigem Deployment erkennen können
- **Fehlerkategorisierung:** Nachverfolgung und Kategorisierung von Fehlern zur Identifikation von Regressionsmustern

## Examples

Ein Team fügt seinem Warenkorb ein neues Feature hinzu, das Nutzern erlaubt, Artikel für später zu speichern. Während der Implementierung modifizieren sie die Warenkorb-Persistenzlogik, um die neue Funktionalität zu unterstützen. Nach dem Deployment entdecken Nutzer, dass ihre Warenkorb-Inhalte nicht mehr erhalten bleiben, wenn sie sich abmelden und wieder anmelden – ein Kern-Feature, das jahrelang einwandfrei funktioniert hatte. Die Regression trat auf, weil das neue „für später speichern"-Feature die Datenstruktur änderte, die zur Speicherung von Warenkorb-Artikeln genutzt wurde, aber die bestehende Warenkorb-Ladelogik wurde nicht aktualisiert, um das neue Format zu handhaben. Die automatisierten Tests erfassten dies nicht, weil sie nur den Happy Path des neuen Features testeten, nicht die bestehende Warenkorb-Funktionalität. Ein weiteres Beispiel betrifft eine Banking-Anwendung, bei der ein Sicherheitspatch zur Verhinderung von SQL-Injection unbeabsichtigt die Anzeige der Transaktionshistorie für Konten mit bestimmten Sonderzeichen in ihren Namen bricht, was den Kundenservice mit Anrufen von Nutzern überflutet, die nicht auf ihre Transaktionshistorie zugreifen können.
