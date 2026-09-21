---
title: Flaky Tests
description: Tests schlagen zufällig fehl aufgrund von Timing, Setup oder Abhängigkeiten,
  was das Vertrauen in die Testsuite untergräbt.
category:
- Code
- Process
related_problems:
- slug: testing-environment-fragility
  similarity: 0.7
- slug: outdated-tests
  similarity: 0.65
- slug: configuration-chaos
  similarity: 0.6
- slug: difficult-to-test-code
  similarity: 0.55
- slug: inadequate-test-infrastructure
  similarity: 0.55
- slug: test-debt
  similarity: 0.55
solutions:
- test-coverage-strategy
- isolated-test-environments
- platform-independent-test-frameworks
- dependency-breaking-techniques
- containerization
- mass-test-data-generation
- ci-cd-pipeline
- characterization-tests
- production-like-test-data
- fast-feedback-loops
layout: problem
lang: de
en_slug: flaky-tests
---

## Description

Flaky Tests sind automatisierte Tests, die bei mehrfacher Ausführung gegen denselben Code widersprüchliche Ergebnisse liefern und manchmal bestehen, manchmal fehlschlagen, ohne dass sich an der Codebasis etwas geändert hat. Diese Tests untergraben das Vertrauen in die gesamte Testsuite und erschweren es, zwischen echten Regressionen und falsch-positiven Ergebnissen zu unterscheiden. Im Laufe der Zeit beginnen Teams, Testfehlschläge zu ignorieren oder Flaky Tests zu deaktivieren, was die Wirksamkeit automatisierten Testens als Sicherheitsnetz für Codeänderungen verringert.

## Indicators ⟡

- Tests, die gelegentlich bei der Continuous Integration fehlschlagen, aber lokal bestehen
- Teammitglieder führen fehlgeschlagene Testsuiten regelmäßig erneut aus, um zu sehen, ob sie beim zweiten Mal bestehen
- Tests, die bei hoher Systemlast oder zu bestimmten Tageszeiten häufiger fehlschlagen
- Intermittierende Testfehlschläge, die sich schwer konsistent reproduzieren lassen
- Tests, die von externen Diensten oder Netzwerkverbindung abhängen
- Test-Setup- oder Teardown-Prozesse, die den Systemzustand nicht konsistent zurücksetzen
- Tests mit fest codierten Timing-Annahmen oder Sleep-Anweisungen

## Symptoms ▲

- [Erhöhter manueller Testaufwand](erhoehter-manueller-testaufwand.md)
<br/>  Wenn automatisierte Tests unzuverlässig sind, kompensieren Teams durch verstärktes manuelles Testen, um Regressionen zu erfassen.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Entwickler verschwenden Zeit mit erneutem Ausführen von Testsuiten, dem Untersuchen falscher Fehlschläge und verlieren das Vertrauen in automatisiertes Testen.
- [Qualitäts-blinde Flecken](qualitaets-blinde-flecken.md)
<br/>  Wenn Flaky Tests deaktiviert oder ignoriert werden, entstehen Lücken in der Testabdeckung, in denen sich echte Fehler unentdeckt verstecken können.
- [Review-Engpässe](review-engpaesse.md)
<br/>  Durch Flaky-Test-Fehlschläge blockierte CI-Pipelines verzögern Code-Review- und Merge-Prozesse.
- [Lange Build- und Testzeiten](lange-build-und-testzeiten.md)
<br/>  Flaky Tests führen zu längeren Build-Zeiten, weil Entwickler Testsuiten mehrfach erneut ausführen und CI-Pipelines blockiert werden.

## Causes ▼

- [Brüchigkeit der Testumgebung](bruechigkeit-der-testumgebung.md)
<br/>  Unzuverlässige Testinfrastruktur führt dazu, dass Tests je nach Umgebungsbedingungen unterschiedliche Ergebnisse liefern.
- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Tests, die an externe Dienste, gemeinsamen Zustand oder andere Tests gekoppelt sind, liefern aufgrund von Umgebungsabhängigkeiten inkonsistente Ergebnisse.
- [Schwer testbarer Code](schwer-testbarer-code.md)
<br/>  Code, der sich schwer isoliert testen lässt, zwingt Tests dazu, von Timing, externen Diensten oder gemeinsamem Zustand abhängig zu sein, was Flakiness schafft.
- [Unzureichendes Testdatenmanagement](unzureichendes-testdatenmanagement.md)
<br/>  Unrealistische oder inkonsistente Testdaten führen dazu, dass Tests über verschiedene Durchläufe hinweg unterschiedliche Ergebnisse liefern.

## Detection Methods ○

- Nachverfolgung von Testfehlerraten und -mustern über die Zeit, um inkonsistente Tests zu identifizieren
- Mehrfache aufeinanderfolgende Ausführung von Testsuiten, um nicht-deterministisches Verhalten zu identifizieren
- Überwachung von CI/CD-Pipeline-Metriken für Tests, die fehlschlagen und dann ohne Codeänderungen bestehen
- Nutzung von Werkzeugen zur Erkennung von Test-Flakiness, die historische Testergebnisse analysieren
- Implementierung von Test-Quarantäne-Systemen, die unzuverlässige Tests markieren
- Überprüfung des Testcodes auf Timing-Abhängigkeiten, externe Serviceaufrufe und gemeinsamen Zustand
- Analyse von Testfehlschlägen nach Tageszeit, Systemlast oder Umgebungsfaktoren

## Examples

Die Testsuite einer Webanwendung enthält einen Integrationstest, der die Funktionalität der Nutzerregistrierung verifiziert. Der Test erstellt ein Nutzerkonto, sendet eine Bestätigungs-E-Mail und prüft, dass das Konto aktiv wird. Manchmal schlägt der Test jedoch fehl, weil er nicht lange genug wartet, bis der E-Mail-Dienst die Anfrage verarbeitet hat, bevor der Kontostatus geprüft wird. In schnellen Testumgebungen besteht der Test, aber auf langsameren Systemen oder bei hoher Last dauert die E-Mail-Verarbeitung länger, und der Test schlägt fehl. Das Team ignoriert diese Fehlschläge zunächst als "Umgebungsprobleme", aber im Laufe der Zeit entwickeln mehr Tests ähnliche Timing-Probleme. Schließlich verliert das Team das Vertrauen in die Testsuite und verlässt sich zunehmend auf manuelles Testen, wobei mehrere echte Fehler übersehen werden, die automatisierte Tests hätten erfassen können.
