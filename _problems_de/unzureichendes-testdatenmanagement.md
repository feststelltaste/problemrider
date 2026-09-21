---
title: Unzureichendes Testdatenmanagement
description: Die Nutzung unrealistischer, veralteter oder unzureichender Testdaten
  führt zu Tests, die reale Szenarien nicht akkurat widerspiegeln.
category:
- Code
- Process
related_problems:
- slug: insufficient-testing
  similarity: 0.7
- slug: high-defect-rate-in-production
  similarity: 0.65
- slug: incomplete-projects
  similarity: 0.6
- slug: gold-plating
  similarity: 0.6
- slug: misunderstanding-of-oop
  similarity: 0.6
- slug: inconsistent-behavior
  similarity: 0.6
solutions:
- test-coverage-strategy
- mass-test-data-generation
- sampling
- simulation-environments
- production-like-test-data
- isolated-test-environments
- containerized-databases
- datensparsamkeit
- data-archiving
layout: problem
lang: de
en_slug: inadequate-test-data-management
---

## Description
Unzureichendes Testdatenmanagement ist die Praxis, Testdaten zu nutzen, die nicht repräsentativ für die Produktionsumgebung sind. Dies kann zu einer Reihe von Problemen führen, einschließlich Tests, die bestehen, wenn sie fehlschlagen sollten, und Tests, die fehlschlagen, wenn sie bestehen sollten. Es kann auch zu einem falschen Sicherheitsgefühl führen, da die Tests den Code möglicherweise nicht auf dieselbe Weise beanspruchen, wie er in Produktion beansprucht wird. Eine gute Testdatenmanagement-Strategie ist essenziell, um die Qualität und Zuverlässigkeit eines Softwareprodukts sicherzustellen.

## Indicators ⟡
- Das Team nutzt Produktionsdaten zum Testen.
- Das Team erstellt manuell Testdaten für jeden Testlauf.
- Das Team kann in Produktion gefundene Fehler nicht konsistent reproduzieren.
- Das Team kann bestimmte Randfälle nicht testen, weil ihm die Daten dafür fehlen.

## Symptoms ▲

- [Flaky Tests](flaky-tests.md)
<br/>  Inkonsistente oder unzuverlässige Testdaten lassen Tests unvorhersehbar bestehen oder fehlschlagen, was das Vertrauen in die Testsuite untergräbt.
- [Qualitäts-blinde Flecken](qualitaets-blinde-flecken.md)
<br/>  Unzureichende Testdaten lassen Randfälle und reale Szenarien ungetestet, was blinde Flecken in der Qualitätssicherung schafft.
- [Erhöhter manueller Testaufwand](erhoehter-manueller-testaufwand.md)
<br/>  Wenn automatisierten Tests aufgrund schlechter Daten nicht vertraut werden kann, greifen Teams auf manuelles Testen zurück, um Verhalten zu verifizieren.

## Causes ▼

- [Unzureichende Testinfrastruktur](unzureichende-testinfrastruktur.md)
<br/>  Fehlende ordentliche Infrastruktur zur Erzeugung, Verwaltung und Aktualisierung von Testdaten macht realistisches Datenmanagement unpraktikabel.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Das Management priorisiert Feature-Lieferung über Investitionen in ordentliche Testdatenmanagement-Prozesse und -Werkzeuge.
- [Schlechte Planung](schlechte-planung.md)
<br/>  Testdatenbedürfnisse werden nicht geplant oder budgetiert, was zu Ad-hoc- und unzureichenden Datenmanagement-Praktiken führt.

## Detection Methods ○
- **Testdatenanalyse:** Analyse der Testdaten, um zu sehen, ob sie realistisch und repräsentativ für die Produktionsumgebung sind.
- **Fehler-Triage:** Wenn ein Fehler in Produktion gefunden wird, Analyse der Testdaten, die zum Testen des Features genutzt wurden, um zu sehen, ob sie angemessen waren.
- **Entwicklerbefragungen:** Befragung von Entwicklern zu ihrem Vertrauen in die Testdaten und den Testdatenmanagement-Prozess.

## Examples
Ein Team entwickelt ein neues Feature für eine E-Commerce-Anwendung. Sie nutzen einen kleinen, manuell erstellten Datensatz zum Testen. Das Feature funktioniert in der Testumgebung perfekt, aber als es in Produktion deployt wird, schlägt es für eine große Anzahl von Nutzern fehl. Das Problem ist, dass die Testdaten keine Nutzer mit Sonderzeichen in ihren Namen enthielten, was dazu führte, dass das Feature fehlschlug. In einem anderen Beispiel nutzt ein Team eine bereinigte Version von Produktionsdaten zum Testen. Der Bereinigungsprozess ist jedoch nicht perfekt und führt eine Reihe von Inkonsistenzen in die Daten ein. Dies führt zu einer Reihe von Flaky Tests, was es dem Team schwer macht, Vertrauen in ihre Testergebnisse zu haben.
