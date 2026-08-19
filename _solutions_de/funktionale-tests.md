---
title: Funktionale Tests
description: Verifikation der Softwarefunktionalität durch systematisches Testen.
category:
- Testing
problems:
- legacy-code-without-tests
- insufficient-testing
- poor-test-coverage
- regression-bugs
- high-defect-rate-in-production
- fear-of-breaking-changes
- increased-risk-of-bugs
- high-bug-introduction-rate
- inconsistent-behavior
- unpredictable-system-behavior
layout: solution
lang: de
en_slug: functional-tests
related_solutions:
- slug: test-coverage-strategy
  similarity: 0.85
- slug: automated-tests
  similarity: 0.8
- slug: characterization-tests
  similarity: 0.75
- slug: acceptance-tests
  similarity: 0.75
- slug: regression-testing
  similarity: 0.75
- slug: negative-testing
  similarity: 0.75
---

## Description

Funktionale Tests verifizieren, dass sich ein System aus der Perspektive seiner Geschäftsworkflows korrekt verhält, indem sie vollständige Szenarien durchspielen — eine Bestellung aufgeben, einen Schaden berechnen, einen Bericht erzeugen — statt isolierter Codeeinheiten. In Legacy-Systemen, in denen ursprüngliche Spezifikationen oft verloren sind und die Autoren, die das beabsichtigte Verhalten verstanden, weitergezogen sind, übernehmen funktionale Tests eine zweite Rolle über die Defekterkennung hinaus: Sie werden zum nächstbesten verfügbaren Ersatz für Dokumentation und erfassen, was das System tun soll, in ausführbarer, eindeutiger Form. Dies überschneidet sich mit Characterization Testing, das erfasst, was der Code tatsächlich tut, unabhängig davon, ob dieses Verhalten beabsichtigt war, und die beiden Techniken werden häufig kombiniert, wenn mit undokumentiertem Legacy-Code gearbeitet wird — Characterization Tests verankern zuerst das aktuelle Verhalten, und funktionale Tests werden dann darübergelegt, um die Geschäftsregeln auszudrücken, die dieses Verhalten erfüllen soll. Weil Legacy-Codebasen wegen enger Kopplung oft unklare oder nicht existente Unit-Grenzen haben, sind funktionale Tests, die auf der End-to-End-Geschäftsszenario-Ebene operieren, häufig praktischer zuerst zu schreiben als granulare Unit-Tests. Ihr zentraler Wert in einem Modernisierungskontext ist, dass sie Refactoring- und Extraktionsarbeit von einem Vertrauensvorschuss in eine verifizierbare Aktivität verwandeln: Eine Änderung, die die funktionale Testsuite besteht, ist eine Änderung, die das beobachtbare Geschäftsverhalten nicht verändert hat, was die primäre Garantie ist, die Legacy-Arbeit braucht.

## How to Apply ◆

> In Legacy-Systemen sind funktionale Tests das primäre Sicherheitsnetz, das Änderung ermöglicht — ohne sie ist jede Modifikation ein Glücksspiel.

- Beginnen Sie damit, funktionale Tests für die kritischsten Geschäftsworkflows zu schreiben, bevor Sie Refactoring oder Modernisierung versuchen, wobei das aktuelle Systemverhalten als Spezifikation dient.
- Nutzen Sie Characterization Tests, um das bestehende Verhalten undokumentierten Legacy-Codes zu erfassen — führen Sie den Code mit bekannten Eingaben aus, zeichnen Sie die Ausgaben auf und wandeln Sie diese Aufzeichnungen in Assertions um.
- Fokussieren Sie sich anfänglich auf End-to-End-Geschäftsszenarien statt auf Unit-Ebene-Abdeckung, weil Legacy-Systeme oft eng gekoppelte Komponenten mit unklaren Unit-Grenzen haben.
- Automatisieren Sie die Testausführung in einer Continuous-Integration-Pipeline, sodass jede Änderung gegen die funktionale Testsuite validiert wird, bevor sie gemerged wird.
- Wenn Legacy-Systeme von externen Diensten oder Datenbanken abhängen, nutzen Sie Test Doubles oder aufgezeichnete Antworten, um funktionale Tests wiederholbar und schnell zu machen.
- Erweitern Sie die Testsuite schrittweise, während neue Bereiche der Legacy-Codebasis geändert werden, nach der Boy-Scout-Regel, das getestet zu hinterlassen, was Sie anfassen.
- Beziehen Sie Fachexperten in die Definition von Testszenarien ein, um sicherzustellen, dass Tests echte Geschäftsregeln abdecken, nicht nur technisches Verhalten.

## Tradeoffs ⇄

> Funktionale Tests bieten Vertrauen für Änderung, erfordern aber laufende Investition zur Erstellung und Pflege, besonders in Legacy-Systemen mit komplexem Zustand.

**Vorteile:**

- Ermöglicht sicheres Refactoring und Modernisierung, indem Regressionen sofort erkannt werden, wenn Legacy-Code geändert wird.
- Dokumentiert das tatsächliche Verhalten des Systems und dient als lebendige Dokumentation, wenn geschriebene Spezifikationen fehlen oder veraltet sind.
- Verringert die Kosten von Defekten, indem sie während der Entwicklung statt in der Produktion gefangen werden.
- Baut Teamvertrauen auf, um Änderungen in unbekannten Teilen der Legacy-Codebasis vorzunehmen.

**Kosten und Risiken:**

- Das Schreiben funktionaler Tests für Legacy-Systeme ohne bestehende Testinfrastruktur erfordert erhebliche Vorabinvestition in Test-Setup und Tooling.
- Legacy-Systeme mit enger Kopplung an Datenbanken, Dateisysteme oder externe Dienste können funktionale Tests langsam und brüchig machen, ohne sorgfältiges Testumgebungsmanagement.
- Übermäßiges Vertrauen auf funktionale Tests ohne Unit-Tests kann zu langen Testausführungszeiten führen, die die Entwicklungsfeedback-Schleife verlangsamen.
- Tests, die zu eng an Implementierungsdetails statt an Geschäftsverhalten gekoppelt sind, werden zu Wartungslasten, wenn das System refaktoriert wird.

## How It Could Be

> Die folgenden Szenarien zeigen, wie funktionale Tests die sichere Weiterentwicklung von Legacy-Systemen ermöglichen.

Ein Gesundheitsunternehmen erbte ein 20 Jahre altes Schadenbearbeitungssystem, geschrieben in einer Mischung aus Java und gespeicherten Prozeduren. Bevor versucht wurde, die Preis-Engine in einen separaten Dienst zu extrahieren, verbrachte das Team drei Wochen damit, funktionale Tests zu schreiben, die Beispielschäden durch die vollständige Verarbeitungspipeline einreichten und die berechneten Beträge verifizierten. Diese Tests fingen während des Extraktionsprozesses 14 Regressionen ab, die in der Produktion zu falschen Schadenzahlungen geführt hätten. Die Testsuite wurde zum vertrauenswürdigsten Artefakt des Teams — zuverlässiger als jede bestehende Dokumentation.

Eine Regierungsbehörde, die ein Legacy-Steuerberechnungssystem pflegte, musste es jedes Jahr für neue Vorschriften aktualisieren. Durch den Aufbau einer umfassenden funktionalen Testsuite aus historischen Steuererklärungsdaten und bekannten korrekten Ergebnissen verkürzte das Team den jährlichen Aktualisierungszyklus von vier Monaten auf sechs Wochen. Jede regulatorische Änderung konnte innerhalb von Minuten gegen Tausende realer Szenarien implementiert und verifiziert werden statt durch Wochen manuellen Testens.
