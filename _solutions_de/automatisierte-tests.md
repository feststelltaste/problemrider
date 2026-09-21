---
title: Automatisierte Tests
description: Automatisierte Verifikation von Funktionalität auf verschiedenen Ebenen.
category:
- Testing
- Code
problems:
- insufficient-testing
- poor-test-coverage
- legacy-code-without-tests
- regression-bugs
- fear-of-change
- high-bug-introduction-rate
- increased-manual-testing-effort
- difficult-to-test-code
- test-debt
- fear-of-failure
- past-negative-experiences
- defensive-coding-practices
- low-code-customization-sprawl
- upgrade-blocked-by-customization
layout: solution
lang: de
en_slug: automated-tests
related_solutions:
- slug: test-coverage-strategy
  similarity: 0.9
- slug: acceptance-tests
  similarity: 0.8
- slug: regression-testing
  similarity: 0.8
- slug: functional-tests
  similarity: 0.8
- slug: mutation-testing
  similarity: 0.8
- slug: integration-tests
  similarity: 0.8
---

## Description

Automatisierte Tests sind Code, der anderen Code ausführt und seine Ausgabe ohne menschliches Eingreifen gegen ein erwartetes Ergebnis prüft, wiederholt und günstig genug ausgeführt, um bei jeder Änderung statt nur vor einem Release durchgeführt zu werden. Ihre Kernfunktion ist es, eine manuelle, teure Verifikationstätigkeit in eine nahezu kostenlose, kontinuierliche zu verwandeln, was ändert, welche Arten von Änderungen sich ein Team leisten kann. In einem Legacy-System ist dies transformativ, genau weil Legacy-Code per Definition Code ist, den Menschen sich scheuen anzufassen: Ohne Tests wird jede Änderung nur durch manuelles Regressionstesten oder durch Hoffen, dass nichts bricht, validiert, was langsam, unvollständig und direkt verantwortlich für die Angst vor Veränderung ist, die Legacy-Systeme versteinert hält. Der Aufbau einer Test-Suite für ein solches System beginnt selten mit lehrbuchmäßigen Unit-Tests, da der Code nicht mit Testbarkeit im Sinn geschrieben wurde; er beginnt typischerweise mit Charakterisierungstests, die aktuelles Verhalten festhalten, gefolgt von gezielter Abdeckung der risikoreichsten, am häufigsten geänderten Bereiche, da erschöpfende Abdeckung der gesamten Legacy-Codebasis in einer Anstrengung selten erreichbar ist. Die Tests fungieren dann als Sicherheitsnetz, das weitere Refaktorierung und Modernisierung sicher macht, was der Grund ist, warum automatisiertes Testen so oft die erste Investition einer Legacy-Modernisierungsbemühung ist statt der letzten.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Beginnen Sie mit dem Hinzufügen von Charakterisierungstests, die das aktuelle Verhalten des Legacy-Systems dokumentieren, bevor Sie Änderungen vornehmen
- Identifizieren Sie Hochrisikobereiche der Codebasis mithilfe von Fehlerhistorie und Änderungshäufigkeit und priorisieren Sie dort Testabdeckung
- Führen Sie Unit-Tests für neuen Code und geänderten Code ein, der Boy-Scout-Regel folgend, Code besser zu hinterlassen, als Sie ihn vorgefunden haben
- Fügen Sie Integrationstests an Modulgrenzen hinzu, um Interaktionen zwischen Komponenten zu verifizieren
- Nutzen Sie Approval-Testing oder Snapshot-Testing für komplexe Legacy-Ausgaben, wo das Schreiben assertionsbasierter Tests unpraktikabel ist
- Richten Sie eine CI-Pipeline ein, die Tests automatisch bei jedem Commit ausführt, um schnelles Feedback zu bieten
- Etablieren Sie Mindestabdeckungsschwellen für neuen Code, während Sie schrittweise die Gesamtabdeckungsziele erhöhen
- Refaktorieren Sie eng gekoppelten Code, um Testbarkeit schrittweise zu verbessern, indem Abhängigkeiten hinter Interfaces extrahiert werden

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Bietet ein Sicherheitsnetz, das zuversichtliche Refaktorierung und Modernisierung von Legacy-Code ermöglicht
- Fängt Regressionsbugs früh ab, bevor sie Produktion erreichen
- Verringert den Bedarf an teuren manuellen Testzyklen
- Dient als lebendige Dokumentation erwarteten Systemverhaltens
- Beschleunigt die Entwicklungsgeschwindigkeit über die Zeit durch verringerten Debugging-Aufwand

**Kosten und Risiken:**
- Das Schreiben von Tests für Legacy-Code ohne klare Interfaces erfordert erhebliche Anfangsinvestition
- Schlecht geschriebene Tests können selbst zu einer Wartungslast werden
- Hohe Testabdeckung garantiert keine Abwesenheit von Bugs, wenn Tests oberflächlich sind
- Teams, die mit Testpraktiken nicht vertraut sind, brauchen Schulung und Mentoring
- Langsame Test-Suiten können zu einem Engpass werden, wenn sie nicht ordentlich strukturiert und gepflegt werden

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Banking-Anwendung hatte null automatisierte Tests und verließ sich vollständig auf einen vierwöchigen manuellen Regressionstestzyklus vor jedem Release. Das Team begann damit, Charakterisierungstests für das Zahlungsabwicklungsmodul unter Nutzung aufgezeichneter Produktionstransaktionen zu schreiben. Über sechs Monate bauten sie 800 Tests auf, die den kritischen Pfad abdeckten. Releases, die zuvor Wochen manuellen Testens erforderten, konnten in 20 Minuten automatisierter Testausführung validiert werden. Regressionsfehler im Zahlungsmodul sanken um 75 %, und das Team gewann die Zuversicht, mit der Refaktorierung der problematischsten Bereiche der Codebasis zu beginnen.
