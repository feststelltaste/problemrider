---
title: Versionsübergreifende Tests
description: Testen der Software mit unterschiedlichen Versionen.
category:
- Testing
problems:
- dependency-version-conflicts
- regression-bugs
- breaking-changes
- deployment-environment-inconsistencies
- insufficient-testing
- integration-difficulties
- abi-compatibility-issues
layout: solution
lang: de
en_slug: cross-version-testing
related_solutions:
- slug: compatibility-testing
  similarity: 0.9
- slug: compatibility-testing-by-users
  similarity: 0.8
- slug: documentation-of-compatibility-requirements
  similarity: 0.75
- slug: compatibility-certification
  similarity: 0.75
- slug: isolated-test-environments
  similarity: 0.75
- slug: integration-tests
  similarity: 0.75
---

## Description

Versionsübergreifende Tests führen die Testsuite der Software gegen mehrere Versionen ihrer Laufzeitabhängigkeiten aus — Sprachlaufzeiten, Frameworks, Datenbanken, Betriebssysteme —, statt nur gegen die eine Versionskombination zu validieren, die das Entwicklungsteam zufällig lokal nutzt. Legacy-Systeme in Produktion laufen selten alle mit exakt denselben Abhängigkeitsversionen; unterschiedliche Kunden oder Deployments einigen sich auf unterschiedliche Datenbankversionen, unterschiedliche Laufzeit-Patch-Level und unterschiedliche Upgrade-Zeitpläne, und ein Fehler, der sich nur gegen eine bestimmte Versionskombination manifestiert, kann in der eigenen Testumgebung eines Teams unbegrenzt unsichtbar bleiben. Dies als Versionsmatrix in CI zu automatisieren, statt manuell und nur gelegentlich zu testen, verwandelt eine implizite Annahme — „dies funktioniert über die Versionen, die unsere Nutzer tatsächlich betreiben" — in etwas kontinuierlich Verifiziertes, und es erzeugt auch die Belege, die nötig sind, um eine informierte Entscheidung darüber zu treffen, wann es endlich sicher ist, die Unterstützung für eine alte Version fallen zu lassen. Weil die Matrix kombinatorisch mit der Anzahl der im Spiel befindlichen Abhängigkeitsversionen wächst, wird die Praxis bewusst auf die Upgrade-Pfade beschränkt, die Nutzer tatsächlich gehen, statt auf jede theoretische Kombination, was CI-Kosten und Ausführungszeit proportional zum gemanagten Risiko hält. Dies macht versionsübergreifende Tests zu einem gezielten Weg, die „funktioniert auf meiner Version"-Klasse von Produktionsproblemen abzufangen, bevor sie Kunden erreicht, die eine ältere oder neuere Abhängigkeit betreiben als die, gegen die das Team getestet hat.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie alle Laufzeitabhängigkeiten (Sprachversionen, Frameworks, Datenbanken, OS-Versionen), die über Ihre Deployment-Umgebungen hinweg variieren
- Erstellen Sie eine Testmatrix, die die in Produktion vorhandenen Versionskombinationen abdeckt
- Automatisieren Sie versionsübergreifende Testausführung mittels CI-Matrix-Builds oder containerisierten Testumgebungen
- Testen Sie sowohl das aktuelle Release gegen ältere Abhängigkeiten als auch ältere Releases gegen neuere Abhängigkeiten
- Fokussieren Sie sich auf die Upgrade-Pfade, die Ihre Nutzer tatsächlich gehen, nicht auf jede theoretische Kombination
- Beziehen Sie versionsübergreifende Tests in die Release-Checkliste für Major-Version-Sprünge ein

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Erkennt versionsspezifische Fehler, bevor sie Nutzer betreffen, die unterschiedliche Laufzeitversionen betreiben
- Validiert Upgrade-Pfade, um Nutzern Vertrauen bei der Übernahme neuer Versionen zu geben
- Reduziert die „funktioniert auf meiner Version"-Klasse von Produktionsproblemen

**Kosten und Risiken:**
- Die Matrixgröße wächst schnell mit der Anzahl der Abhängigkeitsversionen, was CI-Kosten erhöht
- Die Pflege von Testumgebungen für alte Versionen erfordert laufenden Aufwand
- Manche Versionskombinationen können wegen bekannter Upstream-Fehler flakige Ergebnisse erzeugen
- Abnehmender Ertrag beim Testen sehr alter oder selten genutzter Versionskombinationen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Datenbanktreiber-Bibliothek unterstützte PostgreSQL-Versionen 10 bis 16. Nach der Einführung einer CI-Matrix, die jeden Pull Request gegen alle sieben PostgreSQL-Versionen testete, erkannte das Team eine für PostgreSQL 12 spezifische Query-Plan-Regression, die einen erheblichen Teil ihrer Nutzerbasis betroffen hätte. Die Matrix lieferte dem Team auch Daten, um die Einstellung der PostgreSQL-10-Unterstützung zu rechtfertigen, als CI zeigte, dass null Nutzer ausschließlich auf dieser Version Probleme gemeldet hatten.
