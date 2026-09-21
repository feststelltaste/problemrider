---
title: Massentestdatengenerierung
description: Erzeugung großer Mengen künstlicher Testdaten mit realistischen
  Eigenschaften.
category:
- Testing
- Performance
problems:
- inadequate-test-data-management
- inadequate-test-infrastructure
- slow-database-queries
- gradual-performance-degradation
- database-query-performance-issues
- data-migration-complexities
- flaky-tests
layout: solution
lang: de
en_slug: mass-test-data-generation
related_solutions:
- slug: production-like-test-data
  similarity: 0.8
- slug: simulation-environments
  similarity: 0.7
- slug: load-testing
  similarity: 0.7
- slug: property-based-testing
  similarity: 0.7
- slug: test-coverage-strategy
  similarity: 0.7
- slug: automated-tests
  similarity: 0.65
---

## Description

Massentestdatengenerierung produziert große Mengen synthetischer Datensätze — passend zu Produktionsschemata, Verteilungen, Kardinalitäten und referenziellen Integritätsbeschränkungen — mittels Datengenerierungsbibliotheken oder benutzerdefinierter Generatoren, damit Tests gegen Datenvolumina ausgeführt werden können, die vergleichbar mit oder größer als das sind, was das System in der Produktion handhabt. Die generierten Daten können vollständig für Produktions-Snapshots substituieren oder anonymisierte Produktionsdaten ergänzen, wo synthetische Generierung allein subtile reale Korrelationen nicht erfassen kann, und weil sie skriptgesteuert und neben dem Schema versioniert ist, kann sie bei jedem Testlauf automatisch regeneriert und abgebaut werden. Legacy-Systeme sammeln eine spezifische Klasse von Bugs an, die nur im realistischen Datenmaßstab auftaucht — eine Abfrage, die gegen tausend Zeilen akzeptabel funktioniert, aber gegen fünfzig Millionen ein Timeout erleidet, eine gespeicherte Prozedur mit einer impliziten Annahme, die bricht, sobald sich Kardinalitäten verschieben, ein Migrationsskript, das sich anders verhält, sobald Volumen einen anderen Ausführungsplan auslöst —, und diese Bugs sind in kleinen, handgefertigten Testdatensätzen unsichtbar. Massengenerierte Testdaten bringen genau diese Problemklasse ans Licht, bevor sie die Produktion erreicht, was besonders wertvoll ist, wenn regulatorische Beschränkungen das Team davon abhalten, einfach eine Kopie echter Produktionsdaten zum Testen zu nutzen, wie es bei Gesundheits- oder Finanz-Legacy-Systemen üblich ist. Der Zielkonflikt ist, dass der Bau von Generatoren, die die undokumentierten Beschränkungen und Geschäftsregeln eines Legacy-Schemas respektieren können, selbst eine nicht-triviale Reverse-Engineering-Übung ist, und die Generatoren erfordern dann laufende Pflege, um gültig zu bleiben, während sich das Schema weiterentwickelt.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Analysieren Sie Produktionsdatenverteilungen, Kardinalitäten und Randfälle, um realistische Datengenerierungsprofile zu definieren
- Nutzen Sie Datengenerierungsbibliotheken (z. B. Faker, Bogus oder benutzerdefinierte Generatoren), um synthetische Datensätze zu erstellen, die zu Produktionsschemata passen
- Generieren Sie Datenvolumina, die Produktionsgrößen entsprechen oder übersteigen, um Performance-Probleme aufzudecken, die erst im großen Maßstab auftauchen
- Stellen Sie referenzielle Integrität und Geschäftsregelkonformität in generierten Daten sicher, damit Tests realistische Codepfade ausüben
- Anonymisieren und transformieren Sie Produktionsdaten-Snapshots als ergänzenden Ansatz, wenn synthetische Daten allein unzureichend sind
- Automatisieren Sie die Generierung und den Abbau von Testdatensätzen, damit sie bei jedem Testlauf erneuert werden können
- Versionieren Sie die Datengenerierungsskripte neben der Codebasis, um sie mit Schemaänderungen synchron zu halten

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Ermöglicht realistisches Performance-Testen, ohne das Risiko einer Offenlegung von Produktionsdaten
- Deckt datenvolumenabhängige Bugs auf, wie langsame Abfragen, Paginierungsprobleme und Speicherprobleme
- Unterstützt Datenmigrationsproben, indem große Datensätze zur Validierung von Migrationsskripten bereitgestellt werden
- Ermöglicht parallele Entwicklung von Features, die von Datenszenarien abhängen, die in der Produktion noch nicht vorhanden sind

**Kosten und Risiken:**
- Der Bau realistischer Generatoren für komplexe Legacy-Schemata mit undokumentierten Beschränkungen ist arbeitsintensiv
- Generierte Daten könnten subtile reale Korrelationen vermissen, die spezifische Codepfade auslösen
- Die Pflege von Generatoren, während sich das Schema weiterentwickelt, fügt laufenden Aufwand hinzu
- Sehr große Datensätze erfordern erheblichen Speicher und können die Bereitstellung der Testumgebung verlangsamen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Gesundheitsplattform musste eine Datenbankmigration von Oracle zu PostgreSQL validieren, konnte aber wegen regulatorischer Beschränkungen keine Produktionsdaten nutzen. Das Team baute einen Datengenerator, der 50 Millionen Patientendatensätze mit realistischen Verteilungen von Diagnosen, Terminverläufen und Versicherungsbeziehungen produzierte. Das Ausführen der Migration gegen diesen synthetischen Datensatz offenbarte, dass mehrere gespeicherte Prozeduren implizites Oracle-spezifisches Verhalten hatten, das im kleinen Maßstab korrekt funktionierte, aber bei realistischen Datenvolumina Timeouts verursachte. Diese Probleme vor der eigentlichen Migration zu beheben verhinderte, was ein kostspieliger Rollback in der Produktion gewesen wäre.
