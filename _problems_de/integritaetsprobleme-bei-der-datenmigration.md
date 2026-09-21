---
title: Integritätsprobleme bei der Datenmigration
description: Daten verlieren bei der Migration von Legacy- zu modernen Systemen
  aufgrund von Schema-Unstimmigkeiten und Formatinkompatibilitäten an Integrität,
  Konsistenz oder Bedeutung.
category:
- Code
- Database
- Operations
related_problems:
- slug: cross-system-data-synchronization-problems
  similarity: 0.7
- slug: data-migration-complexities
  similarity: 0.7
- slug: legacy-business-logic-extraction-difficulty
  similarity: 0.6
- slug: integration-difficulties
  similarity: 0.6
- slug: database-schema-design-problems
  similarity: 0.55
- slug: silent-data-corruption
  similarity: 0.55
solutions:
- evolutionary-database-design
- automated-migration-tools
- backup-and-recovery
- backward-compatible-data-formats
- backward-compatible-schema-migrations
- checksums
- continuous-data-verification
- data-integrity
- data-quality-checks
- fault-tolerant-data-structures
- idempotency-design
- platform-independent-data-storage
- plausibility-checks
- redundant-checksums
- redundant-data-storage
- regular-backups
- restore-points
- timestamping
- transactions
- write-ahead-logging
- domain-data-versioning
- error-correction-codes
- saga-pattern
- parallel-run
- production-like-test-data
- typed-schema-extraction
- master-data-stewardship
layout: problem
lang: de
en_slug: data-migration-integrity-issues
---

## Description

Integritätsprobleme bei der Datenmigration entstehen, wenn die Übertragung von Daten von Legacy-Systemen zu modernen Plattformen zu Datenkorruption, Verlust von Beziehungen, Änderungen der semantischen Bedeutung oder Konsistenzverletzungen führt. Diese Probleme entstehen aus grundlegenden Unterschieden zwischen Legacy- und modernen Datenmodellen, Kodierungsformaten, Constraint-Systemen und Geschäftsregel-Implementierungen. Anders als einfache Herausforderungen der Datenübertragung bedrohen diese Probleme die grundlegende Vertrauenswürdigkeit und Nutzbarkeit der migrierten Daten im neuen System.

## Indicators ⟡

- Legacy-Datenmodelle, die sich nicht sauber auf moderne Datenbankschemata oder Datenstrukturen abbilden lassen
- Entdeckung impliziter Geschäftsregeln, die in Legacy-Datenformaten oder -Constraints eingebettet sind
- Inkonsistenzen bei der Zeichenkodierung zwischen Legacy- und Zielsystemen
- Komplexe Beziehungen in Legacy-Daten, die kein Äquivalent im Design des Zielsystems haben
- Datenvalidierungsregeln, die sich erheblich zwischen Quell- und Zielsystemen unterscheiden
- Legacy-Systeme, die proprietäre Datenformate oder benutzerdefinierte Serialisierungsmethoden nutzen
- Fehlende oder unvollständige Datenwörterbücher für Legacy-Systemfelder und deren Bedeutungen

## Symptoms ▲

- [Stille Datenkorruption](stille-datenkorruption.md)
<br/>  Integritätsprobleme während der Migration können zunächst unentdeckt bleiben, wobei korrupte Daten falsche Ergebnisse liefern, ohne Fehler auszulösen.
- [Kundenunzufriedenheit](kundenunzufriedenheit.md)
<br/>  Nutzer, die nach der Migration auf falsche Daten, fehlende Datensätze oder korrupte Informationen stoßen, werden frustriert und verlieren das Vertrauen in das System.
- [Erhöhte manuelle Arbeit](erhoehte-manuelle-arbeit.md)
<br/>  Integritätsprobleme erfordern nach Abschluss der Migration umfangreiche manuelle Datenabstimmungs- und Korrekturmaßnahmen.
- [Systemausfälle](systemausfaelle.md)
<br/>  Schwerwiegende, nach der Migration entdeckte Integritätsprobleme können Notfall-Stopps für eine Neu-Migration erzwingen, was ungeplante Ausfallzeiten verursacht.
- [Erhöhte Fehlerraten](erhoehte-fehlerraten.md)
<br/>  Migrierte Daten mit Integritätsproblemen lösen Validierungsfehlschläge und Anwendungsfehler im neuen System aus.

## Causes ▼

- [Probleme im Datenbankschema-Design](probleme-im-datenbankschema-design.md)
<br/>  Grundlegende Schema-Unterschiede zwischen Legacy- und modernen Systemen schaffen Mapping-Herausforderungen, die die Datenintegrität während der Migration gefährden.
- [Schwierigkeit bei der Extraktion von Legacy-Geschäftslogik](schwierigkeit-bei-der-extraktion-von-legacy-geschaeftslogik.md)
<br/>  In Legacy-Datenformaten und -Constraints eingebettete Geschäftsregeln sind schwer zu identifizieren und während der Migration zu bewahren, was zu semantischem Datenverlust führt.
- [Komplexität der Datenmigration](komplexitaet-der-datenmigration.md)
<br/>  Die Gesamtkomplexität von Migrationsprozessen erhöht die Wahrscheinlichkeit von Fehlern, die die Datenintegrität gefährden.
- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Fehlende oder unvollständige Dokumentation über Legacy-Datenfelder, -Formate und deren Bedeutungen führt zu falschem Mapping und falscher Transformation während der Migration.
- [Veraltete Technologien](veraltete-technologien.md)
<br/>  Legacy-Systeme, die proprietäre Datenformate, veraltete Kodierungen wie EBCDIC oder benutzerdefinierte Serialisierung nutzen, schaffen Konvertierungsherausforderungen, die die Datenintegrität gefährden.

## Detection Methods ○

- Umsetzung umfassender Datenvalidierungs- und Abstimmungstests vor und nach der Migration
- Durchführung statistischer Analysen zum Vergleich von Datensatzanzahlen, Datenverteilungen und Beziehungsintegrität
- Nutzung von Daten-Profiling-Werkzeugen zur Identifikation von Inkonsistenzen zwischen Quell- und Zieldaten
- Durchführung von Nutzerakzeptanztests mit realen Geschäftsszenarien auf migrierten Daten
- Umsetzung automatisierter Datenqualitätsprüfungen zur Überwachung der laufenden Datenintegrität
- Vergleich der Ausgaben von Geschäftsberichten zwischen Legacy- und neuen Systemen auf Konsistenz
- Überwachung von Anwendungsfehlerprotokollen auf datenbezogene Validierungsfehlschläge nach der Migration
- Durchführung regelmäßiger Audits kritischer Geschäftsdaten auf Genauigkeit und Vollständigkeit

## Examples

Ein Finanzinstitut migriert Kundenkontodaten von einem Mainframe-System zu einer modernen Datenbank. Das Legacy-System speicherte Kontosalden als gepackte Dezimalfelder mit impliziten Währungsinformationen basierend auf dem Standort der Filiale, während Kundennamen in EBCDIC-Kodierung mit eingebetteten Formatierungscodes gespeichert waren. Während der Migration geht Dezimalgenauigkeit aufgrund von Fließkomma-Konvertierung verloren, was Cent-Abweichungen in Tausenden von Konten verursacht. Kundennamen werden aufgrund von Kodierungsproblemen korrumpiert, und die implizite Währungslogik geht verloren, was dazu führt, dass internationale Konten falsche Salden anzeigen. Die Migration erscheint mit korrekten Datensatzanzahlen erfolgreich, aber die Integritätsprobleme zeigen sich Wochen später, wenn Kunden falsche Kontoauszüge melden und regulatorische Berichte Audit-Anforderungen nicht erfüllen. Die Bank muss den Betrieb einstellen, um eine Notfall-Datenabstimmung und Neu-Migration durchzuführen, was Millionen an Ausfallzeit und regulatorischen Strafen kostet.
