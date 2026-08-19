---
title: Datenqualitätsprüfungen
description: Sicherstellung der Datenqualität durch Validierung, Bereinigung und
  Anreicherung.
category:
- Database
- Testing
problems:
- silent-data-corruption
- data-migration-integrity-issues
- data-migration-complexities
- inconsistent-behavior
- unpredictable-system-behavior
- unbounded-data-growth
- entity-attribute-value-overuse
- master-data-ownership-gaps
layout: solution
lang: de
en_slug: data-quality-checks
related_solutions:
- slug: data-integrity
  similarity: 0.7
- slug: continuous-data-verification
  similarity: 0.7
- slug: checksums
  similarity: 0.7
- slug: data-enrichment
  similarity: 0.7
- slug: plausibility-checks
  similarity: 0.7
- slug: code-quality-gates
  similarity: 0.7
---

## Description

Datenqualitätsprüfungen sind automatisierte Regeln — Pflichtfelder, gültige Wertebereiche, referenzielle Integrität, Formatbeschränkungen und geschäftsspezifische Validierungen —, die periodisch gegen eine Datenbank ausgeführt oder am Eingabepunkt angewandt werden, um Daten zu erkennen und zu melden, die die Erwartungen der Organisation an Korrektheit verletzen. Der Mechanismus wirkt an zwei Punkten des Datenlebenszyklus: Am Eingabepunkt angewandte Prüfungen verhindern die Entstehung neuer schlechter Daten, während periodisch gegen bestehende Daten ausgeführte Prüfungen bereits vorhandene Probleme aufdecken, kategorisiert nach Schweregrad, sodass der Bereinigungsaufwand priorisiert werden kann. In Legacy-Systemen ist dies essenziell, gerade weil Eingabepunktvalidierung oft jahrelang schwach oder abwesend war, was doppelte Datensätze, verwaiste Referenzen und inkonsistente Formate still anhäufen ließ, bis sie als Debugging-Rätsel oder, schlimmer, als korrumpierte Berichte auftauchten, die niemand hinterfragte, bis die Zahlen sichtbar falsch waren. Datenqualitätsprüfungen sind besonders kritisch vor jeder Migration, da eine umfassende Qualitätsbewertung gegen die Legacy-Quelle es dem Team erlaubt, Probleme zu quantifizieren und zu adressieren, bevor sie in ein neues System übertragen werden, statt sie nach dem Go-live zu entdecken und zu beheben, wenn die Behebungskosten weit höher sind. Weil die Bereinigung selbst riskant sein kann, wenn die „korrekte" Form der Daten nicht gut verstanden ist, werden Qualitätsprüfungen typischerweise zunächst als Erkennungs- und Meldemechanismus implementiert, mit Bereinigungsskripten, die in kontrollierten, überprüften Batches angewandt werden, statt als automatische Korrekturmaßnahme.

## How to Apply ◆

- Definieren Sie Datenqualitätsregeln basierend auf Geschäftsanforderungen: Pflichtfelder, gültige Bereiche, referenzielle Integrität, Formatbeschränkungen und Geschäftslogikvalidierungen.
- Implementieren Sie automatisierte Datenqualitätsprüfungen, die periodisch gegen Legacy-Datenbanken laufen, um Qualitätsprobleme zu erkennen und zu melden.
- Fügen Sie Validierung an Dateneingabepunkten im Legacy-System hinzu, um zu verhindern, dass künftig schlechte Daten ins System gelangen.
- Erstellen Sie Datenbereinigungsskripte für bekannte Qualitätsprobleme (Duplikate, ungültige Formate, verwaiste Datensätze) und führen Sie sie in kontrollierten Batches aus.
- Überwachen Sie Datenqualitätskennzahlen über die Zeit und setzen Sie Alarme, wenn die Qualität unter akzeptable Schwellenwerte fällt.
- Führen Sie umfassende Datenqualitätsbewertungen vor jeder Datenmigration durch, um Probleme proaktiv zu identifizieren und zu adressieren.

## Tradeoffs ⇄

**Vorteile:**
- Verhindert, dass sich Datenqualitätsprobleme durch das System fortpflanzen und nachgelagerte Fehler verursachen.
- Reduziert die für die Fehlersuche bei durch schlechte Daten verursachten Problemen in Legacy-Systemen aufgewendete Zeit.
- Verbessert das Vertrauen in Berichte und Analytics, die aus Legacy-Daten abgeleitet werden.
- Identifiziert Datenqualitätsprobleme, bevor sie während der Migration Probleme verursachen.

**Kosten:**
- Die Implementierung umfassender Datenqualitätsprüfungen für eine große Legacy-Datenbank erfordert erheblichen Aufwand.
- Datenbereinigung kann riskant sein, wenn Geschäftsregeln für „korrekte" Daten nicht gut verstanden werden.
- Automatisierte Prüfungen fügen Verarbeitungsoverhead hinzu und können die Datenbankperformance beeinträchtigen.
- Falsch-Positive bei Qualitätsprüfungen können Alarmmüdigkeit erzeugen.

## How It Could Be

Ein Legacy-Buchhaltungssystem häufte zwanzig Jahre Transaktionsdaten mit verschiedenen Qualitätsproblemen an: doppelte Kundendatensätze, Transaktionen mit fehlenden Referenznummern und Beträge in inkonsistenten Dezimalformaten gespeichert. Vor der Migration zu einem neuen ERP-System implementiert das Team eine Suite von Datenqualitätsprüfungen, die die gesamte Datenbank scannen und Probleme nach Schweregrad kategorisieren. Sie entdecken, dass 8 Prozent der Kundendatensätze Duplikate sind und dass Tausende von Transaktionen auf gelöschte Konten verweisen. Das Team baut Bereinigungsskripte, die doppelte Kunden zusammenführen (unter Erhalt der Transaktionshistorie) und verwaiste Transaktionen abgleichen. Diese Prüfungen vor der Migration auszuführen verhindert, dass Jahre von Datenqualitätsproblemen ins neue System übertragen werden, und vermeidet die teure Aufgabe, sie nach dem Go-live zu beheben.
