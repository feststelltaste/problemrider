---
title: Prüfsummen
description: Berechnung von Prüfsummen zur Erkennung von Datenfehlern oder -änderungen.
category:
- Security
- Code
problems:
- silent-data-corruption
- data-migration-integrity-issues
- cross-system-data-synchronization-problems
- insecure-data-transmission
- dma-coherency-issues
- retention-obligations-block-change
layout: solution
lang: de
en_slug: checksums
related_solutions:
- slug: redundant-checksums
  similarity: 0.9
- slug: error-correction-codes
  similarity: 0.8
- slug: data-integrity
  similarity: 0.8
- slug: continuous-data-verification
  similarity: 0.8
- slug: fault-tolerant-data-structures
  similarity: 0.75
- slug: plausibility-checks
  similarity: 0.7
---

## Description

Eine Prüfsumme ist ein kleiner, fester Wert, berechnet aus einem größeren Datenblock unter Nutzung eines Algorithmus wie CRC32 oder SHA-256, sodass jede Änderung an den ursprünglichen Daten sehr wahrscheinlich den berechneten Wert ändert, was Prüfsummen zu einem effizienten Weg macht, Korruption oder unautorisierte Änderung zu erkennen, ohne die vollständigen Daten selbst zu vergleichen. Sie werden am Punkt der Datenerzeugung oder -übertragung generiert und am Punkt des Konsums neu berechnet, wobei eine Nichtübereinstimmung signalisiert, dass die Daten irgendwo auf dem Weg verändert wurden. Dies ist besonders relevant in Legacy-Modernisierungsarbeit, die Datenmigration beinhaltet, wo sich Datensätze zwischen Systemen mit unterschiedlichen Kodierungen, Präzisionshandhabung oder Speicherformaten bewegen, und stille Korruption während der Übertragung völlig unbemerkt bleiben kann, bis sie viel später als unerklärliche Geschäftsdiskrepanz auftaucht. Das Hinzufügen von Prüfsummenverifikation an beiden Enden eines Migrationsbatches verwandelt dieses Risiko von einem unsichtbaren, kumulativen Problem in ein sofort erkennbares, da eine fehlgeschlagene Prüfsummenverglichen genau markiert, welche Datensätze untersucht werden müssen, bevor ihnen nachgelagert vertraut wird. Derselbe Mechanismus schützt laufende Datenflüsse in Legacy-Systemen — Dateiübertragungen, Nachrichtenwarteschlangen, API-Payloads —, wo Übertragungsfehler über unzuverlässige Kanäle sonst Daten ohne sichtbares Symptom korrumpieren könnten. Prüfsummen sind kein Ersatz für stärkere Integritätsgarantien wie kryptografische Signierung, wenn Manipulation statt versehentlicher Korruption das Anliegen ist, und ihr Schutzwert ist auf die für die Aufgabe gewählte Algorithmusstärke beschränkt.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie kritische Datenflüsse, wo Korruption oder Manipulation auftreten könnte: Dateiübertragungen, Datenbankmigrationen, API-Kommunikation und Nachrichtenwarteschlangen
- Wählen Sie angemessene Prüfsummenalgorithmen basierend auf Anforderungen (CRC32 für Fehlererkennung, SHA-256 für Integritätsverifikation)
- Fügen Sie Prüfsummengenerierung an Datenquellenpunkten und Verifikation an Konsumpunkten hinzu
- Beziehen Sie Prüfsummen in Datenmigrationsskripte ein, um zu verifizieren, dass Quell- und Zieldaten nach der Migration übereinstimmen
- Implementieren Sie Prüfsummenvalidierung in Datei-Upload-/Download-Prozessen, um Übertragungskorruption zu erkennen
- Speichern Sie Prüfsummen zusammen mit Datensätzen für periodische Integritätsaudits
- Protokollieren Sie Prüfsummen-Nichtübereinstimmungen mit ausreichendem Kontext für Untersuchung und Behebung

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Erkennt Datenkorruption früh, bevor sie sich durch das System verbreitet
- Bietet Vertrauen in Datenintegrität während Migrationen zwischen Legacy- und modernen Systemen
- Ermöglicht Verifikation der Vollständigkeit der Datenübertragung über unzuverlässige Netzwerke
- Schafft einen Prüfpfad für Datenänderungen und Integritätsverifikation

**Kosten und Risiken:**
- Fügt Rechen-Overhead für die Prüfsummenberechnung auf hochvolumigen Datenpfaden hinzu
- Prüfsummenspeicherung erfordert zusätzlichen Platz neben den tatsächlichen Daten
- Falsches Sicherheitsgefühl bei Nutzung schwacher Prüfsummenalgorithmen, die bestimmte Fehlermuster nicht erkennen
- Die Nachrüstung von Prüfsummenverifikation in bestehende Datenflüsse erfordert Änderungen sowohl am Sender als auch am Empfänger

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Während einer Legacy-Datenbankmigration von einem On-Premises-SQL-Server zu einer Cloud-gehosteten PostgreSQL-Instanz entdeckte ein Finanzdienstleistungsteam, dass 0,3 % der Datensätze während der Übertragung aufgrund von Zeichenkodierungs-Nichtübereinstimmungen still korrumpiert worden waren. Nach der Behebung des Kodierungsproblems fügten sie SHA-256-Prüfsummen zu jedem Batch migrierter Datensätze hinzu. Das Migrationsskript berechnete Prüfsummen an der Quelle, übertrug die Daten, berechnete Prüfsummen am Ziel neu und verglich sie, bevor jeder Batch committet wurde. Dieser Ansatz fing zwei zusätzliche Korruptionsmuster im Zusammenhang mit Dezimalpräzisionsunterschieden ab und stellte sicher, dass alle 12 Millionen Datensätze intakt ankamen.
