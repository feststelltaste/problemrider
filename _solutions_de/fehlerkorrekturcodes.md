---
title: Fehlerkorrekturcodes
description: Nutzung von Codes zur Erkennung und Korrektur von Fehlern in Daten.
category:
- Code
- Security
problems:
- silent-data-corruption
- data-migration-integrity-issues
- insecure-data-transmission
- cross-system-data-synchronization-problems
layout: solution
lang: de
en_slug: error-correction-codes
related_solutions:
- slug: fault-tolerant-data-structures
  similarity: 0.8
- slug: checksums
  similarity: 0.8
- slug: redundant-checksums
  similarity: 0.8
- slug: error-reporting-and-analysis
  similarity: 0.75
- slug: redundant-data-storage
  similarity: 0.75
- slug: data-integrity
  similarity: 0.75
---

## Description

Fehlerkorrekturcodes fügen Daten strukturierte, berechenbare Redundanz hinzu — Paritätsbits, Hamming-Codes, Reed-Solomon-Codes oder CRC kombiniert mit Forward Error Correction —, sodass ein Empfänger oder Leser Daten, die während der Übertragung oder Speicherung korrumpiert wurden, erkennen und, innerhalb der gestalteten Kapazität des Codes, automatisch rekonstruieren kann, ohne erneute Übertragung oder manuellen Eingriff zu erfordern. In Legacy-Kontexten ist dies am wichtigsten an den Grenzen, an denen alte und neue Komponenten aufeinandertreffen: alternde serielle Verbindungen, Legacy-Protokolle, die nie mit Integritätsprüfung entworfen wurden, oder Archivspeicher auf Medien, die über die Zeit degradieren — all dies ist häufig in Systemen, die die heutigen widerstandsfähigeren Transport- und Speicherschichten vordatieren. Fehlerkorrektur an diesen Grenzen hinzuzufügen erlaubt einem Team, Zuverlässigkeit zu verbessern, ohne die Kernlogik des Legacy-Protokolls anzufassen, und die Korrekturrate selbst wird zu einem nützlichen Gesundheitssignal, das degradierende Kabel, Steckverbinder oder Speichermedien markiert, bevor sie vollständig versagen. Der Tradeoff ist, dass die Korrekturkapazität endlich und durch den gewählten Code begrenzt ist, Verarbeitungs- und Datengrößenoverhead hinzufügt und, unvorsichtig genutzt, ein Infrastrukturproblem verschleiern kann, das tatsächlich ersetzt statt korrigiert werden müsste.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie korruptionsanfällige Datenkanäle: Netzwerkübertragungen, Festplattenspeicher, prozessübergreifende Kommunikation und Legacy-Protokoll-Integrationen
- Wählen Sie angemessene Fehlerkorrekturschemata basierend auf der erwarteten Fehlerrate und Performance-Einschränkungen (z. B. Reed-Solomon, Hamming-Codes, CRC mit Forward Error Correction)
- Implementieren Sie Fehlerkorrektur auf der Transportschicht für kritische Datenübertragungen zwischen Legacy- und modernen Komponenten
- Fügen Sie Paritäts- oder Prüfsummenfelder zu Datenstrukturen und Nachrichtenformaten hinzu, die in Legacy-Integrationen genutzt werden
- Nutzen Sie fehlerkorrigierende Speicherformate für kritische Archivdaten, die über lange Zeiträume lesbar bleiben müssen
- Überwachen Sie Fehlerkorrekturraten, um degradierende Hardware oder Netzwerkinfrastruktur zu erkennen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Korrigiert kleinere Datenfehler automatisch, ohne erneute Übertragung oder manuellen Eingriff zu erfordern
- Verbessert die Datenzuverlässigkeit über unzuverlässige Kommunikationskanäle oder alternde Speichermedien
- Bietet einen messbaren Indikator für Infrastrukturgesundheit durch Verfolgung der Korrekturhäufigkeit
- Verlängert die Nutzungsdauer von Legacy-Daten, die auf alternden Medien gespeichert sind

**Kosten und Risiken:**
- Fehlerkorrektur fügt Datengröße und Verarbeitungszeit Overhead hinzu
- Kann Fehler jenseits der gestalteten Korrekturkapazität des Codes nicht korrigieren
- Fügt Implementierungskomplexität hinzu, die korrekt implementiert werden muss, um effektiv zu sein
- Übermäßiges Vertrauen auf Fehlerkorrektur kann zugrunde liegende Infrastrukturprobleme verschleiern, die behoben werden sollten

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Industriesteuerungssystem kommunizierte mit Sensoren über verrauschte serielle Verbindungen, wo Bitfehler häufig waren. Das ursprüngliche Protokoll hatte keine Fehlerkorrektur, und korrumpierte Sensormesswerte lösten gelegentlich Fehlalarme aus oder, schlimmer, versagten dabei, echte Alarme auszulösen. Das Team fügte dem Kommunikationsprotokoll Reed-Solomon-Fehlerkorrektur hinzu, die bis zu drei Bitfehler pro Nachricht korrigieren konnte. Dies reduzierte Datenfehler von 2 Prozent auf unter 0,001 Prozent der Nachrichten, ohne Hardware-Upgrades zu erfordern. Die Fehlerkorrekturrate diente auch als Frühwarnsignal für Sensorverbindungsdegradation und alarmierte Wartungsteams, wenn Kabel oder Steckverbinder ersetzt werden mussten.
