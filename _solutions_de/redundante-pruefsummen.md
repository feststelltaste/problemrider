---
title: Redundante Prüfsummen
description: Nutzung mehrerer unterschiedlicher Prüfsummenalgorithmen.
category:
- Code
- Security
problems:
- silent-data-corruption
- data-migration-integrity-issues
- inadequate-error-handling
- unpredictable-system-behavior
- dma-coherency-issues
layout: solution
lang: de
en_slug: redundant-checksums
related_solutions:
- slug: checksums
  similarity: 0.9
- slug: error-correction-codes
  similarity: 0.8
- slug: redundant-data-storage
  similarity: 0.75
- slug: continuous-data-verification
  similarity: 0.75
- slug: fault-tolerant-data-structures
  similarity: 0.7
- slug: data-integrity
  similarity: 0.7
---

## Description

Redundante Prüfsummen wenden zwei oder mehr unabhängige Prüfsummen- oder Hash-Algorithmen — zum Beispiel eine schnelle CRC32 neben einer kryptografisch stärkeren SHA-256 — auf dasselbe Datenstück an, sodass Korruption an mehreren, mathematisch unabhängigen Verifikationsmethoden vorbeischlüpfen muss, bevor sie unentdeckt bleibt. Da unterschiedliche Algorithmen unterschiedliche Kollisionscharakteristika und blinde Flecken haben, ist die Wahrscheinlichkeit, dass ein korrumpiertes Bytemuster zufällig beide Prüfsummen gleichzeitig bewahrt, weit geringer, als eine einzelne Prüfung zu umgehen. Dies zählt in Legacy-Systemen, weil viele von ihnen mit einem einzelnen, oft schwachen Prüfsummenschema — oder gar keinem — gebaut wurden, gewählt vor Jahrzehnten unter anderen Annahmen über Datenvolumen und Bedrohungsmodelle, und stille Datenkorruption in solchen Systemen bleibt häufig unbemerkt, bis sie als schwer zu diagnostizierender nachgelagerter Fehler auftaucht. Redundante Prüfsummen sind besonders wertvoll während Datenmigrationsaufwänden, wo die Verifikation, dass Millionen von aus einem Legacy-Speicher auf eine neue Plattform übertragenen Datensätzen bitgenau identisch sind, eine Voraussetzung dafür ist, dem neuen System überhaupt zu vertrauen. Die Technik ist günstig, nachträglich an Datengrenzen — Eingang, Speicherung und Übertragungspunkte — einzubauen, ohne die Kernanwendungslogik anzufassen, was sie zu einem pragmatischen ersten Schritt zu stärkeren Datenintegritätsgarantien in Systemen macht, die nicht schnell umarchitektiert werden können.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie kritische Datenflüsse im Legacy-System, bei denen Korruption schwerwiegende Geschäftskonsequenzen hätte
- Wenden Sie mehrere unabhängige Prüfsummenalgorithmen (z. B. CRC32 und SHA-256) auf kritische Daten während der Übertragung und im Ruhezustand an
- Verifizieren Sie Prüfsummen an jeder Systemgrenze, an der Daten empfangen, gespeichert oder übertragen werden
- Speichern Sie Prüfsummen neben den Daten, die sie schützen, und validieren Sie sie während Lesevorgängen
- Implementieren Sie automatisierte Alarmierung, wenn Prüfsummenverifikation fehlschlägt
- Nutzen Sie redundante Prüfsummen während Datenmigrationen, um zu verifizieren, dass Quell- und Zieldaten exakt übereinstimmen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Reduziert die Wahrscheinlichkeit unentdeckter Datenkorruption dramatisch
- Bietet starke Garantien für Datenintegrität während Migration und Replikation
- Erfasst Korruption, verursacht durch Hardwareausfälle, Softwarefehler oder Übertragungsfehler
- Zwei unabhängige Algorithmen machen es praktisch unmöglich, dass Korruption beide Prüfungen besteht

**Kosten und Risiken:**
- Rechenaufwand für die Berechnung mehrerer Prüfsummen bei jeder Datenoperation
- Zusätzlicher Speicher für mehrere Prüfsummenwerte benötigt
- Erhöht die Codekomplexität an Datenverarbeitungsgrenzen
- Legacy-Datenformate benötigen möglicherweise Modifikation, um Prüfsummenfelder aufzunehmen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine wissenschaftliche Forschungseinrichtung entdeckte, dass ihr Legacy-Datenarchivierungssystem still einen kleinen Prozentsatz von Dateien während der Speicherung korrumpiert hatte, unentdeckbar durch die einzelne genutzte CRC32-Prüfsumme. Durch das Hinzufügen einer zweiten SHA-256-Prüfsumme und die Verifikation beider bei jedem Lesevorgang identifizierte das Team 47 korrumpierte Dateien im Archiv, die die CRC32-Prüfung allein bestanden hatten. Für die laufende Datenmigration zu einer neuen Speicherplattform boten duale Prüfsummen Vertrauen, dass jede Datei mit perfekter Genauigkeit übertragen wurde.
