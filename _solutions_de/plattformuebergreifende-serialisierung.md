---
title: Plattformübergreifende Serialisierung
description: Nutzung von Daten-Serialisierern, die über verschiedene Systeme hinweg
  kompatibel sind.
category:
- Architecture
- Dependencies
problems:
- cross-system-data-synchronization-problems
- integration-difficulties
- serialization-deserialization-bottlenecks
- technology-stack-fragmentation
- poor-interfaces-between-applications
- breaking-changes
- endianness-conversion-overhead
- alignment-and-padding-issues
layout: solution
lang: de
en_slug: cross-platform-serialization
related_solutions:
- slug: standardized-data-formats
  similarity: 0.75
- slug: platform-independent-programming-languages
  similarity: 0.7
- slug: platform-independent-data-storage
  similarity: 0.7
- slug: data-format-conversion
  similarity: 0.7
- slug: backward-compatible-data-formats
  similarity: 0.7
- slug: data-formats
  similarity: 0.7
---

## Description

Plattformübergreifende Serialisierung ersetzt sprachnative Serialisierungsmechanismen — Javas Serializable, .NETs BinaryFormatter, Pythons pickle — durch plattformneutrale, explizit schemadefinierte Formate wie JSON, Protocol Buffers oder Avro, sodass von einer Sprachlaufzeit erzeugte Daten direkt von einem in einer anderen Sprache geschriebenen System konsumiert werden können. Legacy-Systeme, die früh ein sprachnatives Serialisierungsformat übernommen haben, taten dies typischerweise, weil es damals der Weg des geringsten Widerstands war, aber diese Wahl wird zum aktiven Hindernis in dem Moment, in dem die Organisation einen Service in einer anderen Sprache einführen möchte — etwa einen Python-Analytics-Service, der Javas Serializable-Format nicht deserialisieren kann —, was eine unbeholfene Übersetzungsschicht erzwingt oder den neuen Service vollständig blockiert. Plattformübergreifende Formate schließen auch eine Sicherheitslücke, die mit mehreren sprachnativen Serialisierern einhergeht, die eine Geschichte von Deserialisierungsschwachstellen haben, die aus dem Design des Formats selbst statt aus Anwendungscode stammen. Explizite, versionierte Schemata kombiniert mit toleranten Lesern, die unbekannte Felder ignorieren, machen es möglich, dass sich das Format weiterentwickelt, ohne bestehende Konsumenten zu brechen, was in einer Legacy-Integrationslandschaft wichtig ist, wo nicht jeder Konsument eines Datenstroms gleichzeitig identifiziert oder aktualisiert werden kann. Die Migration selbst lässt typischerweise altes und neues Format für eine Übergangsperiode parallel laufen, da Konsumenten nicht alle gleichzeitig umgestellt werden können und der Parallellauf einen Sicherheitsspielraum bietet, um Lücken in der Abdeckung des neuen Formats zu finden, bevor das alte stillgelegt wird.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Ersetzen Sie sprachspezifische Serialisierung (Java Serializable, .NET BinaryFormatter, Python pickle) durch plattformneutrale Formate
- Wählen Sie ein für Ihren Anwendungsfall geeignetes Serialisierungsformat: JSON für menschenlesbare APIs, Protocol Buffers oder Avro für hochdurchsatzige interne Kommunikation
- Definieren Sie Schemata für serialisierte Daten und versionieren Sie sie explizit
- Testen Sie Serialisierung und Deserialisierung über alle Plattformen, die Daten austauschen
- Implementieren Sie tolerante Leser, die unbekannte Felder während der Schema-Evolution graziös handhaben
- Migrieren Sie schrittweise, indem Sie während der Übergangsphase sowohl altes als auch neues Serialisierungsformat unterstützen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Ermöglicht Datenaustausch zwischen Systemen, die in unterschiedlichen Sprachen und Frameworks geschrieben sind
- Reduziert das Risiko von Deserialisierungsschwachstellen, die mit sprachnativer Serialisierung verbunden sind
- Vereinfacht das Hinzufügen neuer Systeme zur Integrationslandschaft

**Kosten und Risiken:**
- Plattformneutrale Formate können weniger performant sein als native Binärserialisierung
- Schemaverwaltung fügt Komplexität hinzu, besonders wenn mehrere Versionen koexistieren
- Die Migration von proprietären Serialisierungsformaten erfordert sorgfältige Handhabung der Abwärtskompatibilität
- Manche komplexen Objektgraphen können schwer in einfacheren plattformübergreifenden Formaten darzustellen sein

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Logistikunternehmen hatte ein Java-basiertes Bestellsystem, das Java Serializable nutzte, um Nachrichten in einer Warteschlange zu speichern, was einen neuen Python-basierten Analytics-Service daran hinderte, diese Nachrichten zu konsumieren. Das Team migrierte das Nachrichtenformat zu Avro mit einer Schema-Registry und ließ beide Formate vier Wochen lang parallel laufen. Nach dem Übergang konsumierten sowohl Java- als auch Python-Services denselben Nachrichtenstrom ohne jede Übersetzungsschicht, und die Schema-Registry verhinderte drei inkompatible Schemaänderungen während der nachfolgenden Entwicklung.
