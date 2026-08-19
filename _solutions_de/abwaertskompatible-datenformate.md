---
title: Abwärtskompatible Datenformate
description: Sicherstellung der Abwärtskompatibilität bei der Einführung neuer Datenformate.
category:
- Architecture
- Database
problems:
- breaking-changes
- data-migration-complexities
- data-migration-integrity-issues
- cross-system-data-synchronization-problems
- silent-data-corruption
- integration-difficulties
layout: solution
lang: de
en_slug: backward-compatible-data-formats
related_solutions:
- slug: backward-compatibility
  similarity: 0.85
- slug: standardized-data-formats
  similarity: 0.8
- slug: data-format-conversion
  similarity: 0.8
- slug: forward-compatibility
  similarity: 0.8
- slug: backward-compatible-apis
  similarity: 0.8
- slug: data-formats
  similarity: 0.8
---

## Description

Abwärtskompatible Datenformate sind Schema-Designs — unter Nutzung von Formaten wie Avro, Protocol Buffers oder JSON Schema —, die es Produzenten und Konsumenten von Daten erlauben, sich unabhängig weiterzuentwickeln, weil neue Felder als optional mit Standardwerten hinzugefügt werden, bestehende Felder nie umgenutzt werden und Entfernungen erst nach einer Deprecation-Periode geschehen, sobald alle Konsumenten abgewandert sind. Eine Schema-Registry und Validierung am Punkt der Datenaufnahme setzen diese Regeln mechanisch durch und fangen eine inkompatible Änderung ab, bevor sie Daten nachgelagert korrumpiert, statt danach. Dies ist wichtig für Legacy-Systeme, weil Datenformate dort häufig ad hoc designt wurden, ohne jegliche Evolutionsstrategie, sodass Konsumenten und Produzenten implizit an eine exakte Form der Daten gekoppelt sind, und jede Formatänderung — selbst eine, die geringfügig aussieht — riskiert, still Systeme zu brechen, die nie gebaut wurden, um unerwartete oder fehlende Felder zu tolerieren. Die Einführung expliziter Schema-Evolutionsregeln rüstet diese fehlende Disziplin nach: Sie erlaubt es einem Legacy-System, sein Datenformat schrittweise zu migrieren, Round-Trip-Kompatibilität (neuer Schreiber, alter Leser) zu verifizieren, bevor festgelegt wird, statt der üblichen Alternative eines einzelnen risikoreichen Umstiegs, bei dem jeder Produzent und Konsument gleichzeitig ändern muss. Die Kosten sind eine Beschränkung dessen, was ein einzelnes Release ändern kann, und die laufende Komplexität der Unterstützung älterer Schema-Versionen, was ein bewusster und üblicherweise lohnender Tradeoff gegen die Datenkorruption und Koordinationsfehler ist, die ungesteuerte Formatänderungen in eng verbundenen Legacy-Umgebungen tendenziell produzieren.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Nutzen Sie Schema-Formate, die Evolution unterstützen, wie Avro, Protocol Buffers oder JSON Schema mit optionalen Feldern
- Fügen Sie neue Felder als optional mit Standardwerten hinzu, sodass ältere Leser die Daten ohne Änderung verarbeiten können
- Entfernen oder benennen Sie Felder nie in einem einzigen Schritt um; deprekieren Sie zuerst und entfernen Sie erst, nachdem alle Konsumenten migriert sind
- Implementieren Sie Schema-Validierung an Aufnahmepunkten, um inkompatible Daten früh abzufangen
- Versionieren Sie Ihre Datenformate explizit und pflegen Sie eine Schema-Registry
- Testen Sie Daten-Round-Trip-Kompatibilität: Schreiben Sie mit dem neuen Format, lesen Sie mit dem alten Leser, und verifizieren Sie Korrektheit

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Ermöglicht unabhängige Evolution von Produzenten und Konsumenten in unterschiedlichen Release-Zyklen
- Verhindert Datenverlust oder -korruption während Formatübergängen
- Verringert den Bedarf an koordinierten Big-Bang-Migrationen über Systeme hinweg

**Kosten und Risiken:**
- Schema-Evolutionsregeln beschränken, welche Arten von Änderungen in einem einzelnen Release möglich sind
- Die Aufrechterhaltung der Kompatibilität mit sehr alten Formatversionen häuft Komplexität an
- Standardwerte für neue Felder repräsentieren möglicherweise nicht immer korrekte Geschäftssemantik
- Schema-Registries und Validierungsinfrastruktur fügen operativen Overhead hinzu

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Einzelhandelsunternehmen migrierte seine Event-Streaming-Plattform von einem benutzerdefinierten CSV-Format zu Avro mit einer Schema-Registry. Während des Übergangs sendeten Produzenten Events im neuen Avro-Format mit allen Legacy-Feldern als erforderlich beibehalten und neuen Feldern als optional markiert. Nachgelagerte Konsumenten wurden über einen Zeitraum von sechs Monaten ohne Datenverlust aktualisiert, und die Schema-Registry verhinderte drei versehentliche Breaking Changes davon, Produktion in diesem Zeitraum zu erreichen.
