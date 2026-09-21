---
title: Abwärtskompatible Schema-Migrationen
description: Berücksichtigung der Abwärtskompatibilität bei Datenbankschemata und
  Migrationen.
category:
- Database
- Architecture
problems:
- database-schema-design-problems
- data-migration-complexities
- data-migration-integrity-issues
- schema-evolution-paralysis
- deployment-risk
- breaking-changes
- entity-attribute-value-overuse
layout: solution
lang: de
en_slug: backward-compatible-schema-migrations
related_solutions:
- slug: backward-compatible-data-formats
  similarity: 0.75
- slug: evolutionary-database-design
  similarity: 0.75
- slug: backward-compatibility
  similarity: 0.75
- slug: backward-compatible-apis
  similarity: 0.7
- slug: automated-migration-tools
  similarity: 0.7
- slug: database-abstraction
  similarity: 0.65
---

## Description

Abwärtskompatible Schema-Migrationen wenden das Expand-and-Contract-Muster auf die Evolution von Datenbankschemata an: Eine neue Spalte oder Tabelle wird zuerst hinzugefügt, bestehende Daten werden durch einen Hintergrundprozess nachgefüllt oder transformiert, Anwendungscode wird aktualisiert, um die neue Struktur zu nutzen, während die alte weiterhin toleriert wird, und erst in einem späteren, separaten Deployment wird die alte Struktur schließlich entfernt. Das, was wie eine einzelne Schemaänderung aussieht, in mehrere sequenzielle, unabhängig deploybare Schritte aufzuteilen, ist genau das, was es erlaubt, dass sich das Datenbankschema und der Anwendungscode auf unterschiedlichen, überlappenden Zeitplänen ändern, statt perfekt im Gleichschritt. Dies ist wichtig in Legacy-Systemen, weil ihre Datenbanken typischerweise groß, langlebig sind und von mehr als nur der Anwendung gelesen werden, die das Schema besitzt — Berichtswerkzeuge, andere Services und Batch-Jobs könnten dieselben Tabellen direkt abfragen —, sodass ein naives einstufiges Umbenennen oder Löschen einer Spalte riskiert, Konsumenten zu brechen, die das Team nicht vollständig kontrolliert oder überhaupt kennt. Der mehrstufige Ansatz macht auch das Rollback allein des Anwendungscodes möglich, ohne die Datenbank zu berühren, was genau das Szenario ist, das ein riskantes Legacy-Deployment am meisten braucht, da das Rückgängigmachen einer Schemaänderung auf einer lebenden, mehrere Terabyte großen Tabelle oft weit gefährlicher ist als das Zurücksetzen von Anwendungscode. Die Kosten sind Koordinations-Overhead: Mehrere Releases müssen verfolgen, in welcher Migrationsphase sich die Umgebung befindet, und temporäre Duplizierung von Spalten fügt Übergangskomplexität hinzu, die schließlich bereinigt werden muss, statt sich unbegrenzt anzuhäufen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Nutzen Sie Expand-and-Contract-Migrationen: Fügen Sie zuerst die neue Spalte oder Tabelle hinzu, migrieren Sie dann Daten, entfernen Sie dann die alte Struktur
- Benennen oder löschen Sie Spalten nie in einem einzigen Deployment; nutzen Sie einen mehrstufigen Prozess über Releases hinweg
- Machen Sie neue Spalten nullable oder bieten Sie Standardwerte, sodass die alte Anwendungsversion weiterhin in die Datenbank schreiben kann
- Führen Sie Schema-Migrationen in einem separaten Deployment-Schritt vor Anwendungscodeänderungen aus
- Testen Sie Migrationen gegen eine Kopie eines produktionsgroßen Datensatzes, um Performance- und Kompatibilitätsprobleme abzufangen
- Pflegen Sie eine Migrationskompatibilitätsmatrix, die zeigt, welche Anwendungsversionen mit welchen Schemaversionen funktionieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Ermöglicht Zero-Downtime-Deployments durch Entkopplung von Schemaänderungen von Anwendungs-Releases
- Erlaubt Rollback von Anwendungscode, ohne die Datenbank zurückzusetzen
- Verringert das Risiko von Datenverlust während der Schema-Evolution

**Kosten und Risiken:**
- Mehrstufige Migrationen dauern länger und erfordern Koordination über mehrere Releases hinweg
- Temporäre Duplizierung von Spalten oder Tabellen erhöht Speicher- und Abfragekomplexität
- Teams müssen verfolgen, in welcher Migrationsphase sich jede Umgebung befindet
- Komplexe Migrationen könnten Backfill-Jobs erfordern, die gegen große Datensätze laufen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Gesundheitsplattform musste eine einzelne Adress-Textspalte in strukturierte Felder (Straße, Stadt, Postleitzahl) über eine Datenbank mit 40 Millionen Patientendatensätzen aufteilen. Unter Nutzung des Expand-and-Contract-Musters fügte das Team zuerst die neuen Spalten als nullable hinzu, deployte einen Hintergrundjob, um bestehende Adressen zu parsen und nachzufüllen, aktualisierte die Anwendung, um sowohl in alte als auch neue Spalten zu schreiben, und entfernte schließlich zwei Releases später die alte Spalte. Die gesamte Migration wurde ohne Ausfallzeit und ohne Datenverlust abgeschlossen.
