---
title: Dependency Pinning
description: Fixierung externer Abhängigkeitsversionen für reproduzierbare, kompatible
  Builds.
category:
- Operations
- Dependencies
problems:
- dependency-version-conflicts
- deployment-environment-inconsistencies
- configuration-drift
- breaking-changes
- deployment-risk
- increasing-brittleness
- abi-compatibility-issues
layout: solution
lang: de
en_slug: dependency-pinning
related_solutions:
- slug: third-party-dependency-check
  similarity: 0.7
- slug: dependency-management-strategy
  similarity: 0.7
- slug: containerization
  similarity: 0.65
- slug: cross-version-testing
  similarity: 0.65
- slug: dependency-injection
  similarity: 0.65
- slug: rollback-mechanisms
  similarity: 0.65
---

## Description

Dependency Pinning fixiert die exakte Version jeder direkten und transitiven Abhängigkeit, von der ein System abhängt, sodass ein Build oder Deployment jedes Mal, wenn es läuft, auf dieselbe Menge an Paketen auflöst, unabhängig davon, wann oder wo es ausgeführt wird. Statt Versionsbereiche zur Build-Zeit dynamisch auflösen zu lassen, erfasst Pinning präzise Versionsidentifikatoren — typischerweise über Lock-Dateien oder explizit versionierte Manifeste — und behandelt jede Änderung an diesen Versionen als bewusste, überprüfbare Aktion statt als beiläufigen Nebeneffekt eines Neu-Builds. In Legacy-Systemen, wo Abhängigkeitsgraphen oft über viele Jahre tief und verworren gewachsen sind, ohne dass jemand genau nachverfolgt hätte, welche Versionen im Spiel waren, verwandelt diese Praxis eine unsichtbare, sich ständig verschiebende Grundlage in eine bekannte, stabile. Sie wirkt direkt der Klasse von Fehlern entgegen, bei denen sich ein System über Umgebungen hinweg oder nach einem routinemäßigen Neu-Build unterschiedlich verhält, einfach weil eine transitive Abhängigkeit auf eine neuere Version mit subtil unterschiedlichem Verhalten aufgelöst hat. Dies ist besonders während der Modernisierungsarbeit wichtig, wo Teams eine stabile Baseline brauchen, über die sie nachdenken können, bevor sie Änderungen einführen — ohne Pinning wird es unmöglich zu sagen, ob eine Regression durch das eigene Refactoring des Teams oder durch ein nicht verwandtes Upstream-Update verursacht wurde. Pinning friert ein System nicht dauerhaft ein; es verschiebt Abhängigkeitsupdates von einem impliziten, unkontrollierten Ereignis zu einem expliziten, geplanten, das bewusst getestet und zurückgerollt werden kann.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Nutzen Sie Lock-Dateien (package-lock.json, Gemfile.lock, poetry.lock), um exakte Versionen aller transitiven Abhängigkeiten zu fixieren
- Committen Sie Lock-Dateien in die Versionskontrolle, sodass alle Entwickler und CI-Systeme identische Abhängigkeitsbäume nutzen
- Fixieren Sie Basis-Images und Werkzeugversionen in Container-Builds für reproduzierbare Builds
- Etablieren Sie einen regelmäßigen Takt zur Überprüfung und Aktualisierung fixierter Versionen, statt sie unbegrenzt zu belassen
- Nutzen Sie Abhängigkeits-Scanning-Werkzeuge, um fixierte Versionen mit bekannten Schwachstellen zu identifizieren
- Dokumentieren Sie die Begründung für jede Versionsfixierung, die von der neuesten verfügbaren Version abweicht

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Stellt sicher, dass Builds über Umgebungen und Zeit hinweg reproduzierbar sind
- Verhindert unerwartete Ausfälle durch transitive Abhängigkeitsupdates
- Erleichtert die Diagnose von Problemen, weil genau bekannt ist, welche Versionen genutzt werden

**Kosten und Risiken:**
- Fixierte Abhängigkeiten können veralten und Sicherheitslücken sowie fehlende Fehlerbehebungen anhäufen
- Die Aktualisierung eines tief fixierten Abhängigkeitsbaums kann kaskadierende Versionskonflikte auslösen
- Teams könnten Pinning als Vorwand nutzen, um notwendige Abhängigkeitsupdates zu vermeiden
- Unterschiedliche Pinning-Strategien über Teams hinweg können Inkonsistenz erzeugen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Webanwendungsteam erlebte intermittierende CI-Fehler, die lokal nicht reproduziert werden konnten. Die Untersuchung zeigte, dass der CI-Server eine leicht andere Version einer transitiven Abhängigkeit auflöste als die Maschinen der Entwickler. Nach der Einführung strikten Dependency Pinnings mit committeten Lock-Dateien und fixierten CI-Werkzeugversionen wurde der Build vollständig reproduzierbar. Das Team plante auch monatliche Abhängigkeitsupdate-Reviews, die zwei Sicherheitslücken in fixierten Bibliotheken abfingen, bevor sie ausgenutzt wurden.
