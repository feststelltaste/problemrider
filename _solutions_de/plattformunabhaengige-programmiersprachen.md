---
title: Plattformunabhängige Programmiersprachen
description: Nutzung von Programmiersprachen, die ohne Änderungen auf
  verschiedenen Systemen laufen.
category:
- Architecture
- Code
problems:
- technology-lock-in
- vendor-lock-in
- vendor-dependency-entrapment
- obsolete-technologies
- legacy-skill-shortage
- stagnant-architecture
layout: solution
lang: de
en_slug: platform-independent-programming-languages
related_solutions:
- slug: platform-independence
  similarity: 0.85
- slug: platform-independent-scripting-languages
  similarity: 0.8
- slug: cross-platform-frameworks
  similarity: 0.8
- slug: cross-platform-build-tools
  similarity: 0.8
- slug: platform-independent-data-storage
  similarity: 0.8
- slug: platform-independent-configuration-files
  similarity: 0.75
---

## Description

Plattformunabhängige Programmiersprachen sind Sprachen, deren kompilierte oder interpretierte Ausgabe unverändert über mehrere Betriebssysteme und Hardwarearchitekturen läuft, typischerweise indem sie auf eine portable Laufzeitumgebung zielen — die JVM, .NETs CLR, den eigenen Bytecode einer Sprache — oder indem sie zu statisch gelinkten Binärdateien ohne betriebssystemspezifische Abhängigkeiten kompilieren, wie Go es tut. Die Wahl einer solchen Sprache für neue oder Ersatzkomponenten entkoppelt das System von den Annahmen, die in plattformspezifischen Sprachen wie klassischem VB6 oder Delphi eingebettet sind, deren Laufzeitumgebung und Tooling die Codebasis permanent an ein einziges Betriebssystem binden. In der Legacy-Modernisierung zählt dies, weil plattformgebundene Sprachen eine der hartnäckigeren Formen von Technologie-Lock-in sind: Während sich die Infrastruktur einer Organisation zu Linux-basierten oder Cloud-nativen Umgebungen verlagert, wird eine Legacy-Codebasis, die in einer reinen Windows-Sprache geschrieben ist, zu einem Hindernis für diese Verlagerung, und der Talentpool, der bereit und fähig ist, sie zu warten, schrumpft weiter. Die Migration weg von einer plattformspezifischen Sprache ist selten eine einzelne Umschaltung; sie verläuft typischerweise Modul für Modul, mit Interoperabilitätsmechanismen wie REST-APIs oder Message Queues, die die neuen, portablen Komponenten und den verbleibenden Legacy-Code während eines mehrjährigen Übergangs überbrücken. Der Zielkonflikt ist, dass portable Sprachen bei roher Performance für rechenintensive Arbeit hinter plattformnativem Code zurückbleiben können, und die Migration selbst erheblichen Ingenieuraufwand verbraucht, bevor irgendeine funktionale Verbesserung für Nutzer sichtbar wird.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Bewerten Sie den aktuellen Technologie-Stack auf plattformspezifische Sprachabhängigkeiten wie nativen kompilierten Code oder betriebssystemspezifische APIs
- Bewerten Sie plattformübergreifende Sprachen (z. B. Java, Python, Go, Kotlin, C#/.NET) basierend auf Performance-, Ökosystem- und Teamkompetenzanforderungen des Projekts
- Planen Sie eine inkrementelle Migrationsstrategie, beginnend mit peripheren Modulen statt Kerngeschäftslogik
- Nutzen Sie Interoperabilitätsmechanismen (FFI, REST-APIs, Message Queues), damit neue plattformübergreifende Komponenten mit Legacy-plattformspezifischem Code koexistieren können
- Investieren Sie in Teamschulung für die Zielsprache, bevor Sie mit der groß angelegten Migration beginnen
- Etablieren Sie Coding-Standards, die plattformspezifische Idiome selbst innerhalb plattformübergreifender Sprachen vermeiden

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Beseitigt die Notwendigkeit, separate Codebasen für verschiedene Zielplattformen zu pflegen
- Erweitert den Talentpool, da plattformübergreifende Sprachen typischerweise größere Entwicklergemeinschaften haben
- Vereinfacht die Bereitstellung über heterogene Umgebungen hinweg
- Reduziert langfristige Wartungskosten durch Konsolidierung auf eine einzige portable Codebasis

**Kosten und Risiken:**
- Plattformübergreifende Sprachen können für rechenintensive Aufgaben geringere Performance haben als plattformnative Alternativen
- Sprachmigration erfordert erhebliche Investition in Neuschreibung und Revalidierung
- Manche plattformspezifischen Features sind möglicherweise über plattformübergreifende Sprachabstraktionen nicht zugänglich
- Die Teamproduktivität sinkt während der Übergangsperiode, während Entwickler die neue Sprache lernen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Telekommunikationsunternehmen unterhielt ein Netzwerküberwachungssystem, das in Delphi geschrieben war und nur auf Windows lief. Als sich das Unternehmen zu Linux-basierter Infrastruktur bewegte, wurde die Delphi-Codebasis zu einem Engpass. Das Team wählte Go wegen seiner plattformübergreifenden Kompilierung, seines starken Nebenläufigkeitsmodells und der Single-Binary-Bereitstellung. Sie migrierten Modul für Modul über zwölf Monate, wobei sie REST-APIs nutzten, um die neuen Go-Dienste mit verbleibenden Delphi-Komponenten zu verbinden. Das finale System kompilierte und lief identisch auf Windows und Linux, was eine schrittweise Infrastrukturmigration ohne Betriebsunterbrechung ermöglichte.
