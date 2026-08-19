---
title: Schichtenarchitektur
description: Aufteilung des Softwaresystems in logische Schichten mit klaren
  Verantwortlichkeiten.
category:
- Architecture
problems:
- spaghetti-code
- high-coupling-low-cohesion
- monolithic-architecture-constraints
- tangled-cross-cutting-concerns
- difficult-code-comprehension
- tight-coupling-issues
- ripple-effect-of-changes
- single-entry-point-design
layout: solution
lang: de
en_slug: layered-architecture
related_solutions:
- slug: abstraction-layers
  similarity: 0.75
- slug: hexagonal-architecture
  similarity: 0.75
- slug: microservices-architecture
  similarity: 0.75
- slug: high-cohesion
  similarity: 0.75
- slug: adapter
  similarity: 0.7
- slug: dependency-injection
  similarity: 0.7
---

## Description

Schichtenarchitektur organisiert ein System in einen Stapel horizontaler Schichten — typischerweise Präsentation, Geschäftslogik und Datenzugriff —, wobei jede Schicht eine definierte Schnittstelle offenlegt und nur von der direkt darunterliegenden Schicht abhängt, nie umgekehrt. Die Abhängigkeitsregel ist der Mechanismus, der die eigentliche Arbeit leistet: Indem sie einer Schicht verbietet, über ihren unmittelbaren Nachbarn hinauszugreifen, verhindert sie, dass Präsentationscode direkt die Datenbank berührt oder Geschäftsregeln in UI-Controller lecken — genau die Art Verflechtung, die sich in unverwaltetem Legacy-Code über die Zeit ansammelt. In einem Legacy-System-Kontext geht es bei Schichtung oft weniger um Design von Grund auf als um Archäologie und Extraktion: zu identifizieren, wo SQL, Validierungslogik und Rendering-Code in derselben Datei oder Klasse verschachtelt wurden, und die Verantwortlichkeiten eine Verletzung nach der anderen in ihre richtige Schicht herauszuschneiden. Weil jede Schicht unabhängig getestet und modifiziert werden kann, sobald Grenzen etabliert sind, schrumpft der Explosionsradius einer Änderung von „überall in der Codebasis" auf „innerhalb einer Schicht", was direkt den Kaskadeneffekt von Änderungen und das Problem schwer verständlichen Codes bekämpft, die verworrenen Legacy-Code plagen. Schichtung beseitigt Kopplung nicht — sie organisiert sie —, sodass ihr Wert in der Modernisierung daher kommt, die verbleibenden Abhängigkeiten sichtbar, vorhersagbar und durchsetzbar statt implizit und verstreut zu machen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Definieren Sie klare Schichten wie Präsentation, Geschäftslogik und Datenzugriff, jede mit expliziten Verantwortlichkeiten
- Etablieren Sie eine Abhängigkeitsregel: Jede Schicht darf nur von der direkt darunterliegenden Schicht abhängen
- Identifizieren Sie Verstöße im Legacy-Code, wo Präsentationscode direkt auf die Datenbank zugreift oder Geschäftslogik in UI-Controller eingebettet ist
- Refaktorieren Sie inkrementell, indem Sie fehlplatzierte Logik in die passende Schicht extrahieren
- Nutzen Sie Paket- oder Modulnamenskonventionen, die die geschichtete Struktur widerspiegeln
- Führen Sie Schnittstellen an Schichtgrenzen ein, damit Implementierungen unabhängig ersetzt werden können
- Setzen Sie Schichtgrenzen durch architektonische Fitness Functions oder statische Analysewerkzeuge durch

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Bietet eine gut verstandene Struktur, der die meisten Entwickler sofort folgen können
- Isoliert Änderungen auf eine einzige Schicht, was den Explosionsradius von Modifikationen verringert
- Ermöglicht unabhängiges Testen jeder Schicht durch gut definierte Schnittstellen
- Macht die Codebasis navigierbar durch eine vorhersagbare Organisation

**Kosten und Risiken:**
- Strikte Schichtung kann zu Pass-Through-Methoden führen, die Boilerplate ohne Wert hinzufügen
- Passt möglicherweise nicht gut zu übergreifenden Belangen wie Logging, Sicherheit oder Transaktionsmanagement
- Kann zur Zwangsjacke werden, wenn zu strikt durchgesetzt, was pragmatische Abkürzungen verhindert
- Das Nachrüsten von Schichten in tief verflochtenen Legacy-Code erfordert erheblichen Aufwand

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Regierungsbehörde pflegte ein Legacy-Fallmanagementsystem, in dem JSP-Seiten SQL-Abfragen, Geschäftsvalidierung und HTML-Rendering in derselben Datei enthielten. Die Modifikation einer Geschäftsregel erforderte das Bearbeiten von Präsentationscode, und Datenbankänderungen brachen die UI auf unvorhersehbare Weisen. Das Team führte eine Drei-Schichten-Architektur ein, extrahierte zunächst alles SQL in eine Datenzugriffsschicht mit Repository-Klassen und verschob dann Validierung und Geschäftsregeln in eine Service-Schicht. Die JSP-Seiten wurden auf reine Präsentationsbelange reduziert. Diese Trennung erlaubte dem Team später, das JSP-Frontend durch eine React-Anwendung zu ersetzen, während Service- und Datenzugriffsschicht unverändert blieben.
