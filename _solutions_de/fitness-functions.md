---
title: Fitness Functions
description: Regelmäßige Überprüfung der Einhaltung von Architekturrichtlinien.
category:
- Architecture
- Testing
problems:
- stagnant-architecture
- high-coupling-low-cohesion
- architectural-mismatch
- quality-degradation
- high-technical-debt
- ripple-effect-of-changes
- inconsistent-codebase
- tight-coupling-issues
- premature-technology-introduction
- circular-dependency-problems
layout: solution
lang: de
en_slug: fitness-functions
related_solutions:
- slug: architecture-conformity-analysis
  similarity: 0.8
- slug: architecture-reviews
  similarity: 0.75
- slug: architecture-governance
  similarity: 0.7
- slug: architecture-decision-records
  similarity: 0.7
- slug: static-analysis-and-linting
  similarity: 0.7
- slug: architecture-roadmap
  similarity: 0.7
---

## Description

Fitness Functions sind automatisierte Tests, kontinuierlich als Teil der CI-Pipeline ausgeführt, die eine definierte architektonische Eigenschaft prüfen — Kopplungsgrenzen zwischen Modulen, Abwesenheit zyklischer Abhängigkeiten, Latenzschwellen, Deployment-Unabhängigkeit — und den Build fehlschlagen lassen, wenn diese Eigenschaft über einen vereinbarten Schwellenwert hinaus verkommt, wodurch Architektur dieselbe Art kontinuierlicher, objektiver Verifikation erhält, die Unit-Tests der Geschäftslogik geben. Legacy-Systeme in der Modernisierung sind besonders anfällig für architektonische Regression: Ohne eine automatisierte Prüfung sieht eine Änderung, die still einen modulübergreifenden Datenbankzugriff oder eine zyklische Abhängigkeit wieder einführt, identisch aus wie jeder andere bestehende Commit, und die Drift wird erst viel später sichtbar, in einem teuren Architektur-Review oder einem Produktionsvorfall, zu welchem Zeitpunkt sie rückgängig zu machen weit teurer ist. Fitness Functions für die während eines bestimmten Modernisierungsvorhabens am meisten gefährdeten Eigenschaften zu implementieren — Kopplung, wenn das Ziel die Zerlegung eines Monolithen ist; Latenz, wenn das Ziel die Bewahrung der Performance durch eine Neufassung ist — fängt diese Regressionen in dem Moment ab, in dem sie eingeführt werden, und gibt dem Team ein objektives, kontinuierlich sichtbares Maß dafür, ob sich die Architektur tatsächlich auf ihr erklärtes Ziel zubewegt oder nur weitere Ausnahmen davon anhäuft. Der Ansatz hängt davon ab, überhaupt klare Architekturziele zu haben, woran es organisch gewachsenen Legacy-Systemen oft fehlt, und davon, Schwellenwerte durchdacht zu setzen, da zu strikte Fitness Functions zu viele bereits bestehende Verstöße markieren, um nützlich zu sein, während Funktionen, die die falsche Eigenschaft messen, ein falsches Gefühl architektonischer Gesundheit erzeugen.

## How to Apply ◆

> In Legacy-Systemen bieten Fitness Functions automatisierte, kontinuierliche Verifikation, dass sich die Architektur in die richtige Richtung entwickelt statt mit jeder Änderung still zu verkommen.

- Definieren Sie messbare architektonische Eigenschaften, die für das Legacy-System zählen — Kopplung zwischen Modulen, Antwortzeitschwellen, Deployment-Unabhängigkeit, Abhängigkeitszahlen oder Abwesenheit zyklischer Abhängigkeiten.
- Implementieren Sie jede Fitness Function als automatisierten Test, der in der CI-Pipeline läuft und fehlschlägt, wenn eine architektonische Eigenschaft über ihren Schwellenwert hinaus verkommt.
- Beginnen Sie mit den architektonischen Eigenschaften, die während der Modernisierung am meisten gefährdet sind — wenn das Ziel etwa die Reduzierung der Kopplung ist, erstellen Sie eine Fitness Function, die Kopplungsgrenzen zwischen definierten Modulgrenzen misst und durchsetzt.
- Nutzen Sie bestehende Werkzeuge, wo möglich: ArchUnit für strukturelle Regeln, Performance-Testsuiten für Latenz-Fitness-Functions, Abhängigkeitsanalysewerkzeuge für Kopplungsmetriken.
- Setzen Sie anfängliche Schwellenwerte auf oder leicht besser als den aktuellen Zustand, um Regression zu verhindern, und verschärfen Sie die Schwellenwerte dann schrittweise, sobald sich die Architektur verbessert.
- Überprüfen Sie Fitness-Function-Ergebnisse in Architektur-Meetings, um den Modernisierungsfortschritt zu verfolgen und Bereiche zu identifizieren, in denen Architekturziele nicht erreicht werden.
- Erstellen Sie Fitness Functions sowohl für positive Ziele (die Architektur sollte diese Eigenschaften haben) als auch für negative Einschränkungen (die Architektur darf diese Antipatterns nicht entwickeln).

## Tradeoffs ⇄

> Fitness Functions bieten kontinuierliches architektonisches Feedback, erfordern aber klare Architekturziele und Investition in Automatisierung.

**Vorteile:**

- Verhindert architektonische Regression durch automatische Erkennung, wenn Änderungen definierte architektonische Eigenschaften verschlechtern.
- Macht architektonische Verbesserung messbar und ermöglicht datengestützte Gespräche mit Stakeholdern über den Modernisierungsfortschritt.
- Fängt architektonische Verstöße zur Build-Zeit ab statt sie während teurer Architektur-Reviews oder Produktionsvorfälle zu entdecken.
- Richtet das gesamte Team auf Architekturziele aus, indem sie explizit, automatisiert und kontinuierlich sichtbar gemacht werden.

**Kosten und Risiken:**

- Das Definieren sinnvoller Fitness Functions erfordert klare Architekturziele, die für organisch gewachsene Legacy-Systeme möglicherweise nicht existieren.
- Fitness Functions, die die falschen Eigenschaften messen, können ein falsches Gefühl architektonischer Gesundheit erzeugen.
- Übermäßig strikte Fitness Functions können die Entwicklung verlangsamen, indem sie zu viele Verstöße in einer Legacy-Codebasis mit vielen bestehenden Problemen markieren.
- Manche architektonischen Qualitäten (konzeptionelle Integrität, angemessene Abstraktionsebenen) sind schwer als automatisierte Fitness Functions auszudrücken.

## How It Could Be

> Das folgende Szenario zeigt, wie Fitness Functions die Legacy-System-Modernisierung leiten und schützen.

Ein Finanztechnologieunternehmen zerlegte eine monolithische Handelsplattform in domänenausgerichtete Module als ersten Schritt in Richtung Microservices. Sie definierten fünf Fitness Functions: keine zyklischen Abhängigkeiten zwischen Modulen, der Fan-out jedes Moduls (Anzahl der Module, von denen es abhängt) darf vier nicht überschreiten, API-Antwortzeiten müssen unter 200ms beim 95. Perzentil bleiben, kein Modul darf direkt auf die Datenbanktabellen eines anderen Moduls zugreifen, und die Testabdeckung für Code an Modulgrenzen muss 85 % überschreiten. Diese Fitness Functions liefen bei jedem Pull Request und in nächtlichen Builds. Im ersten Monat fingen die Fitness Functions 12 Pull Requests ab, die neuen modulübergreifenden Datenbankzugriff eingeführt hätten, und drei, die zyklische Abhängigkeiten erzeugt hätten. Über sechs Monate sank der durchschnittliche Modul-Fan-out von 7,2 auf 3,8, während das Team gemäß dem Feedback der Fitness Functions refaktorierte. Als eine Performance-Optimierung versehentlich die API-Latenz auf 350ms erhöhte, fing die Fitness Function dies ab, bevor die Änderung die Produktion erreichte.
