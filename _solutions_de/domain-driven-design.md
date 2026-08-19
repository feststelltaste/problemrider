---
title: Domain-Driven Design
description: Strukturierung der Softwarearchitektur basierend auf der Geschäftsdomäne.
category:
- Architecture
- Code
problems:
- poor-domain-model
- complex-domain-model
- architectural-mismatch
- legacy-business-logic-extraction-difficulty
- monolithic-architecture-constraints
- high-coupling-low-cohesion
- stakeholder-developer-communication-gap
- inconsistent-naming-conventions
- over-reliance-on-utility-classes
- procedural-background
- god-object-anti-pattern
- poor-naming-conventions
- insufficient-design-skills
- procedural-programming-in-oop-languages
- entity-attribute-value-overuse
layout: solution
lang: de
en_slug: domain-driven-design
related_solutions:
- slug: domain-modeling
  similarity: 0.8
- slug: domain-aligned-architecture
  similarity: 0.8
- slug: bounded-contexts
  similarity: 0.75
- slug: domain-patterns
  similarity: 0.75
- slug: business-event-processing
  similarity: 0.75
- slug: hexagonal-architecture
  similarity: 0.7
---

## Description

Domain-Driven Design ist ein Ansatz zur Strukturierung von Software, sodass ihr Code direkt die Konzepte, die Sprache und die Grenzen der Geschäftsdomäne widerspiegelt, der sie dient, mittels einer gemeinsamen, universellen Sprache zwischen Entwicklern und Domänenexperten, expliziter Bounded Contexts und taktischer Muster wie Entities, Value Objects, Aggregates und Domain Events anstelle prozeduraler oder anämischer Datenmodelle. Legacy-Systeme driften über die Zeit häufig weit von diesem Ideal ab: Geschäftslogik häuft sich über gespeicherte Prozeduren, Service-Klassen und UI-Code an welchem Ort auch immer bequem war, als jedes Feature hinzugefügt wurde, und ein einzelner Begriff wie „Policy" kann am Ende in unterschiedlichen Teilen des Systems subtil unterschiedliche Dinge bedeuten, was zu einer anhaltenden Kommunikationslücke zwischen Stakeholdern und Entwicklern führt. DDD auf ein solches System anzuwenden bedeutet, bewusst zu identifizieren, wo diese Bounded Contexts tatsächlich liegen, oft über Workshops mit Domänenexperten, und dann die entsprechende Logik zu refaktorieren, sodass die Struktur und das Vokabular des Codes dazu passen, wie das Geschäft tatsächlich über diesen Teil der Domäne spricht und denkt. Weil der Aufwand, dieses gemeinsame Verständnis aufzubauen, erheblich ist, zahlt er sich am besten aus, wenn er auf die Kerndomäne konzentriert wird — den Teil des Systems, der dem Geschäft seine tatsächliche wettbewerbliche Differenzierung verleiht — statt gleichmäßig über jede Subdomäne verteilt zu werden, einschließlich generischer, undifferenzierter. DDD auf eine etablierte Legacy-Codebasis nachträglich anzuwenden ist notwendigerweise graduell und trägt echtes Risiko, ohne erfahrene Anleitung falsch angewandt zu werden, aber gut gemacht schrumpft es die Übersetzungsdistanz zwischen dem, was das Geschäft braucht, und dem, was der Code ausdrückt, was sich konkret als schnellere, weniger fehleranfällige Lieferung domänenspezifischer Features zeigt.

## How to Apply ◆

- Entwickeln Sie eine universelle Sprache, die zwischen Entwicklern und Domänenexperten geteilt wird, und ersetzen Sie den in Legacy-Code eingebetteten technischen Fachjargon.
- Identifizieren Sie Bounded Contexts innerhalb des Legacy-Systems und definieren Sie explizite Grenzen zwischen ihnen.
- Refaktorieren Sie Kerndomänenlogik mittels taktischer DDD-Muster (Entities, Value Objects, Aggregates, Domain Events), um prozedurale oder anämische Domänenmodelle zu ersetzen.
- Nutzen Sie Context Mapping, um zu dokumentieren, wie sich die Module des Legacy-Systems zueinander und zu externen Systemen verhalten.
- Priorisieren Sie DDD-Bemühungen auf die Kerndomäne (den Teil, der dem Geschäft Wettbewerbsvorteil verschafft), statt zu versuchen, es überall anzuwenden.
- Führen Sie Anti-Corruption Layers ein, um neue Domänenmodelle davor zu schützen, von Legacy-Systemstrukturen kontaminiert zu werden.

## Tradeoffs ⇄

**Vorteile:**
- Richtet die Codestruktur an Geschäftskonzepten aus, was das System für Entwickler und Stakeholder intuitiver macht.
- Reduziert die Lücke zwischen Geschäftsanforderungen und ihrer Implementierung.
- Bietet einen prinzipiengeleiteten Ansatz zur Zerlegung monolithischer Legacy-Systeme.
- Schafft ein gemeinsames Vokabular, das die Kommunikation zwischen technischen und geschäftlichen Teams verbessert.

**Kosten:**
- Erfordert erhebliche Investition in das Verständnis der Geschäftsdomäne, was Zeit von der Feature-Auslieferung abzieht.
- DDD-Konzepte haben eine steile Lernkurve und können ohne erfahrene Anleitung falsch angewandt werden.
- DDD nachträglich in ein Legacy-System einzubauen ist ein gradueller Prozess, der Jahre dauern kann.
- Übermäßige Anwendung von DDD auf einfache oder generische Subdomänen verschwendet Aufwand ohne proportionalen Nutzen.

## How It Could Be

Ein Legacy-Versicherungsunternehmen hat ein Kern-Policenverwaltungssystem, in dem Geschäftslogik über gespeicherte Prozeduren, Service-Klassen und UI-Code verstreut ist. Der Begriff „Policy" bedeutet in unterschiedlichen Teilen des Systems unterschiedliche Dinge, was zu häufigen Missverständnissen zwischen Underwritern und Entwicklern führt. Das Team bindet Domänenexperten in Workshops ein, um eine universelle Sprache zu etablieren und Bounded Contexts zu identifizieren: Underwriting, Schadensbearbeitung und Abrechnung haben jeweils ihre eigene Vorstellung einer Policy. Innerhalb des Underwriting-Kontexts refaktorieren sie das anämische Datenmodell in reiche Domänenobjekte mit Verhalten und ersetzen Hunderte Zeilen prozeduralen Validierungscodes durch ausdrucksstarke Domänenregeln. Der resultierende Code liest sich wie Geschäftsdokumentation, und neue Underwriting-Features, die zuvor Wochen zur Implementierung brauchten, können jetzt in Tagen geliefert werden, weil die Codestruktur dazu passt, wie das Geschäft über die Domäne denkt.
