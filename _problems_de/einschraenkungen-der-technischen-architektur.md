---
title: Einschränkungen der technischen Architektur
description: Das Design der Systemarchitektur schafft Beschränkungen, die Performance,
  Skalierbarkeit, Wartbarkeit oder Entwicklungsgeschwindigkeit limitieren.
category:
- Architecture
- Code
- Performance
related_problems:
- slug: architectural-mismatch
  similarity: 0.65
- slug: monolithic-architecture-constraints
  similarity: 0.65
- slug: stagnant-architecture
  similarity: 0.65
- slug: complex-implementation-paths
  similarity: 0.6
- slug: tool-limitations
  similarity: 0.6
- slug: integration-difficulties
  similarity: 0.55
solutions:
- strangler-fig-pattern
- api-deprecation-policy
- architecture-conformity-analysis
- forward-compatibility
- high-availability-architectures
- security-architecture-analysis
- security-by-design
- architecture-roadmap
- modularization-and-bounded-contexts
- incremental-refactoring
- application-portfolio-inventory
layout: problem
lang: de
en_slug: technical-architecture-limitations
---

## Description

Einschränkungen der technischen Architektur treten auf, wenn das fundamentale Design und die Struktur eines Softwaresystems Beschränkungen schaffen, die Performance, Skalierbarkeit, Wartbarkeit oder Entwicklungsgeschwindigkeit behindern. Diese Einschränkungen entstehen oft aus architektonischen Entscheidungen, die früh in der Entwicklung getroffen wurden und problematisch werden, während das System wächst oder sich Anforderungen ändern. Anders als Bugs oder Implementierungsprobleme erfordern architektonische Einschränkungen fundamentale Designänderungen zur Lösung.

## Indicators ⟡

- Die Systemperformance verbessert sich trotz Hardware-Upgrades nicht
- Das Hinzufügen neuer Features erfordert unverhältnismäßigen Aufwand aufgrund architektonischer Beschränkungen
- Das System kann trotz angemessener Ressourcen nicht skalieren, um wachsende Nachfrage zu erfüllen
- Die Entwicklungsgeschwindigkeit sinkt, während das System an Komplexität wächst
- Workarounds sind nötig, um Funktionalität zu implementieren, die eigentlich einfach sein sollte

## Symptoms ▲

- [Verzögerte Wertlieferung](verzoegerte-wertlieferung.md)
<br/>  Architektonische Beschränkungen zwingen Entwickler, fundamentale Designprobleme zu umgehen, was die Feature-Entwicklung erheblich verlangsamt.
- [Anhäufung von Workarounds](anhaeufung-von-workarounds.md)
<br/>  Entwickler erstellen Workarounds, um architektonische Beschränkungen zu umgehen, statt einfache Lösungen zu implementieren.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Das Hinzufügen neuer Features erfordert unverhältnismäßigen Aufwand, weil Änderungen innerhalb begrenzender architektonischer Beschränkungen funktionieren müssen.
- [Technologie-Lock-in](technologie-lock-in.md)
<br/>  Tief eingebettete architektonische Entscheidungen machen es unerschwinglich teuer, Technologien zu wechseln oder moderne Ansätze zu übernehmen.

## Causes ▼

- [Stagnierende Architektur](stagnierende-architektur.md)
<br/>  Eine Architektur, die sich nicht weiterentwickelt hat, um sich ändernde Anforderungen zu erfüllen, wird über die Zeit zunehmend einschränkend.
- [Angehäufte Entscheidungsschulden](angehaeufte-entscheidungsschulden.md)
<br/>  Frühe architektonische Entscheidungen, die nie überarbeitet wurden, verdichten sich zu fundamentalen Beschränkungen, während das System wächst.
- [Einschränkungen durch monolithische Architektur](einschraenkungen-durch-monolithische-architektur.md)
<br/>  Ein monolithisches Design schränkt inhärent die Fähigkeit ein, verschiedene Teile des Systems unabhängig zu skalieren, zu modifizieren oder weiterzuentwickeln.

## Detection Methods ○

- **Performance-Profiling:** Identifikation, ob Performance-Probleme aus architektonischen Beschränkungen stammen
- **Skalierbarkeitstests:** Testen, ob die Architektur erwartetes Wachstum handhaben kann
- **Verfolgung der Entwicklungsgeschwindigkeit:** Überwachung, ob die Feature-Entwicklung über die Zeit langsamer wird
- **Architektonische Komplexitätsanalyse:** Bewertung, ob die Systemkomplexität durch Funktionalität gerechtfertigt ist
- **Technologie-Eignungsbewertung:** Bewertung, ob die aktuelle Architektur den Systemanforderungen entspricht

## Examples

Eine Webanwendung wurde mit einer einzigen monolithischen Datenbank designt, die alle Datenspeicherung handhabt. Während die Anwendung wächst, werden Datenbankabfragen langsamer, und die einzige Datenbank wird zu einem Engpass für alle Operationen. Die Architektur macht es unmöglich, verschiedene Teile des Systems unabhängig zu skalieren, und jedes neue Feature muss innerhalb der Beschränkungen des Einzeldatenbankdesigns funktionieren. Versuche, die Performance zu optimieren, sind begrenzt, weil die fundamentale Architektur keine horizontale Skalierung oder Datenpartitionierung unterstützt. Ein weiteres Beispiel betrifft ein Nachrichtensystem, das mit synchronen Kommunikationsmustern designt wurde, die für kleine Volumina gut funktionieren, aber kaskadierende Ausfälle und Timeout-Probleme schaffen, wenn das Nachrichtenvolumen zunimmt. Die synchrone Architektur macht es unmöglich, Lastspitzen anmutig zu handhaben, und das gesamte System wird unter normalen Produktionsbedingungen unzuverlässig.
