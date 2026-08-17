---
title: Abhängigkeit vom Zulieferer
description: Externe Anbieter kontrollieren kritische Teile des Systems, was die
  organisatorische Flexibilität verringert und das Lock-in-Risiko erhöht.
category:
- Architecture
- Dependencies
- Management
related_problems:
- slug: vendor-dependency
  similarity: 0.9
- slug: vendor-dependency-entrapment
  similarity: 0.75
- slug: implementation-partner-dependency
  similarity: 0.7
- slug: vendor-lock-in
  similarity: 0.7
- slug: knowledge-dependency
  similarity: 0.55
- slug: vendor-relationship-strain
  similarity: 0.55
solutions:
- dependency-management-strategy
- vendor-management-practice
- anti-corruption-layer
- abstraction-layers
- adapter
- standardized-interfaces
- third-party-dependency-check
- application-portfolio-inventory
- technology-radar
- system-decommissioning
layout: problem
lang: de
en_slug: dependency-on-supplier
---

## Description

Abhängigkeit vom Zulieferer entsteht, wenn eine Organisation übermäßig auf externe Anbieter für kritische Systemkomponenten, Dienste oder Expertise angewiesen wird, was strategische Verwundbarkeiten schafft und die Autonomie verringert. Diese Abhängigkeit kann sich als technisches Lock-in, Wissensabhängigkeit oder operative Abhängigkeit äußern, was es schwierig oder teuer macht, Anbieter zu wechseln oder Fähigkeiten intern aufzubauen.

## Indicators ⟡

- Kritische Systemfunktionalität hängt von anbieterspezifischen Technologien oder Diensten ab
- Der Organisation fehlt interne Expertise, um vom Anbieter gelieferte Komponenten zu warten oder zu ändern
- Die Wechselkosten zu alternativen Anbietern sind unerschwinglich hoch
- Der Anbieter hat erhebliche Kontrolle über Roadmap, Preisgestaltung oder Servicelevel
- Die Organisation kann nicht effektiv arbeiten, wenn die Anbieterbeziehung endet

## Symptoms ▲

- [Technologie-Lock-in](technologie-lock-in.md)
<br/>  Das Vertrauen auf die proprietäre Technologie eines Anbieters macht einen Wechsel unerschwinglich teuer, was die Organisation in dessen Ökosystem einsperrt.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Vom Anbieter kontrollierte Komponenten kommen oft mit eskalierenden Lizenz- und Support-Kosten, die die Organisation nicht wegverhandeln kann.
- [Verringerte Teamflexibilität](verringerte-teamflexibilitaet.md)
<br/>  Die Abhängigkeit von einem Anbieter schränkt die Fähigkeit des Teams ein, Technologien oder Ansätze zu wählen, die am besten zu seinen Bedürfnissen passen.
- [Wissensabhängigkeit](wissensabhaengigkeit.md)
<br/>  Wenn kritisches Wissen beim Anbieter liegt statt bei der Organisation, können interne Teams das System nicht unabhängig warten oder Probleme beheben.
- [Verzögerte Wertlieferung](verzoegerte-wertlieferung.md)
<br/>  Änderungen und Verbesserungen müssen auf den Zeitplan und die Prioritäten des Anbieters warten, was die Wertlieferung an Nutzer verzögert.

## Causes ▼

- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Die Wahl von Anbieterlösungen für kurzfristige Bequemlichkeit, ohne langfristige Lock-in-Risiken zu bewerten, schafft Anbieterabhängigkeit.
- [Wissenslücken](wissensluecken.md)
<br/>  Fehlende interne Expertise in kritischen Technologiebereichen erzwingt das Vertrauen auf externe Anbieter.
- [Schlechtes Vertragsdesign](schlechtes-vertragsdesign.md)
<br/>  Verträge, die es versäumen, vor Lock-in zu schützen oder Wissenstransfer sicherzustellen, schaffen Bedingungen für tiefe Anbieterabhängigkeit.
- [Angehäufte Entscheidungsschulden](angehaeufte-entscheidungsschulden.md)
<br/>  Das Aufschieben von Build-vs.-Buy-Entscheidungen und das Nicht-Investieren in interne Fähigkeiten summiert die Anbieterabhängigkeit im Laufe der Zeit.

## Detection Methods ○

- **Anbieterabhängigkeits-Mapping:** Identifikation und Bewertung aller kritischen Anbieterabhängigkeiten
- **Wechselkostenanalyse:** Berechnung von Kosten und Aufwand, die für einen Anbieterwechsel bei kritischen Diensten nötig sind
- **Anbieter-Performance-Monitoring:** Nachverfolgung der Anbieter-Performance und Beziehungsgesundheit im Zeitverlauf
- **Bewertung alternativer Anbieter:** Bewertung der Verfügbarkeit und Tragfähigkeit alternativer Anbieter
- **Analyse interner Fähigkeitslücken:** Bewertung der Fähigkeit der Organisation, Anbieterabhängigkeiten zu verringern

## Examples

Ein Unternehmen baut sein gesamtes Kundenverwaltungssystem auf einer proprietären Plattform eines bestimmten Anbieters auf. Über fünf Jahre entwickeln sie Hunderte benutzerdefinierter Integrationen und Workflows, die spezifisch für diese Plattform sind. Als der Anbieter die Lizenzkosten erheblich erhöht und die Support-Qualität verringert, entdeckt das Unternehmen, dass eine Migration zu einer Alternative den Neubau der meisten seiner Systeme zu Kosten von Millionen von Dollar und Jahren an Aufwand erfordern würde. Ein weiteres Beispiel betrifft eine Organisation, die die gesamte Datenbankadministration an einen Anbieter auslagert und es versäumt, interne Datenbankexpertise aufrechtzuerhalten. Als Performance-Probleme auftreten, können sie Probleme nicht unabhängig diagnostizieren und müssen sich vollständig auf die Verfügbarkeit und Expertise des Anbieters verlassen, was zu verlängerten Ausfallzeiten und hohen Support-Kosten führt.
