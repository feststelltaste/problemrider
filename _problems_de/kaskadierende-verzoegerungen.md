---
title: Kaskadierende Verzögerungen
description: Verpasste Termine in einem Bereich verursachen Verzögerungen in abhängigen
  Arbeitssträngen, was einen Welleneffekt erzeugt, der mehrere Projekte und Teams
  betrifft.
category:
- Business
- Management
- Process
related_problems:
- slug: delayed-project-timelines
  similarity: 0.75
- slug: approval-dependencies
  similarity: 0.7
- slug: missed-deadlines
  similarity: 0.7
- slug: cascade-failures
  similarity: 0.65
- slug: extended-cycle-times
  similarity: 0.65
- slug: delayed-decision-making
  similarity: 0.65
solutions:
- iterative-development
- short-iteration-cycles
- capacity-based-planning
- work-in-progress-limits
- explicit-prioritization-framework
- value-stream-mapping
- team-boundaries-aligned-to-architecture
- regular-stakeholder-demonstrations
layout: problem
lang: de
en_slug: cascade-delays
---

## Description

Kaskadierende Verzögerungen entstehen, wenn Verzögerungen in einem Projekt oder Arbeitsstrang Verzögerungen in anderen abhängigen Projekten auslösen, was einen Dominoeffekt erzeugt, der die Auswirkung anfänglicher Terminverschiebungen verstärkt. Dieses Problem ist besonders schwerwiegend in Organisationen mit komplexen Projektabhängigkeiten, wo ein verzögerter Liefergegenstand eines Teams mehrere andere Teams und Projekte blockieren kann, was die geschäftliche Auswirkung der ursprünglichen Verzögerung vervielfacht.

## Indicators ⟡

- Verzögerungen in einem einzelnen Projekt betreffen mehrere andere Projekte oder Teams
- Projektzeitpläne in der gesamten Organisation werden häufig aufgrund von Abhängigkeitsverzögerungen angepasst
- Teams sind häufig blockiert und warten auf Liefergegenstände anderer Teams
- Release-Zeitpläne müssen über mehrere abhängige Projekte hinweg koordiniert werden
- Verzögerungen summieren sich und werden größer, während sie sich durch abhängige Arbeit fortpflanzen

## Symptoms ▲

- [Budgetüberschreitungen](budgetueberschreitungen.md)
<br/>  Sich fortpflanzende Verzögerungen erhöhen die Kosten, da Teams untätig bleiben oder Überstunden benötigen, um verlorene Zeit über mehrere Projekte hinweg aufzuholen.
- [Unzufriedenheit der Stakeholder](unzufriedenheit-der-stakeholder.md)
<br/>  Geschäftliche Stakeholder verlieren Vertrauen, da Verzögerungen in einem Bereich sichtbar mehrere abhängige Liefergegenstände beeinträchtigen.
- [Verpasste Termine](verpasste-termine.md)
<br/>  Verzögerungen, die sich durch Abhängigkeitsketten fortpflanzen, führen dazu, dass mehrere nachgelagerte Projekte ihre geplanten Liefertermine verpassen.
- [Ständig verschobene Termine](staendig-verschobene-termine.md)
<br/>  Während sich Verzögerungen fortpflanzen, müssen Projektzeitpläne wiederholt angepasst werden, was ein Umfeld instabiler Zeitpläne schafft.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Teams, die durch vorgelagerte Verzögerungen blockiert sind, erleben Frustration und sinkende Moral aufgrund der Unfähigkeit, Fortschritte zu machen.

## Causes ▼

- [Engpassbildung](engpassbildung.md)
<br/>  Engpässe in der Entwicklungspipeline verlangsamen Liefergegenstände, von denen mehrere nachgelagerte Teams abhängen.
- [Schlechte Planung](schlechte-planung.md)
<br/>  Unzureichende Planung berücksichtigt Projektabhängigkeiten nicht und lässt keinen Puffer, um Verzögerungen abzufangen.
- [Freigabe-Abhängigkeiten](freigabe-abhaengigkeiten.md)
<br/>  Verpflichtende Freigaben durch bestimmte Personen schaffen Verzögerungspunkte, die ganze Ketten abhängiger Arbeit blockieren.

## Detection Methods ○

- **Abhängigkeits-Auswirkungsanalyse:** Nachverfolgung, wie Verzögerungen in einem Projekt andere Projekte beeinflussen
- **Kritischer-Pfad-Analyse:** Identifikation von Projektabhängigkeitsketten und potenziellen Engpässen
- **Verzögerungsfortpflanzungs-Tracking:** Beobachtung, wie sich anfängliche Verzögerungen durch die Organisation verbreiten
- **Ressourcenauslastungsanalyse:** Messung von Leerlaufzeit, die durch Abhängigkeitsverzögerungen verursacht wird
- **Stakeholder-Auswirkungsbewertung:** Bewertung der geschäftlichen Auswirkung kaskadierender Projektverzögerungen

## Examples

Der Release einer mobilen App hängt von einer neuen API ab, die vom Backend-Team geliefert werden soll, welches wiederum von Datenbankschemaänderungen des Infrastruktur-Teams abhängt. Als das Infrastruktur-Team auf unerwartete Compliance-Anforderungen stößt, die seine Arbeit um 3 Wochen verzögern, muss das Backend-Team seine API-Arbeit verzögern, was das Mobile-Team zwingt, seinen Release zu verschieben. Eine Marketingkampagne, die an den App-Release gebunden ist, muss ebenfalls verzögert werden, und eine Ankündigung einer Geschäftspartnerschaft, die von der App-Funktionalität abhängt, wird ins nächste Quartal verschoben, wodurch eine dreiwöchige technische Verzögerung zu einer erheblichen geschäftlichen Auswirkung wird. Ein weiteres Beispiel betrifft eine E-Commerce-Plattform, bei der die verzögerte Zahlungsintegration des Checkout-Teams den neuen Fulfillment-Prozess des Bestandsteams blockiert, was wiederum die neuen Auftragsmanagement-Werkzeuge des Kundenservice-Teams blockiert, was letztlich einen wichtigen Produktlaunch verzögert, der über alle drei Bereiche hinweg koordiniert war.
