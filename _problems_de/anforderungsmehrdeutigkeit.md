---
title: Anforderungsmehrdeutigkeit
description: Systemanforderungen sind unklar, unvollständig oder für mehrere Interpretationen
  offen, was zu fehlausgerichteten Implementierungen und Nacharbeit führt.
category:
- Communication
- Process
- Requirements
related_problems:
- slug: inadequate-requirements-gathering
  similarity: 0.65
- slug: frequent-changes-to-requirements
  similarity: 0.65
- slug: unclear-goals-and-priorities
  similarity: 0.6
- slug: complex-implementation-paths
  similarity: 0.6
- slug: team-confusion
  similarity: 0.6
- slug: poorly-defined-responsibilities
  similarity: 0.6
solutions:
- evolutionary-requirements-development
- requirements-analysis
- stakeholder-feedback-loops
- behavior-driven-development-bdd
- business-process-modeling
- business-quality-scenarios
- business-test-cases
- compatibility-requirements
- decision-tables
- on-site-customer
- personas
- prototypes
- prototyping
- requirements-traceability-matrix
- security-requirements-definition
- specification-by-example
- story-mapping
- subject-matter-reviews
- ubiquitous-language
- user-acceptance-tests
- user-stories
- domain-experts
- domain-modeling
- domain-specific-languages
- event-storming
- functional-gap-analysis
- wireframing
- definition-of-ready
- regular-stakeholder-demonstrations
- domain-immersion
- exploratory-testing
layout: problem
lang: de
en_slug: requirements-ambiguity
---

## Description

Anforderungsmehrdeutigkeit tritt auf, wenn Systemanforderungen auf Weisen ausgedrückt werden, die mehrere Interpretationen erlauben, ausreichende Details für die Implementierung vermissen lassen oder es versäumen, kritische Randfälle und Einschränkungen zu behandeln. Diese Mehrdeutigkeit zwingt Entwickler, Annahmen über beabsichtigte Funktionalität zu treffen, was oft zu Implementierungen führt, die nicht den Stakeholder-Erwartungen entsprechen. Das Problem wird verstärkt, wenn mehrdeutige Anforderungen nicht früh im Entwicklungsprozess geklärt werden, was zu kostspieliger Nacharbeit führt, wenn die Fehlausrichtung entdeckt wird.

## Indicators ⟡

- Entwickler fragen während der Implementierung häufig nach Klärung zu Anforderungen
- Verschiedene Teammitglieder interpretieren dieselbe Anforderung widersprüchlich
- Anforderungen nutzen vage Sprache wie „nutzerfreundlich" oder „schnell" ohne spezifische Kriterien
- Randfälle und Fehlerbedingungen werden in Anforderungen nicht behandelt
- Stakeholder äußern Überraschung oder Unzufriedenheit über implementierte Funktionalität, die technisch schriftliche Anforderungen erfüllt

## Symptoms ▲

- [Annahmenbasierte Entwicklung](annahmenbasierte-entwicklung.md)
<br/>  Wenn Anforderungen unklar sind, werden Entwickler gezwungen, die Lücken mit ihren eigenen Annahmen über beabsichtigtes Verhalten zu füllen.
- [Implementierungs-Nacharbeit](implementierungs-nacharbeit.md)
<br/>  Mehrdeutige Anforderungen führen zu Implementierungen, die nicht den Stakeholder-Erwartungen entsprechen, was kostspielige Neuaufbauten erfordert.
- [Fehlausgerichtete Liefergegenstände](fehlausgerichtete-liefergegenstaende.md)
<br/>  Vage Anforderungen erlauben unterschiedliche Interpretationen, was in gelieferten Features resultiert, die nicht dem entsprechen, was Stakeholder tatsächlich benötigten.
- [Scope Creep](scope-creep.md)
<br/>  Mehrdeutige Anforderungen lassen Raum für sich erweiternde Interpretation dessen, was gebaut werden muss, was es dem Umfang erlaubt, ungebremst zu wachsen.
- [Teamverwirrung](teamverwirrung.md)
<br/>  Mehrere valide Interpretationen derselben Anforderung verursachen, dass Teammitglieder aneinander vorbeiarbeiten.
- [Verschwendeter Entwicklungsaufwand](verschwendeter-entwicklungsaufwand.md)
<br/>  Entwicklungsarbeit, die auf missverstandenen mehrdeutigen Anforderungen basiert, wird zu Wegwerfaufwand, wenn die Fehlausrichtung entdeckt wird.

## Causes ▼

- [Unzureichende Anforderungserhebung](unzureichende-anforderungserhebung.md)
<br/>  Unzureichende Analyse und Stakeholder-Einbindung während der Anforderungserhebung versäumt es, klare, vollständige Spezifikationen zu erfassen.
- [Kommunikationslücke zwischen Stakeholdern und Entwicklern](kommunikationsluecke-zwischen-stakeholdern-und-entwicklern.md)
<br/>  Schlechte Kommunikation zwischen Stakeholdern und Entwicklern bedeutet, dass Anforderungen während der Entwicklung nicht geklärt oder verfeinert werden.
- [Unklare Ziele und Prioritäten](unklare-ziele-und-prioritaeten.md)
<br/>  Wenn organisatorische Ziele unklar sind, können Anforderungen nicht präzise geschrieben werden, weil die gewünschten Ergebnisse selbst mehrdeutig sind.

## Detection Methods ○

- **Nachverfolgung von Klärungsanfragen:** Überwachung, wie oft Entwickler nach Anforderungsklärungen fragen
- **Implementierungsvarianzanalyse:** Vergleich gelieferter Funktionalität mit ursprünglichen Anforderungen
- **Bewertung der Stakeholder-Zufriedenheit:** Bewertung, ob Liefergegenstände Stakeholder-Erwartungen erfüllen
- **Effektivität des Anforderungs-Reviews:** Bewertung der Qualität von Anforderungs-Review-Prozessen
- **Nacharbeitsmetriken:** Nachverfolgung, wie viel Entwicklungsarbeit aufgrund von Anforderungsproblemen neu gemacht wird
- **User-Acceptance-Testing-Ergebnisse:** Analyse, ob Implementierungen Nutzer-Abnahmekriterien bestehen

## Examples

Eine Anforderung besagt „Das System sollte schnelle Suchfunktionalität bieten", spezifiziert aber nicht, was „schnell" bedeutet oder unter welchen Bedingungen. Ein Entwickler implementiert eine Suche, die Ergebnisse in 100ms für einfache Abfragen zurückgibt, während ein anderer annimmt, „schnell" bedeute umfassende Suche einschließlich Volltextindizierung, die 2 Sekunden dauert, aber relevantere Ergebnisse findet. Wenn Stakeholder das System testen, entdecken sie, dass ihre Definition von „schnell" Antwortzeiten unter einer Sekunde für jede Abfrage war, was erhebliche Nacharbeit der Suchimplementierung erfordert. Ein weiteres Beispiel betrifft eine Anforderung für „nutzerfreundliche Dateneingabeformulare" ohne zu spezifizieren, was Formulare nutzerfreundlich macht. Das Entwicklungsteam erstellt Formulare, die technisch funktional sind, aber nicht die Tastaturkürzel, Validierungsmuster und Workflow-Abkürzungen unterstützen, die Nutzer basierend auf ihren aktuellen Werkzeugen erwarten, was zur Ablehnung des neuen Systems durch Nutzer führt, obwohl die schriftlichen Anforderungen erfüllt sind.
