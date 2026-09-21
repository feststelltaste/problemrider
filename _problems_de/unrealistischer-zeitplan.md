---
title: Unrealistischer Zeitplan
description: Projektzeitpläne basieren auf optimistischen Annahmen statt realistischen
  Schätzungen, was zu Stress und Abkürzungen führt.
category:
- Management
- Process
related_problems:
- slug: unrealistic-deadlines
  similarity: 0.7
- slug: delayed-project-timelines
  similarity: 0.65
- slug: planning-dysfunction
  similarity: 0.6
- slug: deadline-pressure
  similarity: 0.6
- slug: time-pressure
  similarity: 0.6
- slug: missed-deadlines
  similarity: 0.6
solutions:
- iterative-development
- requirements-analysis
- short-iteration-cycles
- capacity-based-planning
- explicit-prioritization-framework
- regular-stakeholder-demonstrations
- story-mapping
- work-in-progress-limits
- definition-of-ready
layout: problem
lang: de
en_slug: unrealistic-schedule
---

## Description

Unrealistische Zeitpläne treten auf, wenn Projektzeitpläne basierend auf Wunschdenken, externem Druck oder unzureichendem Verständnis der tatsächlich erforderlichen Arbeit gesetzt werden, statt auf sorgfältiger Schätzung und Planung. Diese Zeitpläne unterschätzen typischerweise Komplexität, ignorieren Abhängigkeiten, versäumen es, Risiken zu berücksichtigen, und nehmen an, dass alles perfekt laufen wird. Das Ergebnis ist chronischer Zeitplandruck, der Teams zwingt, Abkürzungen zu nehmen, exzessive Stunden zu arbeiten und Qualität zu kompromittieren.

## Indicators ⟡

- Geschätzte Fertigstellungszeiten sind erheblich kürzer als historische Daten für ähnliche Arbeit
- Zeitpläne haben keine Pufferzeit für unerwartete Probleme oder Nacharbeit
- Der Zeitplan wird von externen Terminen statt realistischen Arbeitsschätzungen diktiert
- Teammitglieder äußern konsequent Bedenken, dass Termine unmöglich einzuhalten sind
- Ähnliche Projekte in der Vergangenheit haben ihre geplanten Zeitpläne konsequent überschritten

## Symptoms ▲

- [Termindruck](termindruck.md)
<br/>  Unrealistische Zeitpläne setzen Teams konstant unter Druck, schneller zu liefern, als machbar ist.
- [Verpasste Termine](verpasste-termine.md)
<br/>  Zeitpläne, die auf optimistischen Annahmen statt der Realität basieren, führen unvermeidlich zu verpassten Terminen.
- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Teams, die exzessive Stunden arbeiten, um unrealistische Zeitpläne einzuhalten, erleben Burnout und Erschöpfung.
- [Qualitätsverschlechterung](qualitaetsverschlechterung.md)
<br/>  Zeitplandruck zwingt Teams, Code-Reviews, Testen und ordentliches Design zu überspringen, was die Softwarequalität verschlechtert.
- [Qualitätskompromisse](qualitaetskompromisse.md)
<br/>  Unrealistische Zeitpläne zwingen Entwickler, Abkürzungen zu nehmen, die technische Schulden anhäufen.
- [Erhöhte Fehlerraten](erhoehte-fehlerraten.md)
<br/>  Das Hetzen durch die Entwicklung, um unrealistische Zeitpläne einzuhalten, führt dazu, dass mehr Bugs eingeführt werden.
- [Zeitdruck](zeitdruck.md)
<br/>  Geschäfts-Stakeholder erlegen externe Termine auf, die realistische technische Schätzungen überstimmen.

## Causes ▼

- [Schlechte Planung](schlechte-planung.md)
<br/>  Unzureichende Planung, die es versäumt, Komplexität, Abhängigkeiten und Risiken zu berücksichtigen, produziert unrealistische Zeitpläne.
- [Planungsdysfunktion](planungsdysfunktion.md)
<br/>  Dysfunktionale Planungsprozesse, die historische Daten und Team-Input ignorieren, produzieren unrealistische Zeitpläne.

## Detection Methods ○

- **Schätzgenauigkeitsanalyse:** Vergleich tatsächlicher Fertigstellungszeiten mit ursprünglichen Schätzungen
- **Verfolgung des Team-Stressniveaus:** Regelmäßige Befragungen zu Arbeitslast und Zeitplandruck
- **Velocity-Verfolgung:** Überwachung der Teamproduktivität über die Zeit zur Identifikation nicht nachhaltigen Tempos
- **Zeitplanabweichungsberichterstattung:** Verfolgung von Abweichungen von geplanten Zeitplänen über Projekte hinweg
- **Qualitätsmetrik-Korrelation:** Analyse der Beziehung zwischen Zeitplandruck und Fehlerraten

## Examples

Einem Mobile-App-Projekt wird ein Vier-Monats-Termin gegeben, weil das Geschäft vor einer wichtigen Branchenkonferenz launchen möchte. Das Entwicklungsteam schätzt jedoch, dass die Arbeit basierend auf den Feature-Anforderungen und ihrer vergangenen Erfahrung mit ähnlichen Anwendungen mindestens acht Monate dauern wird. Das Management besteht darauf, dass der Vier-Monats-Termin aufgrund von Wettbewerbsdruck nicht verhandelbar ist. Das Team wird gezwungen, Nächte und Wochenenden zu arbeiten, Code-Reviews zu überspringen, automatisiertes Testen zu eliminieren und Features mit minimaler Fehlerbehandlung zu implementieren. Die Anwendung launcht pünktlich, ist aber von Abstürzen, Performance-Problemen und Sicherheitslücken geplagt, die den Ruf des Unternehmens schädigen und sechs Monate zusätzliche Arbeit zur Behebung erfordern. Ein weiteres Beispiel betrifft ein Datenbankmigrationsprojekt, bei dem der Zeitplan nur zwei Wochen für Testen erlaubt, weil das Geschäft das neue System bis zum Ende des Geschäftsquartals betriebsbereit braucht. Die Komplexität der Migration von fünf Jahren Kundendaten wird stark unterschätzt, und Testen offenbart zahlreiche Datenintegritätsprobleme. Das Team wird gezwungen, zwischen der Verzögerung der Migration (Verpassen des Geschäftstermins) oder dem Fortfahren mit bekannten Datenproblemen zu wählen, was letztlich zu Kundenservice-Problemen und Notfall-Fixes führt, die mehr kosten als das ursprüngliche Projekt.
