---
title: Schwierigkeit bei der Extraktion von Legacy-Geschäftslogik
description: Kritische Geschäftsregeln sind tief in Legacy-Codestrukturen eingebettet,
  was sie nahezu unmöglich macht zu identifizieren und zu extrahieren.
category:
- Architecture
- Code
- Communication
related_problems:
- slug: modernization-roi-justification-failure
  similarity: 0.65
- slug: poor-domain-model
  similarity: 0.65
- slug: legacy-system-documentation-archaeology
  similarity: 0.65
- slug: integration-difficulties
  similarity: 0.65
- slug: low-code-customization-sprawl
  similarity: 0.6
- slug: legacy-skill-shortage
  similarity: 0.6
solutions:
- strangler-fig-pattern
- bubble-context
- business-event-processing
- business-process-automation
- business-process-modeling
- data-format-conversion
- data-modeling
- decision-tables
- hexagonal-architecture
- rule-based-systems
- domain-driven-design
- domain-experts
- domain-modeling
- domain-patterns
- domain-specific-languages
- event-storming
- characterization-tests
- parallel-run
- domain-immersion
layout: problem
lang: de
en_slug: legacy-business-logic-extraction-difficulty
---

## Description

Schwierigkeit bei der Extraktion von Legacy-Geschäftslogik tritt auf, wenn kritische Geschäftsregeln und -prozesse so tief im Code von Legacy-Systemen eingebettet sind, dass sie nahezu unmöglich zu identifizieren, zu verstehen und für Modernisierungsbemühungen zu extrahieren sind. Anders als einfach schlecht dokumentierter Code beinhaltet dieses Problem Geschäftslogik, die mit technischen Implementierungsdetails vermischt ist, über mehrere Module verstreut ist, durch implizite Verhaltensweisen ausgedrückt wird oder in Datenstrukturen und Stored Procedures eingebettet ist. Dies macht Modernisierung extrem riskant, weil Teams essenzielle Geschäftsverhaltensweisen nicht zuversichtlich in neuen Systemen reproduzieren können.

## Indicators ⟡

- Geschäftsregeln, die von aktuellen Geschäfts-Stakeholdern oder Dokumentation nicht erklärt werden können
- Code, in dem Geschäftslogik mit Datenbankzugriff, UI-Rendering und Systemwerkzeugen vermischt ist
- Kritische Geschäftsverhaltensweisen, die sich nur unter bestimmten Datenbedingungen oder Randfällen zeigen
- Fachexperten, die Geschäftsprozesse anders beschreiben, als sich das System tatsächlich verhält
- Datenbank-Stored-Procedures oder Trigger, die komplexe Geschäftslogik ohne Dokumentation enthalten
- Geschäftsregeln, die durch Datenwerte, Konfigurationstabellen oder dateibasierte Einstellungen implementiert sind
- Systemverhalten, das aufgrund fehlenden Geschäftskontexts in Testumgebungen nicht reproduziert werden kann

## Symptoms ▲

- [Scheiternde ROI-Rechtfertigung für Modernisierung](scheiternde-roi-rechtfertigung-fuer-modernisierung.md)
<br/>  Die Unfähigkeit, Geschäftslogik zu extrahieren und zu verstehen, macht es unmöglich, Modernisierungskosten und -nutzen akkurat zu schätzen.
- [Lähmung der Modernisierungsstrategie](laehmung-der-modernisierungsstrategie.md)
<br/>  Teams können keinen Modernisierungsansatz wählen, wenn sie nicht verstehen, welche Geschäftslogik bewahrt werden muss.
- [Große Schätzungen für kleine Änderungen](grosse-schaetzungen-fuer-kleine-aenderungen.md)
<br/>  Wenn Geschäftslogik tief eingebettet ist, erfordern selbst kleine Modifikationen umfangreiche Analyse, um die volle Auswirkung zu verstehen.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Entwickler werden zurückhaltend, Code zu ändern, wenn sie nicht bestimmen können, welche Änderungen unbekannte Geschäftsregeln brechen könnten.
- [Erhöhte Entwicklungskosten](erhoehte-entwicklungskosten.md)
<br/>  Jede Änderung erfordert umfangreiche Analyse, um eingebettete Geschäftsregeln zu verstehen, was die Entwicklungskosten erheblich erhöht.

## Causes ▼

- [Spaghetticode](spaghetticode.md)
<br/>  Verworrener, unstrukturierter Code macht es nahezu unmöglich zu identifizieren, wo Geschäftslogik beginnt und technische Implementierung endet.
- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Fehlende Dokumentation über Geschäftsregeln zwingt Teams, Logik aus dem Code zurückzuentwickeln, statt Spezifikationen zu referenzieren.
- [Implizites Wissen](implizites-wissen.md)
<br/>  Geschäftsregeln existieren als unausgesprochene Annahmen, die nur ausgeschiedenen Mitarbeitern bekannt sind, was die Extraktion von Code-Archäologie abhängig macht.
- [Hohe Kopplung und geringe Kohäsion](hohe-kopplung-und-geringe-kohaesion.md)
<br/>  Geschäftslogik, die mit Datenbankzugriff, UI und Werkzeugen über viele Module hinweg vermischt ist, macht es unmöglich, sie zu isolieren.

## Detection Methods ○

- Durchführung von Geschäftsregel-Archäologie-Sitzungen mit Fachexperten und Legacy-Code-Review
- Nutzung statischer Analysewerkzeuge zur Identifikation von Geschäftslogikmustern, die über die Codebasis verstreut sind
- Durchführung von Datenflussanalyse, um nachzuverfolgen, wie Geschäftsregeln über Systemkomponenten hinweg implementiert sind
- Interview langjähriger Mitarbeiter und Kunden zu Geschäftsverhaltensweisen, auf die sie sich verlassen
- Analyse von Produktionslogs und Fehlermustern zur Identifikation impliziter Geschäftsregeldurchsetzung
- Vergleich von Geschäftsprozessdokumentation mit tatsächlichem Systemverhalten durch Testen
- Nutzung von Code-Komplexitätsmetriken zur Identifikation von Bereichen, in denen Geschäfts- und technische Logik vermischt sind
- Durchführung von Geschäftsauswirkungsanalyse zur Identifikation kritischer Verhaltensweisen, die bewahrt werden müssen

## Examples

Ein Versicherungsunternehmen versucht, sein 20 Jahre altes Schadensverarbeitungssystem zu modernisieren, und entdeckt, dass die Prämienberechnungslogik über 47 unterschiedliche COBOL-Programme, 15 Datenbank-Stored-Procedures und Dutzende Konfigurationstabellen verstreut ist. Die Geschäftsregeln zur Bestimmung der Schadensberechtigung sind teilweise in der Anwendung codiert, teilweise durch Datenbankeinschränkungen durchgesetzt und teilweise durch manuelle Prozesse gehandhabt, die sich über die Zeit entwickelt haben. Als Geschäftsanalysten versuchen, die aktuellen Regeln zu dokumentieren, stellen sie fest, dass das System Tausende von Randfällen handhabt, die niemand aktuell versteht – wie es Schäden für eingestellte Policentypen verarbeitet oder bundesstaatspezifische Vorschriften handhabt, die sich mehrfach geändert haben. Das Team entdeckt, dass manche Geschäftslogik nur ausgeführt wird, wenn bestimmte Kombinationen von Kundendaten, Policenhistorie und Schadensarten auftreten, was es nahezu unmöglich macht, umfassend zu testen. Nach 18 Monaten Analyse können sie immer noch nicht zuversichtlich angeben, was das vollständige Geschäftsregelwerk ist, was sie zwingt, die Modernisierungsbemühung wegen inakzeptablen Risikos aufzugeben, kritische Geschäftsverhaltensweisen zu übersehen.
