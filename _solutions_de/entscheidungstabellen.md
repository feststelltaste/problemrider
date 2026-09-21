---
title: Entscheidungstabellen
description: Definition und Auswertung komplexer Geschäftsregeln in tabellarischer
  Form.
category:
- Code
- Requirements
problems:
- complex-and-obscure-logic
- legacy-business-logic-extraction-difficulty
- difficult-code-comprehension
- poor-domain-model
- requirements-ambiguity
- spaghetti-code
layout: solution
lang: de
en_slug: decision-tables
related_solutions:
- slug: rule-based-systems
  similarity: 0.8
- slug: domain-specific-languages
  similarity: 0.7
- slug: business-process-automation
  similarity: 0.65
- slug: architecture-decision-records
  similarity: 0.65
- slug: business-test-cases
  similarity: 0.65
- slug: code-metrics
  similarity: 0.65
---

## Description

Entscheidungstabellen extrahieren komplexe bedingte Geschäftslogik — tief verschachtelte If-else-Ketten oder große Switch-Anweisungen — aus dem Code in eine tabellarische Repräsentation, die Kombinationen von Eingabebedingungen direkt auf ihre entsprechenden Ausgaben oder Aktionen abbildet, eine Zeile pro Regel. Die Logik so zu strukturieren macht sowohl die einzelnen Regeln als auch die Vollständigkeit des Regelsatzes in einer Form sichtbar, die nicht nur Entwickler, sondern auch Geschäfts-Stakeholder überprüfen können, da Lücken (Bedingungskombinationen, die keine Zeile behandelt) und Widersprüche als fehlende oder doppelte Zeilen erkennbar werden, statt im Kontrollfluss verschachtelter Bedingungen begraben zu sein. Dies ist für Legacy-Systeme direkt relevant, weil Geschäftsregeln, die Preisgestaltung, Berechtigung oder Routing betreffen, häufig über Jahre zu bedingter Logik anwachsen, die so tief verschachtelt ist, dass niemand, einschließlich ihrer ursprünglichen Autoren, sicher sagen kann, was jede Kombination von Eingaben tatsächlich erzeugt. Einmal in eine Tabelle extrahiert und gegen die Erwartungen von Geschäfts-Stakeholdern validiert, wird dieselbe Tabelle zur natürlichen Quelle für Testfälle, da jede Zeile ein konkretes Eingabe-Ausgabe-Paar ist, das die Implementierung erfüllen muss, und die Logik mit einer leichtgewichtigen Rules Engine oder tabellengetriebenem Code neu zu implementieren schrumpft, was zuvor große Mengen bedingter Logik waren, auf einen Bruchteil ihrer ursprünglichen Größe. Der Tradeoff ist, dass sich nicht alle Geschäftslogik sauber in eine flache Tabelle von Bedingungen und Ergebnissen zerlegen lässt, und die Regeln überhaupt erst zu extrahieren erfordert dieselbe sorgfältige Domänenanalyse, die die ursprüngliche Logik schwer zu entwirren machte.

## How to Apply ◆

- Identifizieren Sie komplexe bedingte Logik in der Legacy-Codebasis (tief verschachtelte If-else-Ketten, Switch-Anweisungen mit vielen Fällen), die Geschäftsregeln repräsentiert.
- Extrahieren Sie diese Geschäftsregeln in Entscheidungstabellen, die Eingabebedingungen auf erwartete Ausgaben oder Aktionen abbilden.
- Lassen Sie Geschäfts-Stakeholder die Entscheidungstabellen validieren, um zu bestätigen, dass sie die beabsichtigte Geschäftslogik genau repräsentieren.
- Implementieren Sie Entscheidungstabellen mittels einer Rules Engine (Drools, Easy Rules) oder einfachem tabellengetriebenem Code, der die komplexe bedingte Logik ersetzt.
- Schreiben Sie Testfälle, abgeleitet aus den Entscheidungstabellenzeilen, um zu verifizieren, dass die Implementierung der Spezifikation entspricht.
- Pflegen Sie Entscheidungstabellen als lebende Dokumentation, die aktualisiert wird, wenn sich Geschäftsregeln ändern.

## Tradeoffs ⇄

**Vorteile:**
- Macht komplexe Geschäftslogik sowohl für technische als auch für Geschäfts-Stakeholder sichtbar und verständlich.
- Vereinfacht die Pflege, indem Geschäftsregeln vom Anwendungscode getrennt werden.
- Ermöglicht Vollständigkeitsanalyse: Entscheidungstabellen machen es leicht, fehlende Regelkombinationen zu erkennen.
- Erleichtert Testing, indem eine natürliche Quelle für Testfallgenerierung bereitgestellt wird.

**Kosten:**
- Nicht alle Geschäftslogik lässt sich sauber auf ein tabellarisches Format abbilden; manche Regeln beinhalten komplexe Abhängigkeiten.
- Die Extraktion von Regeln aus tief eingebettetem Legacy-Code erfordert sorgfältige Analyse und Domänenwissen.
- Entscheidungstabellen können unhandlich werden, wenn die Anzahl der Bedingungen und Kombinationen sehr groß ist.
- Die Einführung einer Rules Engine fügt eine Abhängigkeit und Lernkurve hinzu.

## How It Could Be

Ein Legacy-Versicherungspreissystem enthält über 2.000 Zeilen verschachtelter bedingter Logik, die Prämien basierend auf Alter, Standort, Deckungsart, Schadenshistorie und Vertragslaufzeit berechnet. Niemand versteht vollständig alle Interaktionen zwischen den Bedingungen. Das Team extrahiert die Preislogik in Entscheidungstabellen, mit einer Tabelle pro Deckungsart. Jede Zeile spezifiziert eine Kombination von Eingabebedingungen und den resultierenden Prämienmodifikator. Geschäftsanalysten überprüfen die Tabellen und entdecken drei Bedingungskombinationen, die falsche Preisgestaltung erzeugen (die Fehler waren seit Jahren in Produktion). Die Entscheidungstabellen werden dann mittels einer leichtgewichtigen Rules Engine implementiert, was den Preiscode von 2.000 Zeilen auf 200 Zeilen plus die externalisierten Tabellen reduziert und künftige Preisänderungen zu einer Frage des Bearbeitens einer Tabelle statt des Modifizierens von Code macht.
