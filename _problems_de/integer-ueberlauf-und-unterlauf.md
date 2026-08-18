---
title: Integer-Überlauf und -Unterlauf
description: Arithmetische Operationen produzieren Ergebnisse, die den maximalen
  oder minimalen von Integer-Datentypen darstellbaren Wert überschreiten, was zu
  unerwartetem Verhalten führt.
category:
- Code
- Database
- Security
related_problems:
- slug: buffer-overflow-vulnerabilities
  similarity: 0.65
- slug: stack-overflow-errors
  similarity: 0.55
- slug: null-pointer-dereferences
  similarity: 0.5
solutions:
- fuzz-testing
- input-validation
- static-analysis-and-linting
- value-range-definition
- design-by-contract
- plausibility-checks
- property-based-testing
- negative-testing
layout: problem
lang: de
en_slug: integer-overflow-underflow
---

## Description

Integer-Überlauf und -Unterlauf treten auf, wenn arithmetische Operationen Ergebnisse produzieren, die nicht innerhalb der Grenzen des genutzten Integer-Datentyps dargestellt werden können. Überlauf tritt auf, wenn ein Ergebnis den maximal darstellbaren Wert überschreitet, während Unterlauf auftritt, wenn ein Ergebnis kleiner als der minimal darstellbare Wert ist. In den meisten Programmiersprachen führen diese Bedingungen dazu, dass der Wert umschlägt, was zu unerwartetem und potenziell gefährlichem Verhalten einschließlich Sicherheitslücken führt.

## Indicators ⟡

- Berechnungen produzieren unerwartet kleine oder negative Ergebnisse aus großen positiven Eingaben
- Schleifenzähler oder Array-Indizes werden unerwartet negativ
- Sicherheitsprüfungen werden aufgrund unerwarteten Wertumschlags umgangen
- Speicherallokation schlägt fehl oder allokiert falsche Mengen aufgrund von Größenberechnungsfehlern
- Finanz- oder Messberechnungen produzieren offensichtlich falsche Ergebnisse

## Symptoms ▲

- [Pufferüberlauf-Schwachstellen](pufferueberlauf-schwachstellen.md)
<br/>  Integer-Überlauf in Größenberechnungen kann zu unterdimensionierten Pufferallokationen führen, die dann überlaufen.
- [Stille Datenkorruption](stille-datenkorruption.md)
<br/>  Wertumschlag durch Überlauf produziert falsche Daten, die sich unentdeckt durch das System fortpflanzen können.
- [Unvorhersehbares Systemverhalten](unvorhersehbares-systemverhalten.md)
<br/>  Umgeschlagene Werte verursachen unerwartetes Programmverhalten, das schwer zu reproduzieren und zu diagnostizieren ist.

## Causes ▼

- [Unzureichende Design-Fähigkeiten](unzureichende-design-faehigkeiten.md)
<br/>  Entwickler ohne Bewusstsein für Datentypgrenzen versäumen es, ordentliche Bereichsprüfungen zu implementieren.
- [Unzureichendes Testen](unzureichendes-testen.md)
<br/>  Unzureichendes Testen mit Grenzwerten versäumt es, Überlaufbedingungen vor Produktion zu erkennen.
- [Unzureichende Fehlerbehandlung](unzureichende-fehlerbehandlung.md)
<br/>  Fehlende Validierung und Fehlerbehandlung für arithmetische Operationen erlaubt es Überläufen, stillschweigend aufzutreten.

## Detection Methods ○

- **Statische Analysewerkzeuge:** Nutzung von Werkzeugen, die potenzielle Integer-Überlaufbedingungen in arithmetischen Operationen identifizieren können
- **Laufzeit-Überlauf-Erkennung:** Nutzung von Compiler-Flags oder Laufzeitbibliotheken, die Integer-Überlauf während der Ausführung erkennen
- **Bereichsprüfung:** Implementierung expliziter Bereichsprüfungen vor arithmetischen Operationen
- **Eingabevalidierungstests:** Testen mit extremen Eingabewerten zur Identifikation von Überlaufbedingungen
- **Code-Review:** Überprüfung arithmetischer Operationen auf potenzielle Überlaufszenarien
- **Fuzzing:** Nutzung automatisierten Testens mit großen oder ungewöhnlichen Eingabewerten, um Überlaufbedingungen auszulösen

## Examples

Eine Webanwendung berechnet Speicherpuffergrößen, indem sie die Anzahl der Elemente mit der Größe pro Element multipliziert. Ein Angreifer gibt eine extrem große Elementanzahl an, die einen Integer-Überlauf verursacht, sodass die berechnete Puffergröße auf einen kleinen positiven Wert umschlägt. Die Anwendung alloziert einen kleinen Puffer, schreibt dann aber Daten für die ursprüngliche große Anzahl von Elementen, was einen Pufferüberlauf verursacht, der für Code-Injektion ausgenutzt werden kann. Ein weiteres Beispiel betrifft eine Finanzanwendung, die Zinsen berechnet, indem sie Kapitalbeträge mit Zinssätzen multipliziert. Bei sehr großen Kapitalbeträgen läuft die Multiplikation über und schlägt zu einem negativen Wert um, was dazu führt, dass die Anwendung fälschlicherweise negative Zinsen berechnet und potenziell unangemessen Geld auf Konten gutschreibt.
