---
title: Spaghetticode
description: Code mit verworrener, unstrukturierter Logik, der nahezu unmöglich zu
  verstehen, zu debuggen oder sicher zu modifizieren ist.
category:
- Architecture
- Code
related_problems:
- slug: difficult-to-understand-code
  similarity: 0.65
- slug: complex-and-obscure-logic
  similarity: 0.65
- slug: clever-code
  similarity: 0.6
- slug: difficult-to-test-code
  similarity: 0.6
- slug: inconsistent-codebase
  similarity: 0.6
- slug: copy-paste-programming
  similarity: 0.6
solutions:
- incremental-refactoring
- modularization-and-bounded-contexts
- aspect-oriented-programming-aop
- bounded-contexts
- bubble-context
- decision-tables
- facades
- high-cohesion
- layered-architecture
- mediator
- rule-based-systems
layout: problem
lang: de
en_slug: spaghetti-code
---

## Description

Spaghetticode bezeichnet Quellcode, der aufgrund schlechter Organisation, exzessiver Nutzung von Kontrollstrukturen wie Goto-Anweisungen, tief verschachtelter Bedingungen und fehlender klarer Trennung zwischen verschiedenen Belangen verworren, unstrukturiert und schwer nachvollziehbar geworden ist. Der Codefluss springt unvorhersehbar herum, was es extrem schwierig macht, die Programmlogik zu verstehen, Ausführungspfade nachzuverfolgen oder Änderungen vorzunehmen, ohne Bugs einzuführen.

## Indicators ⟡

- Der Ausführungsfluss des Codes ist schwer nachzuvollziehen und springt unvorhersehbar herum
- Funktionen oder Methoden sind extrem lang mit tief verschachtelten Kontrollstrukturen
- Globale Variablen werden extensiv für Kommunikation zwischen verschiedenen Teilen genutzt
- Der Code enthält viele willkürliche Sprünge, Breaks oder Continues, die den logischen Fluss unterbrechen
- Mehrere Austrittspunkte aus Funktionen machen es schwer, Rückgabebedingungen zu verstehen

## Symptoms ▲

- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  Verworrener, unstrukturierter Code mit unvorhersehbarem Kontrollfluss ist für Entwickler extrem schwierig zu lesen und zu verstehen.
- [Brüchige Codebasis](bruechige-codebasis.md)
<br/>  Die verworrenen Abhängigkeiten von Spaghetticode bedeuten, dass Änderungen in einem Bereich häufig nicht verwandte Funktionalität brechen.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Das Hinzufügen von Features zu Spaghetticode erfordert umfangreiche Zeit, um die verworrene Logik zu verstehen und Änderungen sicher zu integrieren.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Über einzelne Features hinaus verlangsamt verworrener Kontrollfluss auch den gesamten Durchsatz des Teams bei Bugfixes und Wartung.
- [Hohe Rate an neu eingeführten Fehlern](hohe-rate-an-neu-eingefuehrten-fehlern.md)
<br/>  Der unvorhersehbare Kontrollfluss und die versteckten Abhängigkeiten in Spaghetticode machen ihn zu einer konstanten Bug-Quelle.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Entwickler werden zögerlich, Spaghetticode zu modifizieren, weil Änderungen unvorhersehbare und weitreichende Konsequenzen haben.

## Causes ▼

- [Inkonsistente Coding-Standards](inkonsistente-coding-standards.md)
<br/>  Ohne durchgesetzte Coding-Standards schreiben Entwickler unstrukturierten Code, der sich zu Spaghetti anhäuft.
- [Unzureichendes Code-Review](unzureichendes-code-review.md)
<br/>  Ohne Code-Review bleibt schlecht strukturierter Code unkontrolliert und häuft sich über die Zeit an.
- [Zeitdruck](zeitdruck.md)
<br/>  Unter dem Druck, schnell zu liefern, nehmen Entwickler Abkürzungen, die in verworrenem, schlecht strukturiertem Code resultieren.
- [Feature-Creep ohne Refactoring](feature-creep-ohne-refactoring.md)
<br/>  Ohne regelmäßiges Refactoring verschlechtert sich die Codestruktur über die Zeit, während Schnelllösungen und Patches verworrene Logik schaffen.

## Detection Methods ○

- **Zyklomatische-Komplexitäts-Analyse:** Nutzung von Werkzeugen zur Messung der Codekomplexität und Identifikation verworrener Methoden
- **Kontrollfluss-Visualisierung:** Erstellung von Diagrammen, die Code-Ausführungspfade zeigen, zur Identifikation von Spaghetti-Mustern
- **Code-Metrik-Bewertung:** Verfolgung von Funktionslänge, Verschachtelungstiefe und Anzahl der Austrittspunkte
- **Entwickler-Feedback:** Befragung von Teammitgliedern zu Codebereichen, die schwer zu verstehen sind
- **Bug-Dichte-Analyse:** Identifikation von Codebereichen mit hohen Bug-Raten, die auf Spaghetti-Struktur hindeuten könnten

## Examples

Ein Legacy-E-Commerce-System hat einen Checkout-Prozess, der als einzelne 2000-zeilige Funktion mit 15 Verschachtelungsebenen von If-Anweisungen, mehreren Goto-Anweisungen, die zu verschiedenen Teilen der Funktion springen, und globalen Variablen implementiert ist, die Zustandsänderungen über den gesamten Prozess hinweg verfolgen. Die Funktion handhabt Zahlungsverarbeitung, Bestandsaktualisierungen, Versandberechnungen, Steuerberechnung und E-Mail-Benachrichtigungen alle in einem verworrenen Durcheinander. Wenn ein Bug in der Steuerberechnung gemeldet wird, verbringen Entwickler Tage damit, den Code nachzuverfolgen, um zu verstehen, welcher Pfad zum Problem führt, und die Behebung riskiert, Zahlungsverarbeitung oder Bestandsverwaltung zu brechen. Ein weiteres Beispiel betrifft ein Berichtssystem, bei dem Datenverarbeitungslogik über mehrere Funktionen verstreut ist, die sich gegenseitig auf unvorhersehbare Weise aufrufen und globale Variablen nutzen, um Daten zwischen verschiedenen Verarbeitungsstufen zu übergeben. Eine einfache Änderung zum Hinzufügen eines neuen Datenfelds erfordert das Verstehen und Modifizieren von sieben verschiedenen Funktionen, jede mit ihrem eigenen komplexen Kontrollfluss und Nebeneffekten.
