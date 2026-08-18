---
title: Globaler Zustand und Nebeneffekte
description: Übermäßige Nutzung globaler Variablen oder Funktionen mit versteckten
  Nebeneffekten erschwert es, das Verhalten des Codes nachzuvollziehen.
category:
- Architecture
- Code
related_problems:
- slug: hidden-side-effects
  similarity: 0.8
- slug: unpredictable-system-behavior
  similarity: 0.7
- slug: hidden-dependencies
  similarity: 0.65
- slug: difficult-to-understand-code
  similarity: 0.6
- slug: high-coupling-low-cohesion
  similarity: 0.6
- slug: inconsistent-behavior
  similarity: 0.6
solutions:
- incremental-refactoring
- dependency-injection
- dependency-injection-container
- dependency-breaking-techniques
- solid-principles
- separation-of-concerns
- characterization-tests
- preparatory-refactoring
- code-reading-sessions
layout: problem
lang: de
en_slug: global-state-and-side-effects
---

## Description
Globaler Zustand und Nebeneffekte sind eine verbreitete Quelle von Komplexität und Fehlern in Softwaresystemen. Globaler Zustand bezeichnet Daten, die von überall in der Codebasis aus zugänglich und veränderbar sind, während Nebeneffekte Zustandsänderungen sind, die als Nebenprodukt eines Funktionsaufrufs auftreten. Bei übermäßiger Nutzung können diese Konstrukte es sehr schwer machen, das Verhalten des Systems nachzuvollziehen, da die Auswirkung einer Änderung weitreichend und unvorhersehbar sein kann.

## Indicators ⟡
- Es ist schwer zu verstehen, welche Auswirkung eine Änderung an einem Codeabschnitt hat.
- Derselbe Fehler tritt in verschiedenen Teilen des Systems auf.
- Das System verhält sich in unterschiedlichen Umgebungen unterschiedlich, obwohl der Code identisch ist.

## Symptoms ▲

- [Unvorhersehbares Systemverhalten](unvorhersehbares-systemverhalten.md)
<br/>  Änderungen am globalen Zustand von beliebiger Stelle der Codebasis aus verursachen unerwartete Nebeneffekte in scheinbar unzusammenhängenden Bereichen.
- [Versteckte Abhängigkeiten](versteckte-abhaengigkeiten.md)
<br/>  Komponenten, die sich globalen Zustand teilen, werden implizit voneinander abhängig, ohne dass dies in der Codestruktur sichtbar ist.
- [Schwer testbarer Code](schwer-testbarer-code.md)
<br/>  Funktionen, die auf globalem Zustand beruhen, können nicht isoliert getestet werden, weil ihr Verhalten von extern veränderbarem Zustand abhängt.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Änderungen am globalen Zustand in einem Bereich brechen bestehende Funktionalität an anderer Stelle, weil die Abhängigkeiten nicht ersichtlich sind.
- [Race Conditions](race-conditions.md)
<br/>  Veränderbarer globaler Zustand, auf den mehrere Threads ohne Synchronisation zugreifen, führt zu Data Races und Datenverfälschung.
- [Debugging-Schwierigkeiten](debugging-schwierigkeiten.md)
<br/>  Die Rückverfolgung von Fehlern ist extrem schwierig, wenn jeder Teil der Codebasis gemeinsam genutzten globalen Zustand unvorhersehbar verändern kann.

## Causes ▼

- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler ohne Erfahrung mit ordentlichen Softwaredesign-Mustern greifen standardmäßig auf globale Variablen als einfachsten Ansatz zurück.
- [Prozedurale Programmierung in OOP-Sprachen](prozedurale-programmierung-in-oop-sprachen.md)
<br/>  Eine prozedurale Denkweise führt dazu, dass Entwickler globale Variablen und Funktionen mit Nebeneffekten nutzen statt gekapselter Objekte.
- [Schlechte Kapselung](schlechte-kapselung.md)
<br/>  Das Versäumnis, Zustand ordentlich innerhalb von Objekten zu kapseln, macht ihn global zugänglich und lädt zu weitreichenden Veränderungen und Nebeneffekten ein.
- [Spaghetticode](spaghetticode.md)
<br/>  Verworrener, unstrukturierter Code neigt natürlich zu globalem Zustand als Weg, Daten zwischen schlecht organisierten Komponenten zu teilen.

## Detection Methods ○
- **Statische Analyse:** Nutzung statischer Analysewerkzeuge zur Identifikation der Nutzung globaler Variablen und Funktionen mit Nebeneffekten.
- **Code-Reviews:** Genaue Beachtung der Nutzung von globalem Zustand und Nebeneffekten während Code-Reviews.
- **Testing:** Schreiben von Tests, die die versteckten Abhängigkeiten und Nebeneffekte im Code offenlegen.

## Examples
Eine Funktion, die den Gesamtpreis eines Warenkorbs berechnet, hat auch den Nebeneffekt, einen Rabatt auf das Konto des Nutzers anzuwenden. Dieser Nebeneffekt ist nicht dokumentiert und weder aus dem Namen noch aus der Signatur der Funktion ersichtlich. Infolgedessen wendet ein Entwickler, der diese Funktion aufruft, um lediglich den Gesamtpreis in der UI anzuzeigen, unbeabsichtigt einen Rabatt auf das Konto des Nutzers an, was zu einem Umsatzverlust für das Unternehmen führt. Dies ist ein klassisches Beispiel dafür, wie versteckte Nebeneffekte zu unerwartetem und unerwünschtem Verhalten führen können.
