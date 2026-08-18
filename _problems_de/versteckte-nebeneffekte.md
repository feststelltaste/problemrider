---
title: Versteckte Nebeneffekte
description: Funktionen haben undokumentierte Nebeneffekte, die Zustand ändern oder
  Aktionen auslösen, die über ihren offensichtlichen Zweck hinausgehen.
category:
- Architecture
- Code
related_problems:
- slug: global-state-and-side-effects
  similarity: 0.8
- slug: hidden-dependencies
  similarity: 0.7
- slug: unpredictable-system-behavior
  similarity: 0.7
- slug: ripple-effect-of-changes
  similarity: 0.6
- slug: difficult-to-understand-code
  similarity: 0.55
- slug: monolithic-functions-and-classes
  similarity: 0.55
solutions:
- clean-code
- design-by-contract
- separation-of-concerns
- solid-principles
- change-impact-analysis
- parallel-run
- exploratory-testing
- code-reading-sessions
- characterization-tests
- dependency-breaking-techniques
layout: problem
lang: de
en_slug: hidden-side-effects
---

## Description

Versteckte Nebeneffekte treten auf, wenn Funktionen oder Methoden Aktionen ausführen, die über ihren offensichtlichen primären Zweck hinausgehen, ohne diese zusätzlichen Verhaltensweisen klar zu dokumentieren oder anzuzeigen. Diese Nebeneffekte könnten das Ändern von globalem Zustand, das Auslösen von Ereignissen, das Schreiben in Logs, das Senden von Benachrichtigungen oder das Aktualisieren von Caches umfassen. Versteckte Nebeneffekte machen Code schwer verständlich, testbar und wartbar, weil Entwickler nicht alle Konsequenzen eines Funktionsaufrufs allein anhand seines Namens und seiner Parameter vorhersagen können.

## Indicators ⟡
- Funktionen mit unschuldig klingenden Namen, die mehrere unzusammenhängende Aktionen ausführen
- Debugging zeigt, dass Funktionen Zustand ändern oder Aktionen auslösen, die aus ihren Signaturen nicht ersichtlich sind
- Unit-Tests sind schwer zu schreiben, weil Funktionen viele externe Abhängigkeiten haben
- Code-Reviews beinhalten häufig Fragen zu unerwartetem Funktionsverhalten
- Das Systemverhalten ändert sich unerwartet, wenn Funktionen in unterschiedlichen Kontexten aufgerufen werden

## Symptoms ▲

- [Unvorhersehbares Systemverhalten](unvorhersehbares-systemverhalten.md)
<br/>  Das Aufrufen von Funktionen erzeugt unerwartete Ergebnisse, weil ihre undokumentierten Nebeneffekte den Systemzustand auf nicht offensichtliche Weise ändern.
- [Schwer testbarer Code](schwer-testbarer-code.md)
<br/>  Funktionen mit versteckten Nebeneffekten erfordern umfangreiches Mocking von Datenbanken, Diensten und Caches, um selbst einfache Berechnungen zu testen.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Das Refactoring oder Wiederverwenden von Funktionen mit versteckten Nebeneffekten bricht unbeabsichtigt Funktionalität, die von diesen Nebeneffekten abhing.
- [Versteckte Abhängigkeiten](versteckte-abhaengigkeiten.md)
<br/>  Nebeneffekte schaffen implizite Abhängigkeiten zwischen der Funktion und externen Systemen, die aus der Schnittstelle nicht sichtbar sind.
- [Erhöhtes Risiko für Fehler](erhoehtes-risiko-fuer-fehler.md)
<br/>  Entwickler, die sich versteckter Nebeneffekte nicht bewusst sind, nehmen Änderungen vor, die unbeabsichtigt unerwünschte Aktionen wie E-Mails oder Datenbankschreibvorgänge auslösen.

## Causes ▼

- [Globaler Zustand und Nebeneffekte](globaler-zustand-und-nebeneffekte.md)
<br/>  Eine Codebasis-Kultur, die globalen Zustand nutzt, führt natürlich dazu, dass Funktionen im Laufe der Zeit versteckte Nebeneffekte anhäufen.
- [Schlecht definierte Verantwortlichkeiten](schlecht-definierte-verantwortlichkeiten.md)
<br/>  Wenn Funktionen keine klaren Einzelverantwortlichkeiten haben, werden zusätzliche Verhaltensweisen schrittweise hinzugefügt, ohne Teil des ursprünglichen Vertrags zu sein.
- [Feature-Creep ohne Refactoring](feature-creep-ohne-refactoring.md)
<br/>  Neue Anforderungen werden als Nebeneffekte an bestehende Funktionen angeflanscht, statt ordentlich in eigenständige Operationen getrennt zu werden.
- [Schlechte Kapselung](schlechte-kapselung.md)
<br/>  Fehlende ordentliche Kapselung erlaubt Funktionen, über Modulgrenzen hinweg Zustand als undokumentierte Nebeneffekte zu ändern.

## Detection Methods ○
- **Code-Analyse:** Überprüfung von Funktionsimplementierungen zur Identifikation von Aktionen über ihren offensichtlichen Zweck hinaus
- **Dokumentation von Nebeneffekten:** Erstellung eines Katalogs aller Nebeneffekte, die jede Funktion erzeugt
- **Testkomplexität:** Identifikation von Funktionen, die umfangreiches Mocking oder Setup zum Testen erfordern
- **Entwicklerinterviews:** Befragung von Teammitgliedern zu Funktionen, die sich anders verhalten als erwartet
- **Statische Analysewerkzeuge:** Nutzung von Werkzeugen, die Funktionen mit mehreren Verantwortlichkeiten oder externen Abhängigkeiten identifizieren können

## Examples

Eine Funktion namens `calculateUserDiscount()` scheint einfach einen Rabattprozentsatz für einen Nutzer zu berechnen. Bei genauerer Untersuchung zeigt sich jedoch, dass sie auch: den "letzte Rabattberechnung"-Zeitstempel des Nutzers in der Datenbank aktualisiert, die Berechnung an einen Analytics-Dienst protokolliert, eine Werbe-E-Mail sendet, falls der Nutzer für ein Sonderangebot qualifiziert, einen Cache von Rabattsätzen aktualisiert und eine Webhook-Benachrichtigung an ein Marketing-System auslöst. Wenn Entwickler diese Funktion während Unit-Tests oder in Batch-Verarbeitungsszenarien aufrufen, lösen sie unwissentlich E-Mails, Webhook-Aufrufe und Datenbankaktualisierungen aus. Die versteckten Nebeneffekte machen es unmöglich, die Funktion sicher in Kontexten zu nutzen, in denen nur die Berechnung benötigt wird. Ein weiteres Beispiel betrifft eine `getUserProfile()`-Methode, die Nutzerdaten abruft, aber auch stillschweigend den "zuletzt aufgerufen"-Zeitstempel des Nutzers aktualisiert, einen Seitenaufruf-Zähler erhöht, den Zugriff für Sicherheitsaudits protokolliert und zwischengespeicherte Nutzerpräferenzen aktualisiert. Diese versteckten Nebeneffekte verursachen Probleme, wenn die Funktion mehrfach in einer einzigen Anfrage aufgerufen wird oder in administrativen Werkzeugen genutzt wird, wo die Nebeneffekte unerwünscht sind.
