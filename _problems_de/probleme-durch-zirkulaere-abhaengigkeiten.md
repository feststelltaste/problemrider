---
title: Probleme durch zirkuläre Abhängigkeiten
description: Komponenten hängen in zirkulären Mustern voneinander ab, was Initialisierungsprobleme,
  Testschwierigkeiten und architektonische Komplexität verursacht.
category:
- Architecture
- Code
related_problems:
- slug: tight-coupling-issues
  similarity: 0.7
- slug: circular-references
  similarity: 0.65
- slug: hidden-dependencies
  similarity: 0.65
- slug: dependency-version-conflicts
  similarity: 0.6
- slug: deployment-coupling
  similarity: 0.6
- slug: cascade-failures
  similarity: 0.6
solutions:
- event-driven-architecture
- incremental-refactoring
- modularization-and-bounded-contexts
- mediator
- change-impact-analysis
- high-cohesion
- separation-of-concerns
- dependency-injection
- architecture-conformity-analysis
- fitness-functions
layout: problem
lang: de
en_slug: circular-dependency-problems
---

## Description

Probleme durch zirkuläre Abhängigkeiten entstehen, wenn Komponenten in zirkulären Mustern voneinander abhängen, wobei Komponente A von Komponente B abhängt, die von Komponente C abhängt, die wiederum von Komponente A abhängt. Diese zirkulären Referenzen erzeugen Probleme mit der Initialisierungsreihenfolge, dem Testen, der Kompilierung und machen die Systemarchitektur komplexer und brüchiger.

## Indicators ⟡

- Build-Systeme melden Fehler durch zirkuläre Abhängigkeiten
- Komponenten können nicht unabhängig initialisiert oder geladen werden
- Unit-Testing erfordert komplexes Setup, um zirkuläre Referenzen zu durchbrechen
- Dependency-Injection-Frameworks können zirkuläre Abhängigkeiten nicht auflösen
- Modul-Lade-Systeme stoßen auf Fehler durch zirkuläre Referenzen

## Symptoms ▲

- [Schwer testbarer Code](schwer-testbarer-code.md)
<br/>  Zirkuläre Abhängigkeiten verhindern, dass Komponenten isoliert getestet werden können, was komplexe Mocking-Setups erfordert.
- [Lange Build- und Testzeiten](lange-build-und-testzeiten.md)
<br/>  Zirkuläre Abhängigkeiten erzwingen unnötige Neukompilierung abhängiger Module, was die Build-Zeiten erhöht.
- [Wellenwirkung von Änderungen](wellenwirkung-von-aenderungen.md)
<br/>  Änderungen an jeder Komponente im Zyklus erfordern Änderungen an anderen Komponenten aufgrund gegenseitiger Abhängigkeiten.
- [Debugging-Schwierigkeiten](debugging-schwierigkeiten.md)
<br/>  Zirkuläre Abhängigkeiten erschweren es, den Ausführungsfluss nachzuverfolgen und Probleme auf bestimmte Komponenten einzugrenzen.
- [Schwierige Code-Wiederverwendung](schwierige-code-wiederverwendung.md)
<br/>  Komponenten in Ketten zirkulärer Abhängigkeiten können nicht extrahiert und unabhängig in anderen Kontexten wiederverwendet werden.

## Causes ▼

- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Übermäßige Kopplung zwischen Komponenten führt zu bidirektionalen Abhängigkeiten, die zirkuläre Muster bilden.
- [Schlecht definierte Verantwortlichkeiten](schlecht-definierte-verantwortlichkeiten.md)
<br/>  Wenn Modulverantwortlichkeiten unklar sind, landet Funktionalität an falschen Stellen, was gegenseitige Abhängigkeiten schafft.
- [Implementierung beginnt ohne Design](implementierung-beginnt-ohne-design.md)
<br/>  Der Beginn der Entwicklung ohne vorheriges architektonisches Design lässt zirkuläre Abhängigkeiten organisch entstehen.
- [Monolithische Funktionen und Klassen](monolithische-funktionen-und-klassen.md)
<br/>  Große Klassen mit gemischten Verantwortlichkeiten tendieren dazu, von vielen anderen Klassen abzuhängen, was die Wahrscheinlichkeit von Ketten zirkulärer Abhängigkeiten erhöht.

## Detection Methods ○

- **Statische Abhängigkeitsanalyse:** Nutzung von Werkzeugen zur Erkennung zirkulärer Abhängigkeiten in der Codebasis
- **Build-Fehler-Monitoring:** Überwachung von Build-Prozessen auf Fehler durch zirkuläre Abhängigkeiten
- **Abhängigkeitsgraph-Visualisierung:** Erstellung visueller Darstellungen von Komponentenabhängigkeiten
- **Initialisierungsfluss-Analyse:** Analyse der Initialisierungsreihenfolge und Abhängigkeiten von Komponenten
- **Modul-Import-Analyse:** Überprüfung von Import-/Require-Anweisungen auf zirkuläre Muster

## Examples

Eine Webanwendung hat einen `UserService`, der von `OrderService` abhängt, um die Bestellhistorie des Nutzers zu erhalten, während `OrderService` von `UserService` abhängt, um Nutzerberechtigungen zu validieren. Beide Services werden beim Start initialisiert, aber keiner kann instanziiert werden, weil jeder erfordert, dass der andere zuerst erstellt wird. Der Dependency-Injection-Container schlägt mit einem Fehler durch zirkuläre Abhängigkeit fehl. Die zirkuläre Abhängigkeit macht es unmöglich, einen der beiden Services unabhängig als Unit zu testen, weil das Testen von `UserService` `OrderService` erfordert, das wiederum die Initialisierung von `UserService` erfordert. Ein weiteres Beispiel betrifft eine Frontend-Anwendung, bei der `ComponentA` `ComponentB` importiert und nutzt, während `ComponentB` Hilfsfunktionen aus `ComponentA` importiert, was eine zirkuläre Modulabhängigkeit erzeugt. Der JavaScript-Modul-Loader kann den zirkulären Import nicht auflösen, was dazu führt, dass die Anwendung während des Bundling-Prozesses fehlschlägt.
