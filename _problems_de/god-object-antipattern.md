---
title: God-Object-Antipattern
description: Einzelne Klassen oder Komponenten übernehmen zu viele Verantwortlichkeiten
  und werden übermäßig komplex sowie schwer zu warten oder zu testen.
category:
- Architecture
- Code
related_problems:
- slug: monolithic-functions-and-classes
  similarity: 0.8
- slug: excessive-class-size
  similarity: 0.7
- slug: poorly-defined-responsibilities
  similarity: 0.65
- slug: bloated-class
  similarity: 0.65
- slug: complex-implementation-paths
  similarity: 0.6
- slug: large-pull-requests
  similarity: 0.6
solutions:
- incremental-refactoring
- modularization-and-bounded-contexts
- high-cohesion
- dependency-injection-container
- solid-principles
- separation-of-concerns
- dependency-breaking-techniques
- code-hotspot-analysis
- preparatory-refactoring
- domain-driven-design
- lightweight-design-review
layout: problem
lang: de
en_slug: god-object-anti-pattern
---

## Description

Das God-Object-Antipattern tritt auf, wenn einzelne Klassen oder Komponenten zu viele Verantwortlichkeiten anhäufen und übermäßig komplex werden, wobei sie oft mehrere unzusammenhängende Belange innerhalb einer einzigen Einheit behandeln. Diese Objekte werden schwer zu verstehen, zu warten, zu testen und zu ändern, weil sie das Single-Responsibility-Prinzip verletzen und Engpässe für Entwicklung und Wartung schaffen.

## Indicators ⟡

- Klassen mit Hunderten oder Tausenden Zeilen Code
- Einzelne Objekte, die mehrere unzusammenhängende Geschäftsbelange handhaben
- Methoden, die viele verschiedene Arten von Operationen durchführen
- Klassen, die schwer zu benennen sind, weil sie zu viel tun
- Komponenten, die mehrere Teams aus unterschiedlichen Gründen ändern müssen

## Symptoms ▲

- [Schwer testbarer Code](schwer-testbarer-code.md)
<br/>  God Objects erfordern umfangreiches Setup und Mocking zum Testen, weil sie gleichzeitig von vielen unzusammenhängenden Belangen abhängen.
- [Merge-Konflikte](merge-konflikte.md)
<br/>  Mehrere Entwickler müssen häufig aus unterschiedlichen Gründen dasselbe God Object ändern, was ständige Versionskontrollkonflikte verursacht.
- [Hohe Rate an neu eingeführten Fehlern](hohe-rate-an-neu-eingefuehrten-fehlern.md)
<br/>  Änderungen an einer Verantwortlichkeit innerhalb eines God Objects riskieren, andere unzusammenhängende Verantwortlichkeiten zu brechen, die es handhabt.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Entwickler müssen das gesamte God Object verstehen, bevor sie sicher irgendeinen Teil davon ändern können, was die Entwicklung erheblich verlangsamt.
- [Erhöhte kognitive Last](erhoehte-kognitive-last.md)
<br/>  Das Verständnis eines God Objects erfordert, viele unzusammenhängende Konzepte gleichzeitig im Kopf zu behalten.
- [Wellenwirkung von Änderungen](wellenwirkung-von-aenderungen.md)
<br/>  Das Ändern einer Verantwortlichkeit innerhalb eines God Objects erfordert oft Änderungen an anderen Teilen desselben Objekts und seiner Konsumenten.
- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  God Objects mit Tausenden Zeilen und Dutzenden Methoden sind inhärent schwer zu verstehen, was dies zu einem direkten Symptom macht.

## Causes ▼

- [Schlecht definierte Verantwortlichkeiten](schlecht-definierte-verantwortlichkeiten.md)
<br/>  Ohne klare Verantwortlichkeitsgrenzen wird neue Funktionalität zu bestehenden großen Objekten hinzugefügt, statt ordentlich getrennt zu werden.
- [Feature-Creep ohne Refactoring](feature-creep-ohne-refactoring.md)
<br/>  Das kontinuierliche Hinzufügen von Features ohne Refactoring führt dazu, dass Klassen im Laufe der Zeit Verantwortlichkeiten anhäufen.
- [Missverständnis der Objektorientierung](missverstaendnis-der-objektorientierung.md)
<br/>  Mangelndes Verständnis des Single-Responsibility-Prinzips und ordentlichen OO-Designs führt zu monolithischen Klassenstrukturen.
- [Refactoring-Vermeidung](refactoring-vermeidung.md)
<br/>  Teams vermeiden es, wachsende Klassen aufzuteilen, wegen des wahrgenommenen Risikos, wodurch God Objects unkontrolliert wachsen können.

## Detection Methods ○

- **Code-Metrik-Analyse:** Beobachtung von Klassengröße, Methodenanzahl und Komplexitätsmetriken
- **Verantwortlichkeitsanalyse:** Überprüfung, was verschiedene Methoden und Eigenschaften in Klassen tun
- **Änderungsauswirkungsanalyse:** Nachverfolgung, wie oft und warum große Objekte geändert werden
- **Testabdeckungsanalyse:** Identifikation von Objekten, die schwer umfassend zu testen sind
- **Team-Kollaborationsmetriken:** Beobachtung, wie oft mehrere Entwickler dieselben Objekte ändern

## Examples

Eine E-Commerce-Anwendung hat eine `OrderManager`-Klasse, die Bestellerstellung, Zahlungsabwicklung, Bestandsaktualisierungen, Versandberechnungen, Steuerberechnungen, Kundenbenachrichtigungen, Bestellstatus-Verfolgung, Rückerstattungsabwicklung und Reporting handhabt. Die Klasse hat über 2.000 Zeilen Code und 50+ Methoden. Wenn sich die Steuerberechnungslogik ändern muss, riskieren Entwickler, die Zahlungsabwicklung zu brechen. Wenn die Bestandsverwaltung Aktualisierungen benötigt, betrifft das die Versandberechnungen. Die Klasse ist so komplex, dass umfassendes Testen die Einrichtung von Datenbanken, Zahlungsabwicklern, Versanddiensten und E-Mail-Systemen erfordert, was Unit-Testing nahezu unmöglich macht. Ein weiteres Beispiel betrifft ein Nutzerverwaltungssystem mit einer `User`-Klasse, die Authentifizierung, Autorisierung, Profilverwaltung, Präferenzen, Benachrichtigungseinstellungen, Aktivitätsverfolgung, Freundschaftsbeziehungen, Content-Erstellung und Reporting handhabt. Jede Änderung an Nutzerpräferenzen betrifft den Authentifizierungscode, und Änderungen an Freundschaftsbeziehungen können Content-Erstellungsfeatures brechen, was das System brüchig und schwer zu warten macht.
