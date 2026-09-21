---
title: Prozedurale Programmierung in OOP-Sprachen
description: Code wird im prozeduralen Stil innerhalb objektorientierter Sprachen
  geschrieben, was zu großen, monolithischen Funktionen und schlechter Kapselung
  führt.
category:
- Architecture
- Code
related_problems:
- slug: procedural-background
  similarity: 0.7
- slug: monolithic-functions-and-classes
  similarity: 0.55
- slug: poor-encapsulation
  similarity: 0.55
- slug: over-reliance-on-utility-classes
  similarity: 0.55
- slug: god-object-anti-pattern
  similarity: 0.55
- slug: spaghetti-code
  similarity: 0.5
solutions:
- clean-code
- separation-of-concerns
- solid-principles
- technical-skills-development
- code-reading-sessions
- internal-technical-coaching
- lightweight-design-review
- domain-driven-design
- communities-of-practice
- refactoring-katas
layout: problem
lang: de
en_slug: procedural-programming-in-oop-languages
---

## Description

Prozedurale Programmierung in OOP-Sprachen tritt auf, wenn Entwickler Code unter Nutzung prozeduraler Paradigmen innerhalb objektorientierter Programmiersprachen schreiben und es versäumen, die Vorteile objektorientierter Designprinzipien zu nutzen. Dies resultiert in Code, der prozeduralen Programmen ähnelt, mit langen Sequenzen von Anweisungen, minimaler Nutzung von Klassen und Objekten und schlechter Kapselung. Während prozedurale Programmierung ihren Platz hat, führt ihre unangemessene Nutzung in objektorientierten Kontexten zu Code, der schwer zu warten, zu testen und zu erweitern ist.

## Indicators ⟡
- Klassen enthalten primär statische Methoden mit wenig oder keinem Instanzzustand
- Lange Methoden, die mehrere sequenzielle Operationen ohne sinnvolle Objektinteraktionen durchführen
- Daten und Verhalten sind getrennt, wobei Datenstrukturen zwischen Utility-Methoden weitergereicht werden
- Minimale Nutzung von Vererbung, Polymorphie oder anderen objektorientierten Features
- Code ähnelt einer Reihe von Utility-Funktionen statt interagierender Objekte

## Symptoms ▲

- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  Lange prozedurale Methoden mit sequenzieller Logik sind schwerer zu verstehen als gut strukturierter OOP-Code.
- [Spaghetticode](spaghetticode.md)
<br/>  Ohne OOP-Struktur wächst prozeduraler Code zu verworrenen Sequenzen, die schwer zu verfolgen und zu modifizieren sind.
- [Schwierige Code-Wiederverwendung](schwierige-code-wiederverwendung.md)
<br/>  Prozeduraler Code koppelt Daten und Logik eng in monolithischen Funktionen, was Wiederverwendung über Kontexte hinweg unpraktikabel macht.
- [Schlechte Kapselung](schlechte-kapselung.md)
<br/>  Datenstrukturen werden zwischen Utility-Funktionen weitergereicht, statt in sinnvollen Objekten gekapselt zu sein.
- [Gemischte Coding-Stile](gemischte-coding-stile.md)
<br/>  Prozeduraler Code, gemischt mit OOP-Code von anderen Entwicklern, schafft inkonsistente Coding-Muster über die Codebasis hinweg.

## Causes ▼

- [Prozeduraler Hintergrund](prozeduraler-hintergrund.md)
<br/>  Entwickler, die in prozeduraler Programmierung ausgebildet sind, tragen diese Gewohnheiten in OOP-Sprachen hinein.
- [Missverständnis der Objektorientierung](missverstaendnis-der-objektorientierung.md)
<br/>  Entwickler, die OOP-Prinzipien nicht verstehen, verfallen standardmäßig auf den prozeduralen Stil, den sie kennen.
- [Unzureichende Design-Fähigkeiten](unzureichende-design-faehigkeiten.md)
<br/>  Fehlende Design-Fähigkeiten hindern Entwickler daran zu erkennen, wann OOP-Muster angemessener wären.

## Detection Methods ○
- **Analyse statischer Methoden:** Identifikation von Klassen mit hohem Anteil statischer Methoden relativ zu Instanzmethoden
- **Klassen-Kohäsions-Metriken:** Messung, wie gut Methoden und Daten innerhalb von Klassen zusammenwirken
- **Methodenlängenanalyse:** Suche nach ungewöhnlich langen Methoden, die sequenzielle Operationen durchführen
- **Objektinteraktionsanalyse:** Untersuchung, ob Objekte sinnvoll interagieren oder nur als Datencontainer dienen
- **Nutzung von Design-Mustern:** Bewertung, ob Code objektorientierte Designmuster angemessen nutzt

## Examples

Eine Java-Anwendung zur Verarbeitung von Kundenbestellungen enthält eine `CustomerOrderProcessor`-Klasse mit einer einzigen statischen Methode `processOrder(OrderData orderData)`, die 800 Zeilen lang ist. Die Methode führt Validierung, Bestandsprüfung, Zahlungsverarbeitung, Versandberechnung, E-Mail-Versand und Datenbank-Updates auf sequenzielle, prozedurale Weise durch. Statt sinnvolle Objekte wie `Order`, `PaymentProcessor`, `InventoryManager` und `ShippingCalculator` zu erstellen, die Verhalten und Zustand kapseln, ist die gesamte Logik in prozeduralen Funktionen enthalten, die Datenstrukturen zwischen sich weiterreichen. Wenn neue Bestellungstypen hinzugefügt werden, muss die gesamte Funktion modifiziert werden, was das Open-Closed-Prinzip verletzt und den Code zunehmend komplex macht. Ein weiteres Beispiel betrifft ein C#-Content-Management-System, bei dem alle Funktionalität in statischen Utility-Klassen wie `ContentUtils`, `UserUtils` und `DatabaseUtils` implementiert ist. Diese Klassen enthalten Dutzende statische Methoden, die Data-Transfer-Objekte manipulieren, aber es gibt keine sinnvollen Domänenobjekte, die Geschäftsverhalten kapseln. Das Hinzufügen neuer Content-Typen erfordert Modifikationen über mehrere Utility-Klassen hinweg, und das Fehlen von Polymorphie bedeutet, dass umfangreiche If-Else-Anweisungen genutzt werden, um verschiedene Content-Typen in der gesamten Codebasis zu handhaben.
