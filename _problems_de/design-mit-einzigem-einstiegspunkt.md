---
title: Design mit einzigem Einstiegspunkt
description: Ein Design, bei dem alle Anfragen an ein System durch ein einziges Objekt
  oder eine einzige Komponente gehen müssen.
category:
- Architecture
related_problems:
- slug: single-points-of-failure
  similarity: 0.5
- slug: god-object-anti-pattern
  similarity: 0.5
solutions:
- architecture-reviews
- separation-of-concerns
- solid-principles
- api-gateway
- modularization-and-bounded-contexts
- loose-coupling
- high-cohesion
- layered-architecture
- hexagonal-architecture
layout: problem
lang: de
en_slug: single-entry-point-design
---

## Description
Ein Design mit einzigem Einstiegspunkt ist ein Design, bei dem alle Anfragen an ein System durch ein einziges Objekt oder eine einzige Komponente gehen müssen. Dies kann ein Problem sein, weil es zu einem God-Object-Antipattern führen kann, bei dem der einzige Einstiegspunkt für zu viele Dinge verantwortlich wird. Es kann auch einen Wartungsengpass schaffen, da alle Änderungen am System durch den einzigen Einstiegspunkt gehen müssen.

## Indicators ⟡
- Eine einzige Klasse oder Komponente, die für die Handhabung aller eingehenden Anfragen verantwortlich ist.
- Der einzige Einstiegspunkt ist oft sehr groß und komplex.
- Es ist schwierig, Änderungen am System vorzunehmen, ohne den einzigen Einstiegspunkt zu berühren.
- Der einzige Einstiegspunkt ist eine häufige Quelle von Bugs.

## Symptoms ▲

- [God-Object-Antipattern](god-object-antipattern.md)
<br/>  Der einzige Einstiegspunkt häuft über die Zeit Verantwortlichkeiten an und wird zu einem God-Object, das zu viele Belange handhabt.
- [Wartungsengpässe](wartungsengpaesse.md)
<br/>  Alle Änderungen müssen durch den einzigen Einstiegspunkt fließen, was einen Engpass schafft, an dem sich Modifikationen aufstauen und die Entwicklung verlangsamen.
- [Brüchige Codebasis](bruechige-codebasis.md)
<br/>  Änderungen am einzigen Einstiegspunkt riskieren, viele nicht verwandte Features zu brechen, da alle Anfragen von ihm abhängen.
- [Hohe Kopplung und geringe Kohäsion](hohe-kopplung-und-geringe-kohaesion.md)
<br/>  Alle Komponenten werden durch den einzigen Einstiegspunkt gekoppelt, was exzessive gegenseitige Abhängigkeiten schafft.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Das Hinzufügen neuer Features erfordert die Modifikation des einzigen Einstiegspunkts, was aufgrund seiner Komplexität riskant und zeitaufwendig ist.

## Causes ▼

- [Einschränkungen durch monolithische Architektur](einschraenkungen-durch-monolithische-architektur.md)
<br/>  Monolithische Designs leiten naturgemäß alle Anfragen durch zentralisierte Komponenten statt Verantwortung zu verteilen.

## Detection Methods ○
- **Code-Reviews:** Suche nach einzelnen Klassen oder Komponenten, die für die Handhabung aller eingehenden Anfragen verantwortlich sind.
- **Statische Analysewerkzeuge:** Nutzung von Werkzeugen zur Identifikation großer Klassen und Klassen mit einer großen Anzahl an Abhängigkeiten.
- **Architekturdiagramme:** Erstellung eines Diagramms der Systemarchitektur zur Identifikation einzelner Eintrittspunkte.

## Examples
Eine Webanwendung hat ein einziges `FrontController`-Servlet, das für die Handhabung aller eingehenden HTTP-Anfragen verantwortlich ist. Der `FrontController` ist für das Routing von Anfragen an den passenden Handler verantwortlich, aber auch für Authentifizierung, Autorisierung, Logging und eine Reihe anderer Querschnittsbelange. Der `FrontController` umfasst über 1000 Codezeilen und hat Abhängigkeiten zu Dutzenden anderer Klassen. Er ist ein bedeutender Wartungsengpass und eine häufige Bug-Quelle.
