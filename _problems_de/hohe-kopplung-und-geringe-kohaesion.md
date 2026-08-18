---
title: Hohe Kopplung und geringe Kohäsion
description: Softwarekomponenten sind übermäßig voneinander abhängig und übernehmen
  zu viele unzusammenhängende Funktionen, was das System schwer änderbar und verständlich
  macht.
category:
- Architecture
- Code
related_problems:
- slug: tight-coupling-issues
  similarity: 0.75
- slug: poor-encapsulation
  similarity: 0.7
- slug: unpredictable-system-behavior
  similarity: 0.65
- slug: poorly-defined-responsibilities
  similarity: 0.65
- slug: deployment-coupling
  similarity: 0.65
- slug: difficult-to-understand-code
  similarity: 0.6
solutions:
- architecture-reviews
- loose-coupling
- separation-of-concerns
- solid-principles
- abstraction
- architecture-conformity-analysis
- architecture-governance
- aspect-oriented-programming-aop
- bounded-contexts
- bridges
- bubble-context
- bulkhead
- event-driven-integration
- facades
- high-cohesion
- layered-architecture
- mediator
- microservices-architecture
- modulith
- object-relational-mapping-orm
- dependency-injection
- dependency-injection-container
- domain-aligned-architecture
- domain-driven-design
- event-driven-architecture
- fitness-functions
- incremental-refactoring
- modularization-and-bounded-contexts
- dependency-breaking-techniques
- mikado-method
layout: problem
lang: de
en_slug: high-coupling-low-cohesion
---

## Description
Hohe Kopplung und geringe Kohäsion sind zwei der häufigsten Design-Probleme in der Softwareentwicklung. Kopplung bezeichnet den Grad der gegenseitigen Abhängigkeit zwischen Softwaremodulen, während Kohäsion den Grad bezeichnet, in dem die Elemente eines Moduls zusammengehören. Ein gut gestaltetes System sollte geringe Kopplung und hohe Kohäsion haben. Das macht das System leichter verständlich, wartbar und erweiterbar. Ein System mit hoher Kopplung und geringer Kohäsion hingegen ist ein Albtraum, mit dem zu arbeiten.

## Indicators ⟡
- Eine kleine Änderung in einem Teil des Systems erfordert Änderungen in vielen anderen scheinbar unzusammenhängenden Teilen.
- Es ist schwer, den Zweck eines Moduls oder einer Funktion zu verstehen, ohne viele andere Teile des Systems zu verstehen.
- Änderungen führen leicht zu neuen Fehlern aufgrund unerwarteter Nebeneffekte in eng gekoppelten Komponenten.
- Entwickler verbringen mehr Zeit damit, Abhängigkeiten zu navigieren und komplexe Interaktionen zu verstehen.

## Symptoms ▲

- [Wellenwirkung von Änderungen](wellenwirkung-von-aenderungen.md)
<br/>  Kleine Änderungen in einem Modul erfordern aufgrund enger Kopplung zwischen Komponenten Modifikationen in vielen anderen Modulen.
- [Schwer testbarer Code](schwer-testbarer-code.md)
<br/>  Eng gekoppelte Komponenten können nicht isoliert getestet werden, weil sie stark von anderen Komponenten abhängen.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Entwickler müssen selbst für einfache Feature-Ergänzungen mehrere voneinander abhängige Komponenten verstehen und ändern.
- [Hohe Rate an neu eingeführten Fehlern](hohe-rate-an-neu-eingefuehrten-fehlern.md)
<br/>  Änderungen in eng gekoppeltem Code haben unbeabsichtigte Auswirkungen in abhängigen Komponenten, was häufig neue Fehler einführt.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Die unvorhersehbaren kaskadierenden Auswirkungen von Änderungen in gekoppeltem Code machen Entwickler zurückhaltend, das System zu ändern.
- [Erhöhte kognitive Last](erhoehte-kognitive-last.md)
<br/>  Das Verständnis einer einzelnen Komponente erfordert das Verständnis vieler anderer Komponenten, mit denen sie gekoppelt ist, was Entwickler überfordert.
- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  Stark gekoppelter Code mit geringer Kohäsion erfordert das Verständnis vieler miteinander verbundener Module, um auch nur einen einzelnen Teil zu verstehen.

## Causes ▼

- [Schlecht definierte Verantwortlichkeiten](schlecht-definierte-verantwortlichkeiten.md)
<br/>  Ohne klare Modulverantwortlichkeiten verteilt sich Funktionalität über mehrere Komponenten, was unnötige Abhängigkeiten schafft.
- [Feature-Creep ohne Refactoring](feature-creep-ohne-refactoring.md)
<br/>  Das Hinzufügen von Features ohne Umstrukturierung der Codebasis führt dazu, dass Verantwortlichkeiten im Laufe der Zeit über Modulgrenzen hinweg verschwimmen.
- [Schlechte Kapselung](schlechte-kapselung.md)
<br/>  Das Offenlegen interner Implementierungsdetails erlaubt anderen Modulen, davon abhängig zu werden, was enge Kopplung schafft.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler, die mit Designprinzipien nicht vertraut sind, schaffen eng gekoppelten Code, der unzusammenhängende Belange innerhalb einzelner Module vermischt.
- [Refactoring-Vermeidung](refactoring-vermeidung.md)
<br/>  Das Vermeiden notwendigen Refactorings erlaubt es der Kopplung, sich anzuhäufen, während das System wächst und sich weiterentwickelt.

## Detection Methods ○

- **Code-Metrik-Werkzeuge:** Nutzung statischer Analysewerkzeuge, die Kopplung (z. B. afferente/efferente Kopplung, CBO – Coupling Between Objects) und Kohäsion (z. B. LCOM – Lack of Cohesion in Methods) messen.
- **Code-Review:** Achten auf Code, der schwer verständlich ist, viele Abhängigkeiten hat oder mehrere unzusammenhängende Aufgaben ausführt.
- **Abhängigkeitsgraphen:** Visualisierung der Abhängigkeiten zwischen Modulen oder Klassen zur Identifikation stark gekoppelter Komponenten.
- **Refactoring-Herausforderungen:** Wenn sich das Refactoring eines kleinen Teils des Systems als extrem schwierig oder riskant erweist, ist das ein Zeichen für hohe Kopplung.

## Examples
Ein Legacy-E-Commerce-System hat eine einzige `OrderProcessor`-Klasse, die alles von der Validierung von Kundendaten über die Steuerberechnung, Zahlungsabwicklung, Bestandsaktualisierung bis hin zu E-Mail-Benachrichtigungen handhabt. Eine kleine Änderung an der Steuerberechnungslogik erfordert das Verständnis und potenziell die Modifikation der gesamten `OrderProcessor`-Klasse, mit dem Risiko unbeabsichtigter Nebeneffekte auf Zahlungsabwicklung oder E-Mail-Versand. In einem anderen Fall greift eine Utility-Funktion `calculate_total` in einer Python-Anwendung direkt auf ein globales `database_connection`-Objekt und eine globale `logging_level`-Variable zu und ändert diese. Das macht es unmöglich, `calculate_total` isoliert zu testen, ohne eine echte Datenbankverbindung einzurichten und die globale Logging-Konfiguration zu beeinflussen. Dieses Problem ist ein fundamentaler Designfehler, der sich oft über die Zeit in Systemen anhäuft, denen kontinuierliche architektonische Aufsicht und Refactoring fehlen. Es ist ein wesentlicher Beitragender zu technischen Schulden und macht die Modernisierung von Legacy-Systemen extrem herausfordernd.
