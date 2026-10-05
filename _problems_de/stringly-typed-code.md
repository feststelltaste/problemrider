---
title: Stringly Typed Code
description: Fachliche Konzepte wie Status, Typ oder Identität werden als rohe,
  unvalidierte Strings statt als Enums oder Value Objects dargestellt, wodurch
  die Typprüfung von der Kompilierzeit in die Laufzeit verschoben wird.
category:
- Code
related_problems:
- slug: brittle-codebase
  similarity: 0.5
solutions:
- domain-driven-design
- value-range-definition
- static-analysis-and-linting
- typed-schema-extraction
- consistent-terminology
- code-conventions
- contract-testing
- property-based-testing
layout: problem
lang: de
en_slug: stringly-typed-code
---

## Description

Stringly Typed Code stellt fachliche Konzepte mit einer kleinen, klar definierten Menge gültiger Werte oder mit fachlicher Bedeutung als reine Strings dar statt als Enums, Value Objects oder andere dedizierte Typen. Der Status eines Nutzers, eine Zahlungsmethode, eine Berechtigungsstufe oder ein Workflow-Schritt werden am Ende als `"ACTIVE"`, `"PAID"` oder `"admin"` herumgereicht, statt als Typ, den der Compiler prüfen kann. Dies verschiebt die Validierung von der Kompilierzeit zur Laufzeit, oder schlimmer, zu welcher Funktion auch immer den String zuerst parst. Weil es keine einzige Stelle gibt, die die Menge gültiger Werte besitzt, werden zugehörige Prüfungen, Vergleiche und Transformationen überall dort neu implementiert, wo der String verwendet wird, und Tippfehler oder unbehandelte Werte rutschen still durch das Typsystem. Der Begriff ist ein Wortspiel mit „strongly typed“ (stark typisiert) und wird von Entwicklern seit Jahren informell verwendet, um diese spezielle Spielart der Primitive Obsession zu beschreiben.

## Indicators ⟡

- Ein einzelner fachlicher Wert, etwa Status, Typ, Rolle oder Modus, taucht als String-Literal an mehr als einer Handvoll Stellen auf, jede mit eigener Vergleichslogik.
- Die Validierung desselben String-Werts ist in verschiedenen Modulen leicht unterschiedlich implementiert.
- Der Code enthält `if (status.equals("ACTIVE"))` oder `switch`-Anweisungen über String-Literale statt über Enum-Konstanten.
- Tippfehler in String-Literalen, etwa `"activ"` statt `"active"`, werden, wenn überhaupt, nur durch einen fehlschlagenden Test oder einen Bugreport entdeckt.
- Die Menge gültiger Werte für ein Feld lässt sich nur durch Grep in der Codebasis herausfinden, nicht durch das Lesen einer Typdefinition.

## Symptoms ▲

- [Erhöhtes Risiko für Fehler](erhoehtes-risiko-fuer-fehler.md)
<br/>  Ohne eine vom Compiler geprüfte Menge gültiger Werte rutschen Tippfehler und unbehandelte Fälle in String-Vergleichen durch und zeigen sich als Laufzeitdefekte.
- [Code-Duplizierung](code-duplizierung.md)
<br/>  Weil kein einziger Typ die gültigen Werte oder ihre Validierung besitzt, wird dieselbe Vergleichs- und Parsing-Logik überall dort neu implementiert, wo der String verwendet wird.
- [Dynamic Connascence](dynamic-connascence.md)
<br/>  Sobald ein Stringly-Typed-Wert auch dynamische Lookups, Präfixe oder Query-Pfade steuert, wird aus dem Wertproblem eine versteckte Laufzeitkopplung zwischen Erzeuger und Verbraucher.
- [Schwer verständliche Codebasis](schwer-verstaendliche-codebasis.md)
<br/>  Leser können die gültigen Werte oder die beabsichtigte Bedeutung eines Feldes nicht aus seinem Typ erschließen; sie müssen den Code nachverfolgen, um ein implizites Enum zu rekonstruieren.

## Causes ▼

- [Missverständnis der Objektorientierung](missverstaendnis-der-objektorientierung.md)
<br/>  Entwickler, die nicht zu Value Objects oder Enums als Modellierungswerkzeug greifen, verwenden standardmäßig den Primitivtyp, der gerade bequem ist — meist einen String.
- [Termindruck](termindruck.md)
<br/>  Ein Enum oder Value Object einzuführen erfordert mehr Änderungen an mehr Stellen, als einfach einen String weiterzureichen, daher ist es die Abkürzung unter Zeitdruck.
- [Unerfahrene Entwickler](unerfahrene-entwickler.md)
<br/>  Entwickler, die mit typisierten Alternativen zu rohen Strings nicht vertraut sind, erkennen die langfristigen Kosten des Verzichts darauf nicht.

## Detection Methods ○

- **Clustering von String-Literalen:** Nutzen Sie statische Analyse, um dasselbe String-Literal zu finden, das in vielen unzusammenhängenden Dateien verglichen oder zugewiesen wird — ein Zeichen dafür, dass es ein nicht modelliertes fachliches Konzept repräsentiert.
- **Suche nach Gleichheitsketten:** Durchsuchen Sie den Code per Grep nach wiederholten `.equals("...")`, `== "..."` oder stringbasierten `switch`-Anweisungen über dieselbe kleine Wertemenge.
- **Review von Tippfehler-Vorfällen:** Prüfen Sie Bugtracker auf Defekte, die durch einen falsch geschriebenen oder unerwarteten String-Wert verursacht wurden, der eigentlich ein Enum hätte sein sollen.
- **Schema-Review:** Prüfen Sie Datenbankspalten und API-Payload-Felder auf Freitext-String-Typen, die eigentlich eine feste Menge fachlicher Zustände abbilden.

## Examples

Ein Auftragsverwaltungssystem stellt den Auftragsstatus als `String`-Feld dar, das `"PENDING"`, `"SHIPPED"`, `"DELIVERED"` oder `"CANCELLED"` sein kann. Im Laufe der Zeit implementieren fünf verschiedene Module jeweils ihre eigene Vergleichslogik: Das Versandmodul prüft `status.equals("SHIPPED")`, das Reporting-Modul prüft `"shipped".equalsIgnoreCase(status)`, und das Kundenbenachrichtigungsmodul prüft eine lokal zwischengespeicherte Liste erlaubter Strings, die ein Entwickler zu aktualisieren vergaß, als an anderer Stelle `"RETURNED"` als neuer Status eingeführt wurde. Die Rücksendung eines Kunden löst stillschweigend keine Rückerstattungsbenachrichtigung aus, weil die hartcodierte String-Liste des Benachrichtigungsmoduls den neuen Status nicht kennt. Wäre der Auftragsstatus als Enum modelliert gewesen, hätte das Hinzufügen von `RETURNED` den Compiler gezwungen, jede `switch`-Anweisung zu markieren, die den neuen Fall nicht behandelte.
