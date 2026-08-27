---
title: Extraktion typisierter Schemata
description: Beförderung der Attribute, die wirklich Teil der Domäne
  sind, in typisierte, eingeschränkte Spalten, sodass nur der spärliche
  Rest generisch bleibt.
category:
- Database
- Architecture
- Code
problems:
- entity-attribute-value-overuse
- database-schema-design-problems
- slow-database-queries
- high-number-of-database-queries
- n-plus-one-query-problem
- difficult-to-understand-code
- increased-bug-count
- schema-evolution-paralysis
- testing-complexity
- imperative-data-fetching-logic
- high-technical-debt
- data-migration-integrity-issues
layout: solution
lang: de
en_slug: typed-schema-extraction
related_solutions:
- slug: attribute-usage-analysis
  similarity: 0.7
- slug: evolutionary-database-design
  similarity: 0.7
- slug: query-optimization-process
  similarity: 0.6
- slug: nosql-databases
  similarity: 0.6
- slug: index-lifecycle-management
  similarity: 0.6
- slug: production-like-test-data
  similarity: 0.6
---

## Description

Die Extraktion typisierter Schemata verschiebt die Attribute, die tatsächlich Teil der Domäne sind, aus einem generischen Attributspeicher heraus in echte Spalten mit echten Typen und echten Constraints, während genuin spärliche oder genuin unvorhersehbare Daten in der flexiblen Form belassen werden. Es ist die inkrementelle Antwort auf ein generisches Datenmodell, und es funktioniert, weil solche Modelle fast nie einheitlich generisch sind. Sie enthalten einen stabilen Kern, den jeder nutzt — der nichts davon gewinnt, generisch gespeichert zu werden, und Typsicherheit, Constraints, Indizierung und Lesbarkeit verliert — umgeben von einem langen Schwanz, wo die Flexibilität echte Arbeit leistet. Der Versuch, das generische Modell vollständig zu eliminieren, scheitert, weil der Schwanz echte Vielfalt hat. Der Versuch, es unangetastet zu lassen, scheitert, weil der Kern ist, wo die Kosten liegen. Extraktion nimmt den Kern und hört auf.

## How to Apply ◆

> Die Attribute, die sich zur Extraktion lohnen, identifizieren sich selbst: Sie sind auf fast jeder Entität ausgefüllt, sie werden gefiltert und sortiert, und sie haben einen offensichtlichen Typ.

- **Wählen Sie Kandidaten aus Evidenz**, nicht aus Intuition: hoher Ausfüllungsanteil, kleine Anzahl distinkter Werte oder ein klarer Typ, und Auftreten in Abfragefiltern. Ein Attribut, das auf 98 Prozent der Entitäten gesetzt ist und in jeder Suche genutzt wird, ist der Archetyp.
- **Extrahieren Sie in kleinen Gruppen**, nicht alles auf einmal. Drei oder vier verwandte Attribute, die ein Konzept bilden, sind ein handhabbares Inkrement, das unabhängig verifiziert und ausgeliefert werden kann.
- **Fügen Sie zuerst die typisierten Spalten hinzu und schreiben Sie in beide**, wobei die generischen Zeilen die Wahrheitsquelle bleiben. Nichts liest die neuen Spalten noch, sodass nichts kaputtgehen kann, und der Dual-Write stellt fest, ob die Daten überhaupt typisiert werden können.
- **Lassen Sie die Migration die Datenqualität offenbaren.** Die Konvertierung von Textwerten in eine typisierte Spalte wird bei den Einträgen fehlschlagen, die nie gültig waren, und diese Fehlschläge sind Befunde, über die entschieden werden muss, statt sie still zu erzwingen. Erwarten Sie, dass dieser Schritt länger dauert als die Schemaänderung.
- **Verschieben Sie Leser einzeln**, wobei jeder gegen den generischen Pfad verifiziert wird, bevor umgeschaltet wird. Der Vergleich der beiden Pfade auf Live-Traffic ist es, was die Umschaltung zu einer Messung statt einem Sprung macht.
- **Fügen Sie die Constraints hinzu, sobald Leser umgezogen sind**: Not Null, wo die Daten es unterstützen, Fremdschlüssel, wo eine Beziehung existiert, Checks, wo der Wertebereich bekannt ist. Die Constraints sind der Großteil des Nutzens, und ihr Hinzufügen ist der Schritt, der aufgeschoben wird.
- **Hören Sie auf, die generischen Zeilen zu schreiben, und löschen Sie sie dann**, an einem festgelegten Datum. Beide Pfade unbegrenzt gefüllt zu lassen bedeutet, zwei Modelle zu tragen und nichts zu gewinnen, was das häufigste Ende dieser Arbeit ist.
- **Halten Sie den Schwanz generisch und sagen Sie es explizit.** Zu dokumentieren, welche Daten legitim flexibel bleiben, und warum, verhindert, dass der nächste Entwickler entweder das generische Modell zurück in den Kern erweitert oder versucht, es vollständig zu eliminieren.
- **Erwägen Sie eine strukturierte Dokumentenspalte für den Schwanz**, wo die Datenbank eine unterstützt. Sie behält Flexibilität, während sie Validierung und Indizierung bekannter Pfade erlaubt, was strikt besser ist als untypisierte Attributzeilen für die meisten verbleibenden Fälle.
- **Führen Sie die Nutzungsanalyse anschließend erneut aus**, um zu bestätigen, dass der Schwanz nicht nachwächst, und paaren Sie es mit einer Regel, dass neue Felder in das typisierte Modell gehen, es sei denn, es gibt einen angegebenen Grund.

## Tradeoffs ⇄

> Extraktion stellt wieder her, wofür die Datenbank da ist — Typen, Constraints, Indizes, Lesbarkeit — auf Kosten einer sorgfältigen Migration und des Verlusts der Fähigkeit, ein Feld ohne Schemaänderung hinzuzufügen.

**Vorteile:**

- Abfragen werden effizient und indizierbar, und die Rekonstruktion einer Entität erfordert nicht mehr einen Join oder Pivot über viele Zeilen hinweg.
- Die Datenbank setzt Korrektheit wieder durch, sodass eine ganze Defektklasse zur Schreibzeit abgelehnt wird, statt weit entfernt zur Lesezeit zutage zu treten.
- Die Domäne wird im Schema sichtbar, was eine erhebliche Verbesserung der Lesbarkeit für jeden ist, der neu im System ist.
- Berichterstattung kann gegen das echte Modell laufen statt gegen eine separat gepflegte abgeflachte Kopie, was diese Kopie und ihre Abweichung beseitigt.
- Die Migration selbst bringt angesammelte Datenqualitätsprobleme zutage, die die untypisierte Spalte verborgen hatte, oft jahrelang.

**Kosten und Risiken:**

- Ein Feld jetzt hinzuzufügen erfordert eine Schemaänderung, was genau die Reibung ist, die das generische Modell produzierte — sodass diese Arbeit verschwendet ist, es sei denn, Schemaänderungen wurden auch routinemäßig gemacht.
- Die Migration wird Daten finden, die nicht typisiert werden können, und jeder Fall ist eine Entscheidung, die Domänenwissen erfordert, das möglicherweise nicht mehr in der Organisation existiert.
- Dual-Write-Perioden tragen das Risiko, dass die zwei Repräsentationen auseinanderdriften, und der Abgleich ist echte Arbeit.
- Zu viel zu extrahieren entfernt Flexibilität, die der Schwanz genuin braucht, und sie später wieder einzuführen ist schwerer, als sie unangetastet gelassen zu haben.
- Die endgültige Löschung der generischen Zeilen ist leicht aufzuschieben, und das Aufschieben bedeutet, dass die Organisation beide Modelle dauerhaft trägt.

## How It Could Be

Ein Produktkatalog speicherte jedes Attribut generisch: 380 Attributnamen, von denen die Nutzungsanalyse zeigte, dass 22 auf mehr als 95 Prozent der Produkte ausgefüllt waren und in fast jedem Suchfilter erschienen. Diese 22 — Name, Kategorie, Preis, Währung, Status, Abmessungen und eine Handvoll anderer — waren die tatsächliche Struktur des Katalogs. Das Team extrahierte sie in fünf Inkrementen über ein Quartal, wobei jede Gruppe dual-geschrieben und die zwei Pfade auf Produktions-Traffic verglichen wurden, bevor Leser umzogen. Die Produktsuchlatenz sank von einem Median von 1,9 Sekunden auf 90 Millisekunden, weil die Suche endlich Indizes nutzen konnte. Die verbleibenden 358 Attribute blieben generisch, was korrekt war: Sie waren produkttypspezifische Eigenschaften, die genuin variierten, und kein typisiertes Schema hätte sie berücksichtigt.

Der Typisierungsschritt war, wo die echte Arbeit sich als vorhanden herausstellte. Die Konvertierung des Preisattributs in eine numerische Spalte schlug bei 4.100 von 2,3 Millionen Produkten fehl. Die Untersuchung fand vier distinkte Ursachen: zwei Legacy-Import-Formate, eine Periode, in der ein Defekt die Währung ins Preisfeld geschrieben hatte, und ungefähr 900 Produkte, deren Preis genuin als Textbereich eingegeben worden war, weil das Geschäft keine andere Möglichkeit hatte, "Preis auf Anfrage" auszudrücken. Die ersten drei waren Datenfehler und wurden korrigiert. Der vierte war eine echte Anforderung, die niemand je modelliert hatte, und wurde zu einem separaten nullbaren Feld mit einem expliziten Flag — ein Domänenkonzept, das sechs Jahre lang in einer untypisierten Spalte versteckt gewesen war.
