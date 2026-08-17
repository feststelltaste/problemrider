---
title: Übermäßiger Einsatz von Entity-Attribute-Value
description: Geschäftsdaten werden als generische Attributzeilen statt als typisierte
  Spalten gespeichert, sodass die Datenbank die gespeicherten Daten nicht mehr erzwingen,
  indizieren oder erklären kann.
category:
- Database
- Architecture
- Code
related_problems:
- slug: database-schema-design-problems
  similarity: 0.6
- slug: custom-report-sprawl
  similarity: 0.6
- slug: schema-evolution-paralysis
  similarity: 0.6
- slug: unused-indexes
  similarity: 0.55
- slug: queries-that-prevent-index-usage
  similarity: 0.5
solutions:
- attribute-usage-analysis
- typed-schema-extraction
- data-modeling
- domain-driven-design
- evolutionary-database-design
- backward-compatible-schema-migrations
- parallel-run
- characterization-tests
- change-impact-analysis
- design-by-contract
- input-validation
- materialized-views
- cqrs
- index-lifecycle-management
- explicit-extension-points
- data-quality-checks
- variant-consolidation
- master-data-stewardship
layout: problem
lang: de
en_slug: entity-attribute-value-overuse
---

## Description

Übermäßiger Einsatz von Entity-Attribute-Value entsteht, wenn Daten mit einer bekannten, stabilen Struktur generisch gespeichert werden – eine Zeile pro Attribut, mit dem Attributnamen in einer Spalte und seinem Wert in einer anderen, meist als Text. Das Muster wird aus einem echten Grund eingeführt: Es erlaubt das Hinzufügen neuer Felder ohne Schemaänderung, was attraktiv ist, wenn Schemaänderungen langsam sind, wenn jeder Kunde unterschiedliche Felder braucht oder wenn die Anforderungen wirklich unbekannt sind. Was dabei aufgegeben wird, ist alles, was die Datenbank für einen tut. Typen, Constraints, Fremdschlüssel, Standardwerte und aussagekräftige Indizes werden alle unmöglich, weil die Datenbank nicht mehr sehen kann, was sie speichert. Diese Validierung verschwindet nicht; sie wandert in den Anwendungscode, wo sie inkonsistent oder gar nicht durchgesetzt wird, und die Daten häufen still Werte an, die das beabsichtigte Modell abgelehnt hätte.

## Indicators ⟡

- Eine Tabelle mit Spalten in etwa wie `entity_id`, `attribute_name` und `value`, wobei `value` eine Textspalte ist, die Zahlen, Daten und Flags enthält
- Das Abrufen eines Geschäftsobjekts erfordert einen Join oder Pivot über viele Zeilen, und die Abfrage wird generiert statt geschrieben
- Die Menge gültiger Attributnamen existiert nur im Anwendungscode, in einer Lookup-Tabelle, die niemand pflegt, oder in niemandes Kopf
- Reporting erfolgt gegen eine separate, flach gemachte Kopie der Daten, weil Reporting gegen das Live-Modell unpraktikabel ist
- Einfache Fragen – wie viele Kunden haben dieses Feld gesetzt, welche Werte nimmt es an – erfordern einen Spezialisten zur Beantwortung
- Typfehler tauchen zur Lesezeit auf, weit entfernt von der Stelle, an der der falsche Wert geschrieben wurde
- Dasselbe konzeptionelle Feld erscheint unter mehreren Attributnamen, die sich über die Jahre angehäuft haben

## Symptoms ▲

- [Langsame Datenbankabfragen](langsame-datenbankabfragen.md)
<br/>  Das Rekonstruieren eines Objekts erfordert das Verbinden oder Pivotieren vieler Zeilen, und das Filtern nach einem Attribut kann keinen Index so nutzen, wie es eine typisierte Spalte könnte.
- [Hohe Anzahl an Datenbankabfragen](hohe-anzahl-an-datenbankabfragen.md)
<br/>  Anwendungscode ruft häufig Attribute einzeln ab statt als ein Objekt, was Roundtrips für einen einzigen logischen Lesevorgang vervielfacht.
- [N+1-Abfrageproblem](n-plus-1-abfrageproblem.md)
<br/>  Das Laden einer Sammlung von Entitäten und dann ihrer Attribute pro Entität ist die natürliche Form von Code, der gegen dieses Modell geschrieben ist.
- [Schwer verständlicher Code](schwer-verstaendlicher-code.md)
<br/>  Das Domänenmodell ist im Schema unsichtbar, sodass das Verständnis, woraus eine Entität tatsächlich besteht, das Lesen des Codes erfordert, der sie zusammensetzt.
- [Testkomplexität](testkomplexitaet.md)
<br/>  Testdaten müssen Attribut für Attribut konstruiert werden, und das Fehlen von Constraints bedeutet, dass ungültige Kombinationen konstruierbar sind und darauf getestet werden müssen.
- [Erhöhte Fehleranzahl](erhoehte-fehleranzahl.md)
<br/>  Defekte, die ein typisiertes Schema unmöglich gemacht hätte – ein Datum in einem numerischen Feld, ein fehlendes Pflichtattribut – werden zu gewöhnlichen Laufzeitfehlern.
- [Imperative Datenabruflogik](imperative-datenabruflogik.md)
<br/>  Das Zusammensetzen von Objekten aus Attributzeilen drängt Datenzugriffslogik in prozeduralen Anwendungscode statt in deklarative Abfragen.

## Causes ▼

- [Übermäßige Anpassung](uebermaessige-anpassung.md)
<br/>  Wenn jeder Kunde unterschiedliche Felder braucht, erscheint ein generisches Modell als der einzige Weg, ein Schema pro Installation zu vermeiden.
- [Lähmung der Schema-Evolution](laehmung-der-schema-evolution.md)
<br/>  Wenn das Ändern des Schemas langsam, riskant ist oder ein Release erfordert, umgehen Entwickler es, indem sie neue Daten generisch speichern.
- [Häufige Anforderungsänderungen](haeufige-anforderungsaenderungen.md)
<br/>  Ein Modell, von dem erwartet wird, dass es sich ständig ändert, wird gebaut, um Änderungen aufzunehmen, und die generische Form ist die aufnahmefähigste und am wenigsten aussagekräftige verfügbare Option.
- [Angst vor Breaking Changes](angst-vor-breaking-changes.md)
<br/>  Das Hinzufügen einer Spalte fühlt sich riskant an in einem System, das niemand vollständig versteht, während das Hinzufügen einer Attributzeile keine bestehende Struktur berührt und sich daher sicher anfühlt.
- [Unzureichende Anforderungserhebung](unzureichende-anforderungserhebung.md)
<br/>  Wo die Domäne nie modelliert wurde, verschiebt ein generischer Container die Modellierung auf unbestimmte Zeit – und die Verschiebung wird dauerhaft.
- [Probleme im Datenbankschema-Design](probleme-im-datenbankschema-design.md)
<br/>  Schwache Schema-Design-Praxis macht das generische Modell attraktiv, weil es die Notwendigkeit beseitigt, überhaupt Design-Entscheidungen zu treffen.

## Detection Methods ○

- Suche nach Tabellen, deren Spaltennamen generisch sind – attribute, key, name, property – gepaart mit einer Wertspalte vom Typ Text
- Zählung der eindeutigen verwendeten Attributnamen und Vergleich damit, wie viele bei mehr als einem kleinen Prozentsatz der Entitäten gesetzt sind; ein kurzer Kopf und ein langer Schwanz ist die charakteristische Verteilung
- Prüfung, ob irgendein Constraint, Fremdschlüssel oder Check auf der Wertspalte existiert; typischerweise existiert keiner
- Messung, wie viele Zeilen die Datenbank liest, um ein Geschäftsobjekt zu erzeugen, und Vergleich damit, was ein typisiertes Modell erfordern würde
- Stichprobe der Wertspalte und Zählung der Einträge, die nicht als ihr beabsichtigter Typ geparst werden können
- Suche nach demselben Konzept, gespeichert unter mehreren Attributnamen, was darauf hindeutet, dass das Vokabular nie verwaltet wurde
- Prüfung, ob Reporting gegen dieses Modell oder gegen eine separate, flach gemachte Kopie läuft – eine Kopie ist ein starker Beweis dafür, dass das Modell nicht abfragbar ist

## Examples

Ein Versicherungspolicensystem speicherte Policendetails in einer Attributtabelle, weil Produktmanager Felder hinzufügen mussten, ohne auf ein Release zu warten. Nach neun Jahren enthielt die Tabelle 640 Millionen Zeilen über etwa 1.100 unterschiedliche Attributnamen. Das Laden einer Police erforderte das Zusammensetzen von 40 bis 80 Zeilen. Die Analyse der Attributnamen ergab, dass 31 davon bei mehr als 90 Prozent der Policen gesetzt waren – dies war die tatsächliche Policenstruktur, aus keinem verbleibenden Grund generisch gespeichert –, während 700 bei jeweils weniger als hundert Policen gesetzt waren und 140 seit 2019 nicht mehr geschrieben worden waren. Eine Frage des Regulators dazu, wie viele Policen ein bestimmtes Addendum hatten, brauchte vier Tage zur Beantwortung, weil die Wertspalte den Addendum-Code manchmal als Code, manchmal als Beschreibung und in einer Produktlinie als kommagetrennte Liste enthielt.

Das Fehlen von Constraints hatte ein leiseres Problem erzeugt. Weil die Wertspalte alles akzeptierte, hatte ein Defekt in einer Import-Routine elf Monate lang Daten in zwei unterschiedlichen Formaten geschrieben, bevor es jemand bemerkte, und die Bemerkung geschah, als eine Verlängerungsberechnung eine Police erzeugte, die im Jahr 20024 ablief. Eine typisierte Datumsspalte hätte das Schreiben in dem Moment abgelehnt, in dem es geschah, direkt neben dem Code, der es verursachte. Stattdessen waren die schlechten Daten über zwei Jahre an Datensätzen verteilt, und die Korrektur erforderte Überlegungen dazu, in welchem Format jeder Wert geschrieben worden war.
