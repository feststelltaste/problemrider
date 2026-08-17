---
title: Wildwuchs an individuellen Reports
description: Hunderte maßgeschneiderter Reports, Formulare und Extrakte häufen sich
  an, ohne Aufzeichnung, wer sie nutzt, sodass keiner mit Zuversicht geändert oder
  entfernt werden kann.
category:
- Business
- Database
- Process
related_problems:
- slug: excessive-customization
  similarity: 0.7
- slug: low-code-customization-sprawl
  similarity: 0.65
- slug: customization-outside-version-control
  similarity: 0.65
- slug: entity-attribute-value-overuse
  similarity: 0.6
- slug: authorization-role-explosion
  similarity: 0.6
- slug: reimplemented-standard-functionality
  similarity: 0.6
solutions:
- feature-usage-measurement
- variant-consolidation
- strategic-code-deletion
- attribute-usage-analysis
- consistent-terminology
- ubiquitous-language
- clear-ownership-model
- materialized-views
- data-strategy
- customization-cost-attribution
- master-data-stewardship
layout: problem
lang: de
en_slug: custom-report-sprawl
---

## Description

Wildwuchs an individuellen Reports entsteht, wenn sich die maßgeschneiderten Ausgaben eines kommerziell erworbenen Softwaresystems – Reports, Formulare, Extrakte, Dashboards, gedruckte Dokumente – über Jahre hinweg anhäufen, ohne dass jemals eines davon ausgemustert wird. Jedes wurde von jemandem angefragt, schnell gebaut und nie wieder überprüft. Weil Ausgaben billig hinzuzufügen und unsichtbar sind, wenn ungenutzt, wächst der Bestand monoton, bis er mehrere hundert Elemente enthält, von denen nur ein kleiner Bruchteil tatsächlich konsultiert wird. Die Kosten liegen nicht im Speicherplatz, sondern in der Kopplung: Jede dieser Ausgaben liest das Datenmodell direkt, sodass eine Schemaänderung, ein Upgrade oder eine Datenmigration alle berücksichtigen muss. Sie erzeugen außerdem widersprüchliche Antworten, weil dieselbe Geschäftszahl an elf Stellen von elf Personen über ein Jahrzehnt hinweg leicht unterschiedlich berechnet wurde.

## Indicators ⟡

- Der Report-Bestand hat Hunderte von Einträgen, und niemand kann die benennen, die wichtig sind
- Zwei Reports derselben Kennzahl weichen voneinander ab, und welcher korrekt ist, ist Ansichtssache
- Eine Schemaänderung wird geschätzt, indem zuerst festgestellt wird, welche Reports brechen würden, und das dauert Tage
- Reports tragen den Namen der Person, die sie angefragt hat, manchmal schon lange ausgeschieden
- Nutzer exportieren Report-Ausgaben in Tabellenkalkulationen und arbeiten dort weiter, was darauf hindeutet, dass der Report ihre Frage nicht beantwortet
- Kein Report wurde jemals außer Dienst gestellt, und es gibt keinen Prozess, durch den das geschehen würde

## Symptoms ▲

- [Doppelter Aufwand](doppelter-aufwand.md)
<br/>  Dieselbe Kennzahl wird wiederholt in unterschiedlichen Ausgaben berechnet, und jede Berechnung wird separat gepflegt.
- [Schattensysteme](schattensysteme.md)
<br/>  Wo Reports die tatsächliche Frage nicht beantworten, bauen Nutzer Tabellenkalkulationen daneben, die dann tragend und unsichtbar werden.
- [Lähmung der Schema-Evolution](laehmung-der-schema-evolution.md)
<br/>  Jede Datenmodelländerung muss eine unbekannte Anzahl von Ausgaben berücksichtigen, die die betroffenen Strukturen direkt lesen.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Der Bestand muss durch jedes Upgrade und jede Migration getragen werden, und seine Größe steht in keinem Zusammenhang mit dem Wert, den er liefert.
- [Testkomplexität](testkomplexitaet.md)
<br/>  Ein Upgrade regressionszutesten bedeutet, Ausgaben zu verifizieren, deren korrekte Ergebnisse niemand unabhängig angeben kann.
- [Unsichtbarkeit technischer Schulden](unsichtbarkeit-technischer-schulden.md)
<br/>  Reports werden nicht als Code betrachtet, sodass der Bestand in keiner technischen Bewertung des Systems erscheint.
- [Erhöhte manuelle Arbeit](erhoehte-manuelle-arbeit.md)
<br/>  Das Abgleichen widersprüchlicher Ausgaben wird zu einer wiederkehrenden manuellen Aufgabe in den Geschäftsfunktionen, die sie konsumieren.

## Causes ▼

- [Übermäßige Anpassung](uebermaessige-anpassung.md)
<br/>  Reports sind die billigste Form der Anpassung zum Anfragen und die am wenigsten wahrscheinlich abgelehnte, sodass sie sich am schnellsten anhäufen.
- [Unzureichende Anforderungserhebung](unzureichende-anforderungserhebung.md)
<br/>  Eine Anfrage für einen bestimmten Report wird wie angegeben erfüllt, statt untersucht zu werden, sodass ein Report, der die Frage fast beantwortet, neben einem gebaut wird, der es auch fast tut.
- [Übermäßiger Einsatz von Entity-Attribute-Value](uebermaessiger-einsatz-von-entity-attribute-value.md)
<br/>  Wo das Datenmodell nicht direkt abfragbar ist, erfordert jede Frage einen zweckgebundenen Extrakt statt einer Ad-hoc-Abfrage.
- [Fehlende Eigenverantwortung und Rechenschaftspflicht](fehlende-eigenverantwortung-und-rechenschaftspflicht.md)
<br/>  Niemand besitzt den Ausgabenbestand, sodass nichts eine Überprüfung auslöst und nichts jemals entfernt wird.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Den angefragten Report zu bauen ist schnell; festzustellen, ob ein bestehender das Bedürfnis bereits bedient, ist es nicht, also wird das übersprungen.

## Detection Methods ○

- Zählung der individuellen Ausgaben und, falls die Plattform Ausführungen protokolliert, Zählung, wie viele im letzten Jahr ausgeführt wurden
- Identifikation von Ausgaben, die nie ausgeführt wurden oder zuletzt vor Jahren
- Suche nach mehreren Ausgaben, die dieselbe Geschäftszahl berechnen, und Vergleich ihrer Definitionen
- Prüfung, ob eine Ausgabe einen Verantwortlichen, einen erklärten Zweck oder eine dokumentierte Definition der erzeugten Zahlen hat
- Messung, wie viel des letzten Upgrade-Regressionsaufwands in die Verifikation von Ausgaben floss
- Befragung einer Geschäftsfunktion, auf welche Ausgaben sie sich verlässt, und Vergleich ihrer Liste mit dem Bestand

## Examples

Eine Organisation, die eine Dokumenten- und Aktenverwaltungsplattform betreibt, hatte über zwölf Jahre 780 individuelle Ausgaben angehäuft. Ausführungsprotokollierung, für ein Quartal aktiviert, um die Frage zu beantworten, zeigte, dass 61 über 95 Prozent aller Ausführungen ausmachten, dass 430 in drei Monaten überhaupt nicht ausgeführt worden waren und dass 190 in den zwei Jahren, für die Protokolle aufbewahrt wurden, nicht ausgeführt worden waren. Das Team hatte eine geplante Datenmodelländerung auf vier Monate geschätzt, fast ausschließlich weil die Auswirkung auf Ausgaben unbekannt war. Mit den Nutzungsdaten wurde die Änderung gegen die 61 wichtigen Ausgaben abgegrenzt und in fünf Wochen abgeschlossen. Die 190 ruhenden Ausgaben wurden nach einer Ankündigungsfrist außer Dienst gestellt, während der zwei beansprucht wurden – beide jährlich, beide legitim, und beide nun so erfasst.

Das Inkonsistenzproblem war schwieriger und aufschlussreicher. Vier Ausgaben meldeten monatliches verarbeitetes Volumen, und sie wichen um bis zu elf Prozent voneinander ab. Die Untersuchung ergab, dass jede zu einer anderen Zeit für eine andere Abteilung gebaut worden war, und jede nutzte eine vertretbare, aber unterschiedliche Definition dessen, was als verarbeitet zählte und wann. Keine war falsch. Die Organisation hatte jahrelang Meetings abgehalten, in denen Abteilungen inkompatible Zahlen präsentierten, und die Ursache war immer als Datenqualitätsproblem angenommen worden, statt als vier undokumentierte Definitionen eines Begriffs, dem nie jemand zugestimmt hatte.
