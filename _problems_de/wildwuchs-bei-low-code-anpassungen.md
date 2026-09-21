---
title: Wildwuchs bei Low-Code-Anpassungen
description: Geschäftslogik häuft sich im eigenen Scripting- und Workflow-Werkzeug
  einer Plattform an, wo sie Tests, Reviews und allen anderen Entwicklungspraktiken
  entgeht.
category:
- Code
- Process
- Architecture
related_problems:
- slug: customization-outside-version-control
  similarity: 0.65
- slug: custom-report-sprawl
  similarity: 0.65
- slug: excessive-customization
  similarity: 0.65
- slug: legacy-business-logic-extraction-difficulty
  similarity: 0.6
- slug: accumulation-of-workarounds
  similarity: 0.6
- slug: reimplemented-standard-functionality
  similarity: 0.6
solutions:
- customization-under-version-control
- duplication-detection
- code-review-guidelines
- quality-ratchet
- automated-tests
- technical-debt-assessment
- debt-classification
- strategic-code-deletion
- clear-ownership-model
- internal-technical-coaching
- attribute-usage-analysis
layout: problem
lang: de
en_slug: low-code-customization-sprawl
---

## Description

Wildwuchs bei Low-Code-Anpassungen tritt auf, wenn sich erhebliche Geschäftslogik im eingebauten Scripting, Workflow-Designer, Regelwerk oder in der Formularlogik einer kommerziellen Softwareplattform ansammelt. Jedes einzelne Stück ist klein und wurde schnell von jemandem erstellt, der nicht unbedingt Entwickler war – das ist genau der Zweck des Mechanismus. Was sich ansammelt, ist eine zweite Codebasis, die von jeder Praxis ausgenommen ist, die auf die erste angewendet wird: keine Tests, kein Review, keine statische Analyse, kein Refactoring, häufig keine Versionskontrolle und keine Möglichkeit, sie zu durchsuchen. Nach ein paar Jahren enthält die Plattform Tausende kleiner Logikfragmente, deren Wechselwirkungen niemand nachvollziehen kann, und das Verhalten des Systems wird eher durch diese Ansammlung bestimmt als durch das, was der Anbieter ausgeliefert oder das Entwicklungsteam geschrieben hat.

## Indicators ⟡

- Niemand kann sagen, wie viele Skripte, Regeln oder Workflow-Schritte existieren, ohne sie zu exportieren und zu zählen
- Eine Frage danach, warum das System etwas getan hat, erfordert das Nachverfolgen mehrerer Workflow-Definitionen durch Klicken
- Änderungen werden von Personen außerhalb des Entwicklungsteams vorgenommen, ohne Review, direkt in einer laufenden Umgebung
- Dieselbe Berechnung erscheint an mehreren Stellen, weil die Suche nach einer bestehenden unpraktikabel ist
- Debugging bedeutet, Ausgaben zu einem Skript hinzuzufügen und den Prozess erneut auszuführen, weil keine andere Instrumentierung existiert
- Fragmente referenzieren Felder, Zustände oder Integrationen, die nicht mehr existieren, und nichts erkennt dies
- Logik, geschrieben von jemandem, der gegangen ist, wird unangetastet beibehalten, weil niemand es wagt, sie zu entfernen

## Symptoms ▲

- [Schwer testbarer Code](schwer-testbarer-code.md)
<br/>  Plattform-Scripting kann typischerweise nicht außerhalb der Plattform ausgeführt werden, sodass Verifikation bedeutet, den gesamten Prozess auszuführen und das Ergebnis zu prüfen.
- [Schwer verständlicher Code](schwer-verstaendlicher-code.md)
<br/>  Verhalten ist über viele kleine Fragmente in visueller oder eingebetteter Form verteilt, die nicht linear gelesen werden kann.
- [Code-Duplizierung](code-duplizierung.md)
<br/>  Ohne übergreifende Suche ist der günstigste Weg, ein Verhalten zu erhalten, es neu zu erstellen, sodass sich dieselbe Logik an vielen Stellen ansammelt.
- [Erhöhte Fehleranzahl](erhoehte-fehleranzahl.md)
<br/>  Logik, die ungetestet, ungereviewt und von Nicht-Spezialisten geschrieben ist, erzeugt Defekte in einer Rate, die die auf Anwendungscode angewendeten Praktiken eigentlich verhindern sollen.
- [Langsame Vorfallslösung](langsame-vorfallsloesung.md)
<br/>  Die Nachverfolgung eines unerwarteten Ergebnisses durch interagierende Fragmente ist langsam, und es gibt normalerweise keine Ausführungsspur, mit der man arbeiten kann.
- [Unsichtbarkeit technischer Schulden](unsichtbarkeit-technischer-schulden.md)
<br/>  Diese zweite Codebasis erscheint in keiner Metrik, keinem Review und keiner Schuldenbewertung, sodass ihr Gewicht vollständig unberücksichtigt bleibt.
- [Wissenssilos](wissenssilos.md)
<br/>  Jedes Fragment wird von demjenigen verstanden, der es erstellt hat, und die Erstellung war schnell genug, dass niemand daran dachte, festzuhalten, warum.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Tote Referenzen, überholte Regeln und aufgegebene Workflows häufen sich unbegrenzt an, weil kein Prozess sie jemals entfernt.

## Causes ▼

- [Anpassungen außerhalb der Versionskontrolle](anpassungen-ausserhalb-der-versionskontrolle.md)
<br/>  Wo die Fragmente in der Plattformdatenbank statt in Dateien leben, kann keine der Praktiken, die auf Dateien beruhen, auf sie angewendet werden.
- [Übermäßige Anpassung](uebermaessige-anpassung.md)
<br/>  Ein stetiger Strom von Anpassungsanfragen trifft auf einen Mechanismus, der darauf ausgelegt ist, sie schnell zu erfüllen, und das Volumen häuft sich schneller an, als jemand es reviewt.
- [Mangel an Legacy-Fachkräften](mangel-an-legacy-fachkraeften.md)
<br/>  Die Personen, die in der Plattform bauen, sind häufig Administratoren oder Analysten statt Entwickler, und die Entwicklungspraktiken waren nie Teil ihrer Rolle.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Der Mechanismus existiert genau, um schnell zu liefern, und die Geschwindigkeit wird sofort realisiert, während die Ansammlung erst Jahre später spürbar wird.
- [Fehlende Eigenverantwortung und Rechenschaftspflicht](fehlende-eigenverantwortung-und-rechenschaftspflicht.md)
<br/>  Plattformlogik gehört oft formal keinem Team, angesiedelt zwischen der Geschäftsfunktion, die sie angefordert hat, und der Entwicklungsgruppe, die sie nicht gebaut hat.

## Detection Methods ○

- Export der Skripte, Regeln und Workflow-Definitionen der Plattform und Zählung, dann Messung des Trends über die letzten zwei Jahre
- Durchsuchen des exportierten Inhalts nach Referenzen auf Felder, Zustände oder Integrationen, die nicht mehr existieren
- Fragen, wie eine bestimmte Geschäftsregel implementiert ist, und Zeitmessung, wie lange es dauert, eine vollständige Antwort zu erzeugen
- Prüfung, welcher Anteil der Fragmente einen Test, einen Review-Nachweis oder einen benannten Eigentümer hat
- Suche nach derselben Berechnung, die an mehr als einer Stelle implementiert ist, was die charakteristische Signatur unsuchbarer Logik ist
- Identifikation von Fragmenten, die vor mehr als drei Jahren zuletzt geändert wurden und deren Autor die Organisation verlassen hat

## Examples

Eine IT-Service-Management-Plattform hatte über sieben Jahre etwa 1.400 skriptbasierte Verhaltensweisen und 90 Workflows angesammelt, erstellt von einer Mischung aus Administratoren, einer Partnerberatung und zwei Entwicklern. Als das Ticket-Routing eine Kategorie von Anfragen falsch zuwies, dauerte die Untersuchung neun Tage. Die Ursache erwies sich als zwei Workflow-Regeln, die nie interagieren sollten: eine fügte unter einer 2021 eingeführten Bedingung ein Tag hinzu, und eine andere, geschrieben 2019 von jemandem, der nicht mehr in der Organisation ist, routete basierend auf der Anwesenheit dieses Tags. Keine der Regeln war für sich genommen falsch. Es hatte nie einen Moment gegeben, in dem irgendjemand beide hätte sehen können.

Der Export offenbarte ein zweites, leiseres Ergebnis. Von den 1.400 skriptbasierten Verhaltensweisen referenzierten 310 ein Feld, einen Zustand oder einen Integrationsendpunkt, der nicht mehr existierte, und waren daher entweder tot oder scheiterten still. Niemand hatte es gewusst, weil die Plattform keine Fehler für eine Regel meldete, deren Bedingung niemals zutreffen konnte. Die Anwendungscodebasis der Organisation hatte statische Analyse, Code-Review und eine Testsuite; die Plattform, die ihre Incident-, Change- und Request-Prozesse hielt, hatte keine der drei und hatte mehr Logik angesammelt als die Anwendung.
