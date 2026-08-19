---
title: Architecture Decision Records (ADR)
description: Dokumentation wichtiger architektonischer Entscheidungen und ihrer Begründungen.
category:
- Architecture
- Communication
quality_tactics_url: https://qualitytactics.de/en/maintainability/architecture-decision-records-adr/
problems:
- accumulated-decision-debt
- decision-avoidance
- decision-paralysis
- delayed-decision-making
- implicit-knowledge
- tacit-knowledge
- information-decay
- poor-documentation
- knowledge-gaps
- incomplete-knowledge
- stagnant-architecture
- history-of-failed-changes
- analysis-paralysis
- cv-driven-development
- duplicated-research-effort
- premature-technology-introduction
- team-churn-impact
- conflicting-reviewer-opinions
- implementation-partner-dependency
layout: solution
lang: de
en_slug: architecture-decision-records
related_solutions:
- slug: architecture-documentation
  similarity: 0.85
- slug: documentation-as-code
  similarity: 0.8
- slug: architecture-reviews
  similarity: 0.8
- slug: knowledge-sharing-practices
  similarity: 0.8
- slug: architecture-roadmap
  similarity: 0.8
- slug: architecture-review-board
  similarity: 0.8
---

## Description

Ein Architecture Decision Record erfasst eine spezifische Entscheidung — was gewählt wurde, welche Alternativen erwogen wurden und warum — als kurzes, dauerhaftes Dokument, das zusammen mit dem Code gespeichert wird, den es betrifft. Legacy-Systeme häufen Entscheidungen an, deren Begründung sich niemand mehr merkt, was jeden neuen Betreuer zwingt, die Absicht allein aus dem Code zurückzuentwickeln oder bereits geklärte Fragen von Grund auf neu zu verhandeln. Das rückwirkende Schreiben von ADRs für die folgenreichsten vergangenen Entscheidungen und proaktiv für jede zukünftige Entscheidung verwandelt diese undokumentierte Geschichte in ein nachvollziehbares Verzeichnis, das nicht nur erklärt, was das System tut, sondern warum es so aussieht, wie es aussieht.

## How to Apply ◆

> In Legacy-Systemen zwingt das Fehlen von Entscheidungsbegründungen jedes Teammitglied dazu, die Absicht aus dem Code zurückzuentwickeln, sodass ADRs am besten funktionieren, wenn sie rückwirkend für die wirkungsvollsten vergangenen Entscheidungen und proaktiv für alle zukünftigen eingeführt werden.

- Beginnen Sie damit, rückwirkende ADRs für die Entscheidungen zu schreiben, die heute die meiste Verwirrung verursachen: warum das System eine bestimmte Datenbank nutzt, warum ein bestimmtes Integrationsmuster gewählt wurde oder warum eine Komponente so aufgeteilt wurde, wie sie es wurde. Kombinieren Sie diese Arbeit mit den Menschen, die sich noch an die ursprüngliche Begründung erinnern.
- Speichern Sie ADRs im Quellcode-Repository zusammen mit dem Legacy-Code, nicht in einem Wiki oder geteilten Laufwerk, das separat veraltet. Dies hält Entscheidungen über dasselbe Tooling auffindbar, das Entwickler bereits nutzen.
- Markieren Sie Entscheidungen, die inzwischen als problematisch bekannt sind, als „Deprecated", statt sie zu löschen. In Legacy-Kontexten ist das Verständnis, warum eine schlechte Entscheidung getroffen wurde, oft so wertvoll wie die Entscheidung selbst.
- Wenn eine Modernisierungsinitiative eine alte Wahl überarbeitet — Migration von einer monolithischen Datenbank, Ersatz eines Nachrichtenprotokolls, Aufteilung eines Moduls — schreiben Sie ein neues ADR, das explizit auf das ursprüngliche verweist und erklärt, was sich geändert hat. Dies erzeugt ein nachvollziehbares Verzeichnis davon, wie sich die Architektur entwickelt hat.
- Nutzen Sie ADRs als Gatekeeping-Werkzeug während der Modernisierung: Keine architektonische Änderung am Legacy-System wird genehmigt, es sei denn, ein ADR wird zuerst entworfen, überprüft und gemergt. Dies verhindert, dass uninformierte Änderungen neue Schichten versteckter Schulden hinzufügen.
- Verweisen Sie in Code-Kommentaren direkt auf ADR-Nummern, wo immer eine Entscheidung eine sichtbare Manifestation hat. Legacy-Codebasen sind voller überraschender Konstrukte; ein Kommentar, der auf das ADR verweist, das die Einschränkung erklärt, ist weit dauerhafter als Stammeswissen.
- Übernehmen Sie ein leichtgewichtiges Format, um die Einstiegshürde zu senken. In Legacy-Teams mit hoher Arbeitslast wird ein Y-Statement oder ein kurzes tabellenzeilenbasiertes Entscheidungsprotokoll wahrscheinlich konsistenter genutzt als ein vollständiges Fünf-Abschnitte-Dokument.
- Integrieren Sie ADR-Review in den Pull-Request-Prozess für jede Änderung, die Kernkomponenten des Legacy-Systems betrifft, sodass Senior-Ingenieure markieren können, wenn eine vorgeschlagene Änderung einer zuvor dokumentierten Einschränkung widerspricht.

## Tradeoffs ⇄

> ADRs erlegen eine Schreibdisziplin auf, gegen die sich Legacy-Teams oft wehren, aber die langfristigen Kosten undokumentierter Entscheidungen in alternden Systemen übersteigen bei Weitem den Aufwand, sie zu erfassen.

**Vorteile:**

- Eliminiert die wiederholte Neubewertung geklärter Entscheidungen, ein Muster, das Legacy-Teams plagt, während Schlüsselpersonal geht und institutionelles Gedächtnis erodiert.
- Gibt neuen Betreuern einen strukturierten Einstiegspunkt, um zu verstehen, warum das System so aussieht, wie es aussieht, und verringert dramatisch die Onboarding-Zeit für komplexe Legacy-Codebasen.
- Verhindert, dass gut gemeinte Modernisierungsänderungen versehentlich Einschränkungen verletzen, die das ursprüngliche Design geprägt haben — zum Beispiel das Entfernen eines Workarounds, der eine bekannte Einschränkung eines externen Systems kompensiert.
- Erzeugt einen Prüfpfad, der Compliance-Anforderungen in regulierten Branchen erfüllt, wo Legacy-Systeme oft sensible Daten unter langjährigen rechtlichen Verpflichtungen handhaben.
- Verbessert die Qualität architektonischer Diskussionen während der Modernisierung, indem Debatten von konkurrierenden Meinungen zu dokumentierten Abwägungen verschoben werden.

**Kosten und Risiken:**

- Rückwirkendes ADR-Schreiben erfordert das Extrahieren von Begründungen von Menschen, deren Erinnerungen an vor Jahren getroffene Entscheidungen unvollständig oder inkonsistent sein können, was zu ADRs führt, die beste Vermutungen statt tatsächlicher Begründungen dokumentieren.
- Ohne konsistente Durchsetzung wird das ADR-Verzeichnis unvollständig: Aktuelle, undokumentierte Entscheidungen erzeugen dieselben Wissenslücken, die die Praxis verhindern sollte.
- Übermäßig breite ADRs, die versuchen, Jahre angehäufter Entscheidungen in einem kurzen Ausbruch zu erfassen, neigen dazu, oberflächlich zu sein und aufzuzeichnen, was entschieden wurde, ohne den vollen Kontext des Warum — was der wertvollste Teil ist.
- Teams unter dem Druck, gleichzeitig Legacy-Wartung und Modernisierung zu liefern, könnten ADR-Schreiben als Overhead behandeln und es deprioritisieren, bis die Praxis verblasst.
- ADRs, die Einschränkungen dokumentieren, die inzwischen irrelevant geworden sind, können neue Entwickler in die Irre führen, wenn Status nicht aktuell gehalten werden, was möglicherweise dazu führt, dass sie unnötige Komplexität bewahren.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie ADRs die spezifischen Wissensprobleme angehen, die sich in langlebigen Legacy-Systemen anhäufen.

Ein Finanzdienstleistungsunternehmen, das ein Anfang der 2000er-Jahre gebautes Zahlungsabwicklungssystem betreibt, plante eine Migration von synchronen REST-Aufrufen zu einer asynchronen Nachrichtenwarteschlange. Als der Architekt die Änderung vorschlug, erinnerte sich ein Senior-Ingenieur daran, dass synchrone Aufrufe bewusst gewählt worden waren, weil die nachgelagerte Banking-API zu der Zeit keine Idempotenz garantierte und asynchrone Wiederholungen das Risiko doppelter Transaktionen bargen. Keine Dokumentation dieser Einschränkung existierte. Das Team verbrachte zwei Wochen damit zu untersuchen, ob die Einschränkung noch galt, bevor es sein erstes ADR schrieb, das die ursprüngliche Begründung erfasste. Das neue ADR bestätigte, dass die Banking-API inzwischen Idempotenzschlüssel hinzugefügt hatte, was die Migration sicher machte — und die beiden ADRs erzählten zusammen eine vollständige Geschichte der Entwicklung der Entscheidung.

Eine Regierungsbehörde, die ein Legacy-Genehmigungsmanagementsystem pflegte, kämpfte mit hoher Fluktuation unter Entwicklern, die mit der ungewöhnlichen Zwei-Datenbank-Architektur des Systems vertraut waren. Das Design nutzte eine normalisierte relationale Datenbank für Schreibvorgänge und eine denormalisierte Flat-File-Struktur für Lesevorgänge, ein Muster, das jeden neuen Mitarbeiter verwirrte, der darauf stieß. Nach der Einführung von ADRs schrieb das Team ein rückwirkendes Protokoll, das dokumentierte, dass die Flat-File-Struktur durch eine regulatorische Berichtsanforderung vorgeschrieben worden war, die 2009 Sub-Sekunden-Abfragezeiten benötigte, bevor die aktuelle Datenbankinfrastruktur verfügbar war. Nachfolgende Mitarbeiter konnten das ADR lesen und innerhalb von Minuten verstehen, warum die Architektur so aussah, wie sie aussah, statt Wochen damit zu verbringen, es durch Code-Lektüre zu entdecken.

Ein Logistikunternehmen, das eine Strangler-Fig-Migration seines Legacy-Auftragsmanagementsystems durchführte, stellte fest, dass verschiedene Teams inkonsistente Technologieentscheidungen für die Ersatzservices trafen — ein Team wählte Kafka für Messaging, ein anderes wählte RabbitMQ, ein drittes erwog einen Datenbank-Polling-Ansatz. Das Fehlen eines dokumentierten Standards erlaubte es jedem Team, dasselbe Problem unabhängig von Grund auf zu bewerten. Nachdem Wochen an doppeltem Bewertungsaufwand verloren gegangen waren, führte der Modernisierungsleiter ADRs als Entscheidungsautoritätsmechanismus ein. Das erste teamübergreifende ADR dokumentierte die Wahl von Kafka als Standard-Message-Broker, den Kontext dahinter und die erwogenen Alternativen. Nachfolgende Teams konnten die Entscheidung übernehmen oder formal anfechten, statt stillschweigend abzuweichen.
