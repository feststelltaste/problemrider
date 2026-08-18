---
title: Schattensysteme
description: Alternative Lösungen, die außerhalb offizieller Kanäle entwickelt werden,
  untergraben Standardisierung und schaffen versteckte Abhängigkeiten.
category:
- Management
- Process
related_problems:
- slug: hidden-dependencies
  similarity: 0.6
- slug: implicit-knowledge
  similarity: 0.6
- slug: implementation-partner-dependency
  similarity: 0.55
- slug: information-fragmentation
  similarity: 0.55
- slug: vendor-dependency-entrapment
  similarity: 0.55
- slug: technology-stack-fragmentation
  similarity: 0.55
solutions:
- user-centered-design
- cognitive-load-minimization
- consistent-user-interface
- custom-views
- customizable-user-interface
- intuitive-navigation
- search-function
- usability-tests
- master-data-stewardship
layout: problem
lang: de
en_slug: shadow-systems
---

## Description

Schattensysteme sind informelle, inoffizielle Anwendungen, Werkzeuge oder Prozesse, die Teams erstellen, um Beschränkungen in offiziellen Systemen zu umgehen. Obwohl sie oft aus legitimen Bedürfnissen und guten Absichten entstehen, operieren diese Systeme außerhalb organisatorischer Aufsicht, ohne ordentliche Dokumentation, Sicherheitskontrollen und Wartungsprozeduren. Sie schaffen versteckte Abhängigkeiten, Compliance-Risiken und potenzielle Ausfallpunkte, auf die die Organisation nicht vorbereitet ist.

## Indicators ⟡

- Teams nutzen selbstgebaute Werkzeuge oder Tabellenkalkulationen statt offizieller Unternehmenssysteme
- Daten werden an mehreren Orten mit manueller Synchronisation gepflegt
- Geschäftsprozesse hängen von individuell gepflegten Anwendungen oder Skripten ab
- Die IT-Abteilung ist sich kritischer Geschäftswerkzeuge, die von Teams genutzt werden, nicht bewusst
- Offizielle Berichte stimmen nicht mit dem überein, was Teams tatsächlich für Entscheidungsfindung nutzen

## Symptoms ▲

- [Versteckte Abhängigkeiten](versteckte-abhaengigkeiten.md)
<br/>  Schattensysteme schaffen undokumentierte Abhängigkeiten, die offizielle Systemkarten und Architekturdiagramme nicht erfassen.
- [Datenschutzrisiko](datenschutzrisiko.md)
<br/>  Schattensysteme speichern sensible Daten oft außerhalb organisatorischer Sicherheitskontrollen, was Compliance- und Datenschutzrisiken schafft.
- [Informationsfragmentierung](informationsfragmentierung.md)
<br/>  Kritische Geschäftsdaten werden zwischen offiziellen Systemen und Schattenalternativen verstreut, was es schwierig macht, eine einzige Quelle der Wahrheit zu pflegen.
- [Single Points of Failure](single-points-of-failure.md)
<br/>  Schattensysteme werden typischerweise von einer einzigen Person gepflegt und laufen auf persönlicher Infrastruktur, was kritische Single Points of Failure schafft.
- [Fragmentierung des Technologie-Stacks](fragmentierung-des-technologie-stacks.md)
<br/>  Jedes Schattensystem führt seine eigenen Technologiewahlen ein, was die gesamte Technologielandschaft fragmentiert.

## Causes ▼

- [Schlechtes Nutzererlebnis (UX-Design)](schlechtes-nutzererlebnis-ux-design.md)
<br/>  Offizielle Systeme, die schwierig oder frustrierend zu nutzen sind, treiben Teams dazu, ihre eigenen alternativen Lösungen zu erstellen.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Wenn offizielle Systeme langsam benötigte Fähigkeiten liefern, bauen Teams ihre eigenen Werkzeuge, um Lücken zu füllen, statt zu warten.
- [Ineffiziente Prozesse](ineffiziente-prozesse.md)
<br/>  Bürokratische Prozesse zur Anfrage von Änderungen an offiziellen Systemen drängen Teams dazu, den Prozess ganz mit Schattenlösungen zu umgehen.
- [Funktionslücken](funktionsluecken.md)
<br/>  Fehlende Funktionalität in offiziellen Systemen schafft legitime Bedürfnisse, die Teams durch den Bau inoffizieller Alternativen angehen.
- [Frustration der Stakeholder](frustration-der-stakeholder.md)
<br/>  Frustrierte Stakeholder mit offiziellen Systemen treiben sie dazu, Schattensysteme als Alternativen zu erstellen oder zu unterstützen.

## Detection Methods ○

- **System-Discovery-Audits:** Regelmäßige Umfragen zur Identifikation inoffizieller Werkzeuge und Systeme
- **Datenfluss-Analyse:** Abbildung, wo Geschäftsdaten tatsächlich fließen im Vergleich zu offiziellen Kanälen
- **Zugriffsprotokoll-Überprüfung:** Analyse, welche Systeme und Werkzeuge Mitarbeiter tatsächlich nutzen
- **Geschäftsprozess-Interviews:** Befragung von Teams zu ihren tatsächlichen Arbeitsprozessen
- **Sicherheitsschwachstellenbewertungen:** Scannen nach unbefugten Anwendungen und Datenspeichern

## Examples

Ein Vertriebsteam erstellt eine aufwendige Excel-Tabellenkalkulation mit Makros zur Verfolgung von Leads, weil das offizielle CRM-System zu langsam ist und wichtige benötigte Felder fehlen. Die Tabellenkalkulation wird zur primären Wahrheitsquelle für Umsatzprognosen, wird aber von einer Person gepflegt, die nicht dokumentiert hat, wie sie funktioniert. Wenn diese Person in den Urlaub geht, kann das Vertriebsteam Prognosen nicht aktualisieren, und das Management trifft Entscheidungen basierend auf veralteten Informationen. Die Tabellenkalkulation enthält auch Kundendaten, die nicht gemäß Unternehmensrichtlinien gesichert oder geschützt werden. Ein weiteres Beispiel betrifft ein Entwicklungsteam, das ein individuelles Dashboard zur Überwachung der Anwendungsperformance baut, weil die offiziellen Monitoring-Werkzeuge nicht die spezifischen Metriken liefern, die sie benötigen. Das Dashboard wird kritisch für die Vorfallsreaktion, läuft aber auf dem persönlichen Cloud-Konto eines Entwicklers und nutzt API-Schlüssel, die ohne Vorankündigung ablaufen. Wenn das System während eines Produktionsausfalls fehlschlägt, verliert das Team gerade dann die Sichtbarkeit auf die Systemgesundheit, wenn sie sie am meisten benötigen.
