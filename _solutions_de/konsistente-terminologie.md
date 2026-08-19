---
title: Konsistente Terminologie
description: Nutzung einheitlicher Begriffe in der gesamten Software.
category:
- Communication
- Requirements
quality_tactics_url: https://qualitytactics.de/en/usability/consistent-terminology/
problems:
- user-confusion
- poor-user-experience-ux-design
- inconsistent-behavior
- poor-naming-conventions
- user-frustration
- difficult-developer-onboarding
- knowledge-gaps
- poor-documentation
- inconsistent-naming-conventions
- information-fragmentation
- language-barriers
- poor-communication
- custom-report-sprawl
- master-data-ownership-gaps
layout: solution
lang: de
en_slug: consistent-terminology
related_solutions:
- slug: consistent-user-interface
  similarity: 0.85
- slug: ubiquitous-language
  similarity: 0.85
- slug: intuitive-navigation
  similarity: 0.8
- slug: plain-language
  similarity: 0.8
- slug: style-guide
  similarity: 0.8
- slug: user-centered-design
  similarity: 0.75
---

## Description

Konsistente Terminologie bildet jedes Domänenkonzept auf einen einzigen kanonischen Namen ab und setzt ihn über Schnittstelle, Code und Dokumentation hinweg durch, was die unterschiedlichen Wörter für dieselbe Sache ersetzt, die sich anhäufen, während verschiedene Teams über Jahre Features hinzufügen. Legacy-Systeme neigen besonders zu dieser Drift — „Kontonummer", „Konto-ID" und „Kundenreferenz" alle dasselbe Feld auf verschiedenen Bildschirmen bedeutend —, was genau die Art von Dateneingabefehlern und Verwirrung verursacht, die ein gemeinsames Glossar, konsistent durch Code-Review und UI-Updates angewendet, beseitigen soll.

## How to Apply ◆

> Legacy-Systeme häufen inkonsistente Terminologie an, während verschiedene Teams und Entwickler über die Jahre Features hinzufügen und unterschiedliche Wörter für dasselbe Konzept nutzen. Die Standardisierung von Begriffen verringert Verwirrung sowohl für Nutzer als auch Entwickler.

- Erstellen Sie ein Glossar von Domänenbegriffen, das jedes Konzept auf einen einzigen kanonischen Namen abbildet. Auditieren Sie die UI-Beschriftungen, Fehlermeldungen, Dokumentation und Code-Kommentare des Legacy-Systems, um zu identifizieren, wo dasselbe Konzept mit unterschiedlichen Namen bezeichnet wird.
- Etablieren Sie Namenskonventionen für neue Entwicklung, die auf das Glossar verweisen, und setzen Sie sie durch Code-Review durch. Jede neue UI-Beschriftung, jeder API-Feldname und jede Dokumentationsreferenz sollte den kanonischen Begriff nutzen.
- Priorisieren Sie Terminologiekorrekturen in den Bereichen der Anwendung mit dem höchsten Nutzertraffic und den meisten verwirrungsbezogenen Support-Tickets.
- Aktualisieren Sie Datenbankspaltennamen und API-Feldnamen schrittweise während Routinewartung. Nutzen Sie abwärtskompatible Aliase während der Übergangsperiode, um Integrationen nicht zu brechen.
- Beziehen Sie das Glossar in Onboarding-Material ein, sodass neue Teammitglieder und neue Nutzer die korrekte Terminologie von Anfang an lernen, statt die Inkonsistenzen zu erben.
- Koordinieren Sie Terminologieänderungen mit nutzerorientierter Dokumentation und Schulungsmaterial, um während des Übergangs nicht noch mehr Verwirrung zu schaffen.

## Tradeoffs ⇄

> Konsistente Terminologie verbessert das Verständnis über das gesamte System hinweg, erfordert aber koordinierten Aufwand, um verwurzelte Namensgebung zu ändern.

**Vorteile:**

- Verringert Nutzerverwirrung, verursacht durch unterschiedliche Beschriftungen für dasselbe Konzept in verschiedenen Teilen der Anwendung.
- Beschleunigt die Entwickler-Einarbeitung, weil neue Teammitglieder eine Reihe von Begriffen lernen können, statt mental zwischen mehreren Namensschemata zu übersetzen.
- Verbessert die Durchsuchbarkeit innerhalb der Codebasis und Dokumentation, wenn Konzepte einen einzigen kanonischen Namen haben.
- Verringert Support-Anfragen, verursacht durch Nutzer, die Beschriftungen missverstehen oder Daten in die falschen Felder eingeben.

**Kosten und Risiken:**

- Die Umbenennung etablierter Begriffe in einem Legacy-System kann langjährige Nutzer, die die alten Beschriftungen auswendig gelernt haben, vorübergehend verwirren, was Change-Management-Kommunikation erfordert.
- Das Ändern von Datenbankspaltennamen oder API-Feldern erfordert Migrationsaufwand und Koordination mit allen Konsumenten dieser Schnittstellen.
- Die Erreichung teamweiter Übereinstimmung über kanonische Begriffe kann zeitaufwendig sein, wenn verschiedene Gruppen starke Präferenzen für ihr etabliertes Vokabular haben.
- Teilweise Terminologieaktualisierungen schaffen eine Übergangsperiode, in der alte und neue Begriffe koexistieren, was Verwirrung potenziell erhöht, bevor sie sinkt.

## How It Could Be

> Terminologieinkonsistenz ist eines der häufigsten und am meisten übersehenen Usability-Probleme in Legacy-Systemen.

Ein Legacy-Banking-System nutzt „Kontonummer", „Konto-ID", „Kundenreferenz" und „Portfolio-Code" austauschbar über verschiedene Module hinweg, um dieselbe Kundenkontokennung zu bezeichnen. Neue Kassierer geben regelmäßig die falsche Kennung im falschen Feld ein, weil sie nicht erkennen können, welche Beschriftung welches Konzept meint. Das Team erstellt ein Domänenglossar mit Input von Fachanalysten und einigt sich darauf, dass „Kontonummer" der kanonische Begriff ist. Sie aktualisieren systematisch UI-Beschriftungen Modul für Modul während geplanter Wartungsfenster, beginnend mit den am stärksten genutzten kundenzugewandten Bildschirmen. Nach der Aktualisierung der drei am meisten genutzten Module sinken kassiererseitige Fehlerraten im Zusammenhang mit falscher Kontoidentifikation um die Hälfte, und das Schulungshandbuch schrumpft um mehrere Seiten, weil es die unterschiedlichen Begriffe nicht mehr erklären muss.
