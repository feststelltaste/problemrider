---
title: Komplexes Domänenmodell
description: Die in der Software abgebildete Geschäftsdomäne ist von Natur aus
  komplex, was das System schwer verständlich und schwer korrekt zu implementieren
  macht.
category:
- Architecture
- Business
related_problems:
- slug: poor-domain-model
  similarity: 0.75
- slug: complex-implementation-paths
  similarity: 0.6
- slug: legacy-business-logic-extraction-difficulty
  similarity: 0.6
- slug: difficult-to-understand-code
  similarity: 0.55
- slug: complex-and-obscure-logic
  similarity: 0.55
- slug: complex-deployment-process
  similarity: 0.55
solutions:
- modularization-and-bounded-contexts
- bounded-contexts
- data-modeling
- graph-databases
- domain-aligned-architecture
- domain-driven-design
- domain-modeling
- domain-immersion
layout: problem
lang: de
en_slug: complex-domain-model
---

## Description

Ein komplexes Domänenmodell entsteht, wenn die Geschäftsdomäne, die das Softwaresystem abbilden muss, verwickelte Regeln, Beziehungen und Konzepte enthält, die schwer zu verstehen und korrekt zu implementieren sind. Diese Komplexität kann aus regulatorischen Anforderungen, Legacy-Geschäftsprozessen oder von Natur aus komplexen Problemdomänen wie Finanzhandel, Gesundheitswesen oder wissenschaftlichem Rechnen entstehen. Die Herausforderung ist nicht nur technischer Natur, sondern beinhaltet auch das Verständnis und die genaue Abbildung komplexer Geschäftslogik im Code.

## Indicators ⟡

- Fachexperten haben Schwierigkeiten, Domänenregeln Entwicklern klar zu erklären
- Anforderungsdokumente sind umfangreich und enthalten zahlreiche Sonderfälle und Ausnahmen
- Das Systemverhalten variiert erheblich je nach Kontext, Zeit oder regulatorischen Änderungen
- Mehrere Stakeholder haben unterschiedliche Interpretationen derselben Geschäftsregeln
- Domänenkonzepte erfordern umfangreiches Hintergrundwissen zum Verständnis

## Symptoms ▲

- [Kognitive Überlastung](kognitive-ueberlastung.md)
<br/>  Entwickler müssen umfangreiches Domänenwissen im Arbeitsgedächtnis behalten, um selbst einfache Features korrekt umzusetzen.
- [Komplexe und unklare Logik](komplexe-und-unklare-logik.md)
<br/>  Verwickelte Geschäftsregeln mit zahlreichen Sonderfällen übersetzen sich in verworrenen Code, der schwer zu verstehen ist.
- [Wissenslücken](wissensluecken.md)
<br/>  Die inhärente Komplexität der Domäne erschwert es Entwicklern, Geschäftsregeln vollständig zu verstehen, was anhaltende Wissenslücken schafft.
- [Hohe Rate an neu eingeführten Fehlern](hohe-rate-an-neu-eingefuehrten-fehlern.md)
<br/>  Das Missverstehen komplexer Domänenregeln führt zu häufigen Implementierungsfehlern und neuen Defekten.
- [Verlängerte Rechercheszeit](verlaengerte-recherchezeit.md)
<br/>  Entwickler verbringen erhebliche Zeit damit, komplexe Domänenkonzepte zu recherchieren und zu verstehen, bevor sie Features umsetzen können.
- [Schwieriges Onboarding neuer Entwickler](schwieriges-onboarding-neuer-entwickler.md)
<br/>  Neue Teammitglieder benötigen umfangreiche Zeit, um die komplexe Domäne zu erlernen, bevor sie effektiv beitragen können.

## Causes ▼

- [Kommunikationslücke zwischen Stakeholdern und Entwicklern](kommunikationsluecke-zwischen-stakeholdern-und-entwicklern.md)
<br/>  Wenn Fachexperten Domänenregeln Entwicklern nicht klar erklären können, verstärkt sich die Komplexität in der Implementierung.
- [Schlechtes Domänenmodell](schlechtes-domaenenmodell.md)
<br/>  Ein schlecht entworfenes Domänenmodell scheitert daran, inhärente Geschäftskomplexität ordentlich zu strukturieren, was sie noch schwerer beherrschbar macht.
- [Anforderungsmehrdeutigkeit](anforderungsmehrdeutigkeit.md)
<br/>  Mehrdeutige Anforderungen rund um komplexe Domänenkonzepte führen zu mehreren Interpretationen und fehlerhaften Implementierungen.

## Detection Methods ○

- **Domänenkomplexitätsanalyse:** Bewertung der Anzahl von Geschäftsregeln, Ausnahmen und Sonderfällen in Anforderungen
- **Konsistenz von Stakeholder-Interviews:** Messung, wie konsistent unterschiedliche Stakeholder dieselben Domänenkonzepte erklären
- **Tracking der Implementierungszeit:** Beobachtung, wie lange die Umsetzung von Features im Verhältnis zu ihrer scheinbaren Einfachheit dauert
- **Fehlermuster-Analyse:** Analyse, ob Fehler typischerweise mit Missverständnissen der Geschäftslogik zusammenhängen
- **Dokumentationsumfang:** Bewertung des Umfangs der Dokumentation, die zur Erklärung von Domänenkonzepten nötig ist

## Examples

Ein Krankenversicherungssystem muss Hunderte unterschiedlicher Plantypen verarbeiten, jeder mit eigenen Deckungsregeln, Selbstbehaltsstrukturen, Zuzahlungsanforderungen und Netzwerkeinschränkungen. Das System muss außerdem staatliche und bundesweite Vorschriften einhalten, die geografisch variieren und sich häufig ändern. Eine einfache Anspruchsbearbeitungsanfrage beinhaltet die Prüfung der Mitgliedsberechtigung, der Plandeckung, des Netzwerkstatus des Anbieters, der Anforderungen an vorherige Genehmigungen, der Koordination von Leistungen mit anderen Versicherern und die Anwendung verschiedener Kostenbeteiligungsregeln. Die Geschäftsregeln sind so komplex, dass selbst Versicherungsexperten bei Grenzfällen uneinig sind, und die Umsetzung eines neuen Plantyps erfordert Wochen der Analyse, um alle Wechselwirkungen zu verstehen. Ein weiteres Beispiel ist ein Rohstoffhandelssystem, bei dem die Preisgestaltung vom Lieferort, Vertragstyp, saisonalen Faktoren, Lagerkosten, Währungsschwankungen und regulatorischen Anforderungen abhängt, die je nach Rechtsraum variieren. Das Domänenwissen, das nötig ist, um zu verstehen, warum ein bestimmter Preisalgorithmus funktioniert, erfordert Expertise sowohl in Finanzmärkten als auch im spezifischen gehandelten Rohstoff.
