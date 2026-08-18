---
title: Wellenwirkung von Änderungen
description: Eine kleine Änderung in einem Teil des Systems erfordert Modifikationen
  in vielen anderen, scheinbar unabhängigen Teilen, was auf hohe Kopplung hindeutet.
category:
- Architecture
- Code
related_problems:
- slug: unpredictable-system-behavior
  similarity: 0.7
- slug: tight-coupling-issues
  similarity: 0.7
- slug: cascade-failures
  similarity: 0.7
- slug: change-management-chaos
  similarity: 0.65
- slug: rapid-system-changes
  similarity: 0.65
- slug: deployment-coupling
  similarity: 0.6
solutions:
- design-by-contract
- loose-coupling
- separation-of-concerns
- solid-principles
- abstraction
- architecture-conformity-analysis
- backward-compatibility
- bounded-contexts
- bridges
- fault-containment
- high-cohesion
- layered-architecture
- mediator
- modulith
- semantic-versioning
- tolerant-reader
- domain-aligned-architecture
- fitness-functions
- change-impact-analysis
layout: problem
lang: de
en_slug: ripple-effect-of-changes
---

## Description

Die Wellenwirkung von Änderungen tritt auf, wenn die Modifikation einer Komponente Änderungen in zahlreichen anderen Komponenten im gesamten System erfordert, selbst wenn diese Komponenten logisch unabhängig sein sollten. Dies deutet auf exzessive Kopplung zwischen Systemteilen und schlechte Trennung der Zuständigkeiten hin. Die Wellenwirkung macht einfache Änderungen teuer und riskant, da Entwickler mehrere Bereiche der Codebasis für das modifizieren und testen müssen, was isolierte Änderungen sein sollten.

## Indicators ⟡
- Einfache Feature-Änderungen erfordern Modifikationen über mehrere Module oder Schichten hinweg
- Bugfixes in einem Bereich brechen Funktionalität in nicht verwandten Bereichen
- Das Hinzufügen neuer Funktionalität erfordert das Verständnis und die Modifikation großer Teile der Codebasis
- Entwickler sagen regelmäßig „wenn wir das ändern, müssen wir auch X, Y und Z ändern"
- Auswirkungsanalysen für Änderungen offenbaren konsequent mehr betroffene Komponenten als erwartet

## Symptoms ▲

- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Wenn jede Änderung Modifikationen über viele Komponenten hinweg erfordert, dauern selbst einfache Features unverhältnismäßig lange.
- [Angst vor Breaking Changes](angst-vor-breaking-changes.md)
<br/>  Der weite Wirkungsradius von Änderungen macht Entwickler ängstlich, Code zu modifizieren, da sie wissen, dass scheinbar lokale Änderungen entfernte Komponenten brechen können.
- [Regressionsfehler](regressionsfehler.md)
<br/>  Änderungen, die über Komponenten hinweg wellenartig wirken, führen häufig Regressionen in Bereichen ein, von denen Entwickler nicht wussten, dass sie betroffen sind.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Der verstärkte Aufwand, der für jede Änderung erforderlich ist, treibt die Kosten der Wartung und Weiterentwicklung des Systems in die Höhe.
- [Widerstand gegen Veränderung](widerstand-gegen-veraenderung.md)
<br/>  Teams werden zögerlich, Verbesserungen vorzunehmen, wenn sie wissen, dass jede Änderung in umfangreiche Modifikationen über die Codebasis hinweg kaskadiert.

## Causes ▼

- [Probleme durch enge Kopplung](probleme-durch-enge-kopplung.md)
<br/>  Exzessive Abhängigkeiten zwischen Komponenten bedeuten, dass Änderungen in einer Komponente direkt Änderungen in ihren Abhängigen erfordern.
- [Versteckte Abhängigkeiten](versteckte-abhaengigkeiten.md)
<br/>  Undokumentierte und nicht offensichtliche Abhängigkeiten zwischen Komponenten verursachen, dass sich Änderungen über unerwartete Pfade fortpflanzen.
- [Schlechte Kapselung](schlechte-kapselung.md)
<br/>  Wenn interne Details statt gekapselt offengelegt werden, hängt externer Code von Implementierungsspezifika ab, was kaskadierende Änderungen erzwingt.
- [God-Object-Antipattern](god-object-antipattern.md)
<br/>  God-Objects, auf die im gesamten System verwiesen wird, schaffen einen zentralen Punkt, an dem sich Änderungen zu allem abhängigen Code fortpflanzen.

## Detection Methods ○
- **Änderungsauswirkungsanalyse:** Verfolgung, wie viele Dateien oder Module für typische Änderungen modifiziert werden müssen
- **Abhängigkeitsanalyse-Werkzeuge:** Nutzung statischer Analyse zur Visualisierung und Messung der Kopplung zwischen Komponenten
- **Korrelation der Änderungshäufigkeit:** Identifikation von Komponenten, die häufig zusammen geändert werden, was auf Kopplung hindeutet
- **Entwickler-Feedback:** Befragung von Entwicklern zum typischen Umfang der von ihnen vorgenommenen Änderungen
- **Code-Review-Muster:** Überwachung, wie oft Reviews Diskussionen über weitreichende Änderungen beinhalten

## Examples

Ein E-Commerce-System muss Unterstützung für eine neue Zahlungsmethode hinzufügen. Was eine einfache Ergänzung des Zahlungsverarbeitungsmoduls sein sollte, erfordert stattdessen Änderungen an: der Bestellvalidierungslogik (die Zahlungstypen hartcodiert), der Benutzeroberfläche (die zahlungsspezifische Anzeigelogik über den gesamten Code verstreut hat), dem Berichtssystem (das Zahlungstabellen direkt abfragt), dem E-Mail-Benachrichtigungssystem (das zahlungsspezifische Vorlagen hat) und dem Bestandsverwaltungssystem (das unterschiedliche Reservierungslogik für unterschiedliche Zahlungstypen hat). Eine Änderung, die einige Stunden dauern sollte, endet mit zwei Wochen Entwicklung und umfangreichen Regressionstests über die gesamte Anwendung. Ein weiteres Beispiel betrifft ein Content-Management-System, bei dem das Hinzufügen eines neuen Felds zu Benutzerprofilen Modifikationen am Datenbankschema, den UI-Komponenten, der Validierungslogik, der Suchindizierung, der Exportfunktionalität, den Benutzer-Migrationsskripten, den API-Endpunkten, der mobilen App-Synchronisation und Drittanbieter-Integrationen erfordert. Die Wellenwirkung macht aus einer einfachen Datenbankänderung ein komplexes Projekt mit mehreren Teams und Systemen.
