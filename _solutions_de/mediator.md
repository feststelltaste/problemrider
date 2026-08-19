---
title: Mediator
description: Entkopplung direkter Kommunikation zwischen Komponenten.
category:
- Architecture
- Code
problems:
- high-coupling-low-cohesion
- tight-coupling-issues
- spaghetti-code
- circular-dependency-problems
- monolithic-architecture-constraints
- ripple-effect-of-changes
layout: solution
lang: de
en_slug: mediator
related_solutions:
- slug: adapter
  similarity: 0.75
- slug: facades
  similarity: 0.75
- slug: high-cohesion
  similarity: 0.75
- slug: abstraction
  similarity: 0.75
- slug: dependency-injection
  similarity: 0.7
- slug: protocol-abstraction
  similarity: 0.7
---

## Description

Das Mediator-Muster führt ein dediziertes Objekt ein, das die Kommunikations- und Koordinationslogik zwischen einer Menge von Komponenten kapselt, sodass diese Komponenten keine direkten Referenzen mehr aufeinander halten und sich gegenseitig aufrufen, sondern ausschließlich über den Mediator interagieren. Mechanisch wandelt dies ein dichtes, viele-zu-viele-Geflecht direkter Abhängigkeiten in eine einfachere, sternförmige Struktur um, in der jede Komponente nur vom Mediator abhängt, der dann die Verantwortung übernimmt zu orchestrieren, wie sie zusammenarbeiten. In Legacy-Systemen sammeln Cluster von Klassen über Jahre inkrementeller Feature-Ergänzungen häufig direkte Referenzen aufeinander an, bis eine Änderung an einer Komponente das Verstehen und Modifizieren eines Dutzends anderer erfordert, die alle direkt und auf leicht unterschiedliche Weise mit ihr koordinieren — ein Kennzeichen der verworrenen, spaghetti-artigen Kopplung, die Legacy-Code unverhältnismäßig teuer in der Änderung macht. Einen Mediator um einen solchen Cluster einzuführen verringert nicht die Gesamtmenge an Koordinationslogik im System, aber konsolidiert und zentralisiert sie, sodass das Hinzufügen, Entfernen oder Ersetzen einer Komponente nur erfordert, ihre Interaktion mit dem Mediator zu aktualisieren, statt jeder anderen Komponente, die sie zuvor direkt referenzierte. Das Risiko, auf das zu achten ist, besonders beim Nachrüsten dieses Musters in Legacy-Code, ist, dass der Mediator selbst über die Zeit so viel Logik ansammeln kann, dass er zu einem neuen God Object und einem neuen Engpass wird, sodass er eng auf Koordination begrenzt und frei von Geschäftslogik gehalten werden muss, die anderswo hingehört.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie Cluster von Komponenten, die auf komplexe, verworrene Weise direkt miteinander kommunizieren
- Führen Sie ein Mediator-Objekt ein, das die Interaktionslogik zwischen diesen Komponenten kapselt
- Refaktorieren Sie Komponenten, damit sie über den Mediator kommunizieren, statt direkte Referenzen aufeinander zu halten
- Nutzen Sie den Mediator, um Koordinationsworkflows zu verwalten, die zuvor mehrere eng gekoppelte Klassen umspannten
- Halten Sie den Mediator auf Koordinationslogik fokussiert; vermeiden Sie, ihn zu einem God Object mit Geschäftslogik zu machen
- Führen Sie Mediatoren inkrementell ein, beginnend mit den am stärksten verworrenen Komponenten-Clustern

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verringert die Zahl direkter Abhängigkeiten zwischen Komponenten und vereinfacht den Abhängigkeitsgraphen
- Erleichtert das Hinzufügen, Entfernen oder Ersetzen einzelner Komponenten, ohne andere zu beeinflussen
- Zentralisiert Koordinationslogik, die zuvor verstreut und dupliziert war

**Kosten und Risiken:**
- Der Mediator kann zu einem einzigen Komplexitätspunkt werden, wenn er zu viel Logik ansammelt
- Fügt eine Indirektionsebene hinzu, die den Kontrollfluss schwerer nachvollziehbar machen kann
- Übermäßige Anwendung erzeugt unnötige Mediatoren für einfache Interaktionen
- Der Mediator muss sorgfältig gestaltet werden, um nicht zu einem Engpass zu werden

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-UI-Framework hatte 20 Formularkomponenten, die sich gegenseitig direkt referenzierten, um Validierung, Sichtbarkeit und Datenaktualisierungen zu koordinieren. Das Hinzufügen eines neuen Feldes erforderte die Modifikation von bis zu 12 bestehenden Komponenten. Das Team führte einen FormMediator ein, der die gesamte Interkomponentenkommunikation über Events verwaltete. Nach dem Refactoring erforderte das Hinzufügen eines neuen Feldes nur die Implementierung des Feldes selbst und dessen Registrierung beim Mediator, was den Aufwand von zwei Tagen auf zwei Stunden verringerte.
