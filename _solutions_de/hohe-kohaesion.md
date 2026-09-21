---
title: Hohe Kohäsion
description: Sicherstellung, dass jedes Modul einen fokussierten, klar definierten
  Zweck mit eng verwandten Verantwortlichkeiten hat.
category:
- Architecture
- Code
problems:
- high-coupling-low-cohesion
- bloated-class
- god-object-anti-pattern
- monolithic-functions-and-classes
- spaghetti-code
- difficult-code-comprehension
- ripple-effect-of-changes
- tangled-cross-cutting-concerns
- excessive-class-size
- over-reliance-on-utility-classes
- poor-encapsulation
- circular-dependency-problems
- single-entry-point-design
layout: solution
lang: de
en_slug: high-cohesion
related_solutions:
- slug: facades
  similarity: 0.75
- slug: dependency-injection
  similarity: 0.75
- slug: mediator
  similarity: 0.75
- slug: abstraction
  similarity: 0.75
- slug: layered-architecture
  similarity: 0.75
- slug: loose-coupling
  similarity: 0.75
---

## Description

Hohe Kohäsion ist die Eigenschaft, dass der Inhalt eines Moduls — seine Methoden, Felder und Verantwortlichkeiten — eng verwandt ist und einem einzigen, klar definierten Zweck dient, sodass das Modul einen klaren Änderungsgrund hat. Sie wird diagnostiziert, indem nach dem gegenteiligen Muster gesucht wird: Klassen mit vagen Namen wie „Manager" oder „Helper", Methoden, die auf völlig disjunkten Teilmengen der Felder einer Klasse operieren, oder eine einzige Klasse, die unzusammenhängende Belange wie Registrierung, Abrechnung und Terminplanung gleichzeitig ansammelt — das God-Object-Antipattern, das in langlebigen Legacy-Codebasen endemisch ist, in denen neue Funktionalität wiederholt zu welcher Klasse auch immer gerade bequem war hinzugefügt wurde statt zu einer zweckgebauten. Kohäsion wiederherzustellen bedeutet, Cluster von Methoden und Daten zu identifizieren, die echt zusammengehören, basierend darauf, welche Felder sie tatsächlich nutzen, und jeden Cluster in eine eigene fokussierte Klasse oder ein eigenes Modul zu extrahieren, typischerweise eine Verantwortlichkeit nach der anderen statt in einer einzigen disruptiven Neufassung. Dies ist besonders für Legacy-Systeme wichtig, weil geringe Kohäsion es ist, was kleine Änderungen unvorhersehbar riskant macht: Wenn unzusammenhängende Verantwortlichkeiten veränderlichen Zustand innerhalb einer Klasse teilen, kann eine für einen Bereich gedachte Änderung einen anderen still brechen, nur weil sie zufällig in derselben Datei leben. Das Ergebnis bewusst erhöhter Kohäsion ist, dass Verantwortlichkeiten unabhängig verständlich, testbar und änderbar werden, was direkt den Kaskadeneffekt von Änderungen verringert, der eng verflochtenen Legacy-Code plagt.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Analysieren Sie bestehende Klassen und Module auf Anzeichen geringer Kohäsion: mehrere unzusammenhängende Verantwortlichkeiten, Methoden, die nicht dieselben Felder nutzen, oder vage Namen wie „Manager" oder „Helper"
- Extrahieren Sie Gruppen zusammengehöriger Methoden und Daten in neue, fokussierte Klassen oder Module
- Nutzen Sie das Single-Responsibility-Prinzip als Leitfaden: Jedes Modul sollte einen Änderungsgrund haben
- Refaktorieren Sie God Objects inkrementell, indem Sie eine Verantwortlichkeit nach der anderen in eine eigene Klasse verschieben
- Richten Sie Modulgrenzen an Domänenkonzepten aus, sodass jedes Modul auf eine klare Geschäftsfähigkeit abgebildet ist
- Überprüfen Sie die Methoden-zu-Feld-Nutzung innerhalb von Klassen, um Cluster zu identifizieren, die zusammengehören

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Code wird leichter verständlich, weil jedes Modul einen klaren, begrenzten Zweck hat
- Änderungen werden lokalisiert: Die Modifikation einer Verantwortlichkeit riskiert nicht, unzusammenhängende Funktionalität zu brechen
- Testen wird einfacher, weil fokussierte Module weniger Abhängigkeiten und Szenarien haben
- Verbessert die Teamproduktivität, indem die kognitive Last bei der Arbeit an einzelnen Komponenten verringert wird

**Kosten und Risiken:**
- Das Aufteilen von Modulen erhöht die Gesamtzahl der Dateien und kann sich für kleine Systeme wie Überkonstruktion anfühlen
- Erfordert sorgfältige Identifikation von Verantwortlichkeitsgrenzen, was subjektiv sein kann
- Zwischenzustände des Refactorings können die Komplexität vorübergehend erhöhen, bevor der volle Nutzen realisiert wird
- Kann versteckte Abhängigkeiten offenlegen, die von der monolithischen Struktur verdeckt waren

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Gesundheitsanwendung hatte eine `PatientService`-Klasse mit über 4.000 Codezeilen, die Patientenregistrierung, Abrechnung, Terminplanung und Krankenaktenabfragen handhabte. Jede Änderung an der Abrechnungslogik riskierte, die Terminplanung zu brechen, weil sie veränderlichen Zustand innerhalb der Klasse teilten. Das Team extrahierte systematisch jede Verantwortlichkeit in einen eigenen Dienst: `PatientRegistrationService`, `BillingService`, `AppointmentService` und `MedicalRecordService`. Jeder neue Dienst war kohäsiv und unabhängig testbar. Die Fehlerraten im Patientenmodul sanken im folgenden Quartal spürbar, und Entwickler berichteten, deutlich weniger Zeit damit zu verbringen, Code zu verstehen, bevor sie Änderungen vornahmen.
