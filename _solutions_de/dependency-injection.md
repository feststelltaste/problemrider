---
title: Dependency Injection
description: Externe Verwaltung und Injektion von Abhängigkeiten zwischen Komponenten.
category:
- Code
- Architecture
problems:
- tight-coupling-issues
- difficult-to-test-code
- hidden-dependencies
- high-coupling-low-cohesion
- difficult-code-reuse
- technology-lock-in
- global-state-and-side-effects
- improper-event-listener-management
- circular-dependency-problems
layout: solution
lang: de
en_slug: dependency-injection
related_solutions:
- slug: abstracted-file-system-access
  similarity: 0.8
- slug: adapter
  similarity: 0.75
- slug: dependency-injection-container
  similarity: 0.75
- slug: integration-tests
  similarity: 0.75
- slug: database-abstraction
  similarity: 0.75
- slug: abstraction-layers
  similarity: 0.75
---

## Description

Dependency Injection ist die Praxis, eine Komponente mit den Objekten zu versorgen, von denen sie abhängt, von außen — typischerweise über Konstruktorparameter —, statt die Komponente diese Abhängigkeiten selbst mittels `new`-Aufrufen oder statischen Factory-Methoden konstruieren oder nachschlagen zu lassen. Abhängigkeiten auf diese Weise explizit zu machen bedeutet, dass die Konstruktorsignatur einer Klasse zu einer vollständigen, sichtbaren Liste dessen wird, was sie zum Funktionieren braucht, und jede dieser Abhängigkeiten kann gegen eine alternative Implementierung ausgetauscht werden — ein Test-Double, eine andere umgebungsspezifische Implementierung, einen Cloud-Storage-Adapter anstelle eines lokalen Dateisystems —, ohne die Klasse selbst zu modifizieren. Dies ist grundlegend für Legacy-Modernisierung, weil Code, der seine eigenen Abhängigkeiten intern erzeugt, konstruktionsbedingt resistent gegen Unit-Testing ist: Eine einzelne Klasse auszuüben zieht unweigerlich jede konkrete Abhängigkeit mit sich, die sie konstruiert, was genau der Grund ist, warum Legacy-Codebasen, die auf statischen Helfern und direkter Instanziierung aufgebaut sind, typischerweise wenig bis keine automatisierte Testabdeckung haben. Dependency Injection in einem bestehenden System einzuführen erfolgt schrittweise: Zuerst werden Schnittstellen für die Abhängigkeiten der testbarkeitsbeschränktesten Klassen extrahiert und ihre Konstruktoren refaktoriert, um diese Schnittstellen als Parameter zu akzeptieren, oft wird ein DI-Container eingeführt, um die resultierende Objektverdrahtung zu verwalten, sobald genug Klassen umgewandelt wurden. Über das Ermöglichen von Tests hinaus deckt das Explizitmachen von Abhängigkeiten routinemäßig strukturelle Probleme auf, die in impliziten Objektgraphen versteckt waren — zirkuläre Abhängigkeiten und Klassen mit einer unangemessen großen Anzahl von Mitarbeitern —, was Designprobleme sichtbar macht, die die implizite Verdrahtung des Legacy-Codes still verschleiert hatte.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie Klassen in der Legacy-Codebasis, die ihre eigenen Abhängigkeiten intern mittels new-Operatoren oder statischen Factory-Aufrufen erzeugen
- Extrahieren Sie Schnittstellen für Schlüsselabhängigkeiten, sodass Implementierungen ausgetauscht werden können, ohne Konsumenten zu ändern
- Refaktorieren Sie Konstruktoren, um Abhängigkeiten als Parameter zu akzeptieren, statt sie intern zu erzeugen
- Führen Sie einen DI-Container (Spring, Guice, .NET DI oder eine einfache handgebaute Factory) ein, um Objekterzeugung und -verdrahtung zu verwalten
- Beginnen Sie mit den testbarkeitsbeschränktesten Klassen und erweitern Sie die DI-Einführung schrittweise
- Nutzen Sie DI, um umgebungsspezifische Implementierungen zu injizieren (Produktionsdatenbank vs. Test-Double, Cloud-Storage vs. lokales Dateisystem)
- Vermeiden Sie Service-Locator-Antimuster, die Abhängigkeiten hinter einer globalen Registry verstecken

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Macht Abhängigkeiten in Konstruktorsignaturen explizit und sichtbar
- Ermöglicht Unit-Testing, indem Mock- oder Stub-Implementierungen injiziert werden können
- Reduziert Kopplung zwischen Komponenten, was die Codebasis modularer und portabler macht
- Erleichtert das Austauschen von Implementierungen für unterschiedliche Umgebungen oder Plattformen
- Vereinfacht Refactoring, indem Änderung auf die Injektionskonfiguration isoliert wird

**Kosten und Risiken:**
- DI-Container fügen Framework-Komplexität und eine Lernkurve für mit dem Muster nicht vertraute Teams hinzu
- Übermäßige Nutzung von DI kann das Laufzeitverhalten der Anwendung schwer verständlich machen, indem verschleiert wird, welche Implementierung aktiv ist
- Legacy-Code mit tiefen statischen Methodenketten oder globalem Zustand erfordert erhebliches Refactoring, um DI einzuführen
- Konstruktorparameterlisten können unhandlich werden, wenn zu viele Abhängigkeiten injiziert werden (was anzeigt, dass die Klasse Zerlegung braucht)
- Laufzeit-Verdrahtungsfehler werden möglicherweise erst beim Anwendungsstart erkannt, anders als bei Compile-Zeit-Abhängigkeiten

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Legacy-.NET-Anwendung nutzte durchgehend statische Helferklassen und direkte Instanziierung, was Unit-Testing nahezu unmöglich machte. Das Team musste vor einem kritischen Modernisierungsaufwand Tests hinzufügen. Sie begannen damit, Constructor Injection für die 30 kritischsten Geschäftslogikklassen einzuführen, und extrahierten Schnittstellen für Datenbankzugriff, E-Mail-Versand und Dateioperationen. Mit dem eingebauten DI-Container von .NET verdrahteten sie Produktionsimplementierungen für die Laufzeit und injizierten Mock-Implementierungen in Tests. Innerhalb von drei Monaten stieg die Testabdeckung dieser 30 Klassen von null auf 80 Prozent, und das Team entdeckte während des Prozesses vier latente Fehler. Die expliziten Abhängigkeitsdeklarationen enthüllten auch mehrere zirkuläre Abhängigkeiten, die unsichtbar gewesen waren, als Abhängigkeiten intern erzeugt wurden.
