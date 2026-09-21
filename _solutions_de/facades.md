---
title: Facades
description: Nutzung von Facades, um komplexe Subsysteme hinter einer vereinfachten
  Schnittstelle zu verbergen.
category:
- Architecture
- Code
problems:
- monolithic-architecture-constraints
- difficult-code-comprehension
- high-coupling-low-cohesion
- spaghetti-code
- difficult-code-reuse
- poor-interfaces-between-applications
- poor-encapsulation
layout: solution
lang: de
en_slug: facades
related_solutions:
- slug: adapter
  similarity: 0.8
- slug: pattern-language
  similarity: 0.75
- slug: high-cohesion
  similarity: 0.75
- slug: mediator
  similarity: 0.75
- slug: abstraction
  similarity: 0.75
- slug: dependency-injection
  similarity: 0.75
---

## Description

Eine Facade ist eine einzelne, vereinfachte Schnittstelle, die vor ein komplexes Subsystem gestellt wird und jeden Aufruf an die bestehenden untergeordneten Klassen und Funktionen weiterreicht, ohne deren interne Struktur zu ändern, sodass die meisten Konsumenten nur noch die Handvoll übergeordneter Methoden der Facade lernen müssen statt der gesamten Oberfläche des Subsystems. Legacy-Subsysteme wachsen über Jahre inkrementeller Erweiterung häufig auf Dutzende öffentliche Klassen an, ohne einen einzigen offensichtlichen Einstiegspunkt für jemanden, der lediglich eine übliche Aufgabe erledigen möchte, was Onboarding zu einer archäologischen Übung macht und jede neue Integration zu einer neuen Gelegenheit, die Interna des Subsystems falsch zu nutzen. Die Einführung einer Facade für die häufigsten Anwendungsfälle des Subsystems gibt neuen Konsumenten einen stabilen, erlernbaren Vertrag, während sie fortgeschrittenen Nutzern weiterhin freisteht, direkt an ihr vorbei ins Subsystem zu greifen, wenn sie es wirklich brauchen, und weil die Facade an einer sauberen Grenze sitzt, wird sie auch zu einer bequemen Nahtstelle, um das Subsystem später vollständig zu ersetzen, ohne jeden Aufrufer anfassen zu müssen. Das Risiko besteht darin, dass die Facade, sobald alles darüber geleitet wird, zu einem Engpass werden oder anfangen kann, Logik zu absorbieren, die eigentlich ins Subsystem selbst gehört, und das parallele Pflegen sowohl der Facade als auch etwaiger direkter Zugriffspfade erhöht die Fläche, die konsistent gehalten werden muss.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie komplexe Subsysteme in der Legacy-Codebasis, mit denen Konsumenten über viele untergeordnete Klassen oder Funktionen interagieren
- Erstellen Sie eine Facade-Klasse oder ein Modul, das eine vereinfachte, übergeordnete API für die häufigsten Anwendungsfälle bereitstellt
- Leiten Sie alle Aufrufe von der Facade an die bestehenden Subsystem-Klassen weiter, ohne deren interne Struktur zu ändern
- Nutzen Sie Facades als Einstiegspunkt für neue Konsumenten, während fortgeschrittenen Nutzern erlaubt bleibt, die Facade bei Bedarf zu umgehen
- Führen Sie Facades schrittweise ein, beginnend mit den Subsystemen mit den meisten Konsumenten oder der steilsten Lernkurve
- Schreiben Sie Tests gegen die Facade-Schnittstelle, um einen stabilen Vertrag zu etablieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verringert die Lernkurve für Entwickler, die mit komplexen Legacy-Subsystemen arbeiten
- Schafft eine stabile Schnittstelle, die Konsumenten vor internen Subsystemänderungen abschirmt
- Bietet eine natürliche Nahtstelle für künftiges Refactoring oder den Ersatz des zugrundeliegenden Subsystems

**Kosten und Risiken:**
- Die Facade kann zu einem Engpass werden, wenn aller Zugriff durch sie erzwungen wird
- Risiko, dass die Facade Logik ansammelt, die eigentlich im Subsystem leben sollte
- Eine schlecht gestaltete Facade kann die Schnittstelle übermäßig vereinfachen und notwendige Funktionalität einschränken
- Das Pflegen sowohl der Facade als auch direkter Zugriffspfade vergrößert die Angriffs- bzw. Wartungsfläche

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Legacy-Buchhaltungssystem hatte 47 öffentliche Klassen in seinem Abrechnungsmodul, ohne klaren Einstiegspunkt. Neue Entwickler verbrachten durchschnittlich zwei Wochen damit, zu verstehen, wie das Modul für übliche Aufgaben zu nutzen ist. Das Team führte eine BillingFacade mit fünf Methoden ein, die 90 % der Abrechnungsanwendungsfälle abdeckten. Die Onboarding-Zeit für neue Entwickler sank von zwei Wochen auf zwei Tage, und die Facade diente später als Vertragsgrenze, als das Team begann, die zugrundeliegende Abrechnungs-Engine durch eine moderne Implementierung zu ersetzen.
