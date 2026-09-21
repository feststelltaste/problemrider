---
title: Modulith
description: Strukturierung der Systemarchitektur in unabhängige, austauschbare
  Module.
category:
- Architecture
problems:
- monolithic-architecture-constraints
- high-coupling-low-cohesion
- tight-coupling-issues
- stagnant-architecture
- ripple-effect-of-changes
- difficult-code-reuse
- deployment-coupling
layout: solution
lang: de
en_slug: modulith
related_solutions:
- slug: microservices-architecture
  similarity: 0.75
- slug: modularization-and-bounded-contexts
  similarity: 0.75
- slug: high-cohesion
  similarity: 0.7
- slug: layered-architecture
  similarity: 0.7
- slug: hexagonal-architecture
  similarity: 0.65
- slug: containerization
  similarity: 0.65
---

## Description

Ein Modulith hält ein System als eine einzige Einheit deployt, während er harte interne Grenzen zwischen seinen logischen Modulen durchsetzt — typischerweise mittels Sprachmechanismen wie Paketen oder Build-Modulen, expliziten öffentlichen APIs für jedes Modul und architektonischen Fitness Tests wie ArchUnit, die den Build fehlschlagen lassen, wenn Code über eine Grenze greift, die er nicht überschreiten sollte. Er erreicht viele der mit Microservices verbundenen Kopplungs- und Klarheitsvorteile — gut definierte Schnittstellen, eingeschränkter interner Zugriff, klare Eigentümerschaft über einen begrenzten Funktionalitätsbereich —, ohne die Netzwerkaufrufe, unabhängigen Deployments und Fehlermodi verteilter Systeme einzuführen, die mit der tatsächlichen Aufteilung des Systems in separate Services einhergehen. In einem Legacy-Monolithen adressiert dies ein sehr spezifisches Fehlermuster: Domänenlogik, die über Pakete ohne durchgesetzte Grenzen verwoben wurde, sodass eine Änderung in einem Bereich still in andere kaskadiert, weil nichts in der Codebasis Module davon abhält, in die Interna anderer zu greifen. Weil ein Modulith ein einziges deploybares Artefakt bleibt, ist er merklich leichter in Legacy-Code nachzurüsten als eine vollständige Microservices-Zerlegung, was ihn zu einem praktischen Sprungbrett für Teams macht, die erkennen, dass die Kopplung ihres Monolithen ein Problem ist, aber noch nicht die betriebliche Reife oder die ausreichend klar verstandenen Domänengrenzen haben, um verteilte Services zu rechtfertigen. Sein Hauptrisiko ist, dass die Grenzen in einem Modulith, anders als eine echte, durch Netzwerkaufrufe durchgesetzte Servicegrenze, nur durch Disziplin und Tooling innerhalb einer einzigen Codebasis durchgesetzt werden, sodass sie ohne konsequent ausgeführte Fitness Tests dazu neigen, unter demselben Termindruck erneut zu erodieren, der die ursprüngliche Verflechtung verursachte.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie logische Modulgrenzen innerhalb des Monolithen basierend auf Domänenfähigkeiten
- Setzen Sie Modulgrenzen mittels Sprachmechanismen wie Paketen, Namespaces oder Build-Modulen durch
- Definieren Sie explizite öffentliche APIs für jedes Modul und beschränken Sie den Zugriff auf interne Implementierung
- Nutzen Sie architektonische Fitness Tests oder Werkzeuge wie ArchUnit, um Grenzverletzungen zu erkennen und zu verhindern
- Kommunizieren Sie zwischen Modulen über gut definierte Schnittstellen oder Events statt direkter interner Aufrufe
- Migrieren Sie den Monolithen inkrementell und wandeln Sie einen verworrenen Bereich nach dem anderen in ein ordentliches Modul um
- Halten Sie Module als eine einzige Einheit deploybar, während die Option erhalten bleibt, sie später als Services zu extrahieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Erreicht viele Vorteile von Microservices ohne die betriebliche Komplexität verteilter Systeme
- Bietet ein natürliches Sprungbrett in Richtung Microservices, falls später benötigt
- Bewahrt die Einfachheit eines einzigen Deployments, während klare Grenzen durchgesetzt werden
- Leichter in Legacy-Systeme einzuführen als eine vollständige Microservice-Zerlegung

**Kosten und Risiken:**
- Erfordert Disziplin, um Modulgrenzen innerhalb einer einzigen Codebasis zu pflegen
- Ohne strikte Durchsetzung erodieren Grenzen über die Zeit unter Termindruck
- Bietet keine unabhängige Skalierung oder Deployment einzelner Module
- Teams könnten es als halbe Maßnahme behandeln und nicht genug in Grenzdurchsetzung investieren

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein mittelgroßes SaaS-Unternehmen hatte eine monolithische Spring-Boot-Anwendung, in der die gesamte Domänenlogik ohne klare Grenzen über Pakete verwoben war. Sie erwogen Microservices, hatten aber nicht die betriebliche Reife. Stattdessen strukturierten sie die Anwendung mittels Spring Modulith in einen Modulith um, definierten klare Modulgrenzen für Abrechnung, Nutzerverwaltung und Reporting. Jedes Modul legte ein öffentliches API-Paket offen und kommunizierte über Anwendungsevents. ArchUnit-Tests verhinderten modulübergreifenden internen Zugriff. Dies gab Teams klare Eigentümerschaft über Module und verringerte versehentliche Kopplung erheblich, während das System ein einziges deploybares Artefakt blieb.
