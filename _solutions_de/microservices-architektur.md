---
title: Microservices-Architektur
description: Aufteilung der Anwendung in kleine, unabhängige Services.
category:
- Architecture
problems:
- monolithic-architecture-constraints
- deployment-coupling
- tight-coupling-issues
- scaling-inefficiencies
- high-coupling-low-cohesion
- technology-lock-in
- slow-development-velocity
- large-risky-releases
- stagnant-architecture
layout: solution
lang: de
en_slug: microservices-architecture
related_solutions:
- slug: microservices
  similarity: 0.9
- slug: containerization
  similarity: 0.75
- slug: modulith
  similarity: 0.75
- slug: layered-architecture
  similarity: 0.75
- slug: hexagonal-architecture
  similarity: 0.75
- slug: service-mesh
  similarity: 0.7
---

## Description

Microservices-Architektur zerlegt ein System in eine Menge kleiner, unabhängig deploybarer Services, jeder an eine spezifische Geschäftsfähigkeit ausgerichtet, die über das Netzwerk mittels gut definierter APIs kommunizieren, statt über In-Process-Methodenaufrufe oder eine gemeinsame Datenbank. Die Extraktion aus einem bestehenden Legacy-Monolithen erfolgt typischerweise inkrementell durch das Strangler-Fig-Muster, indem Bounded Contexts mittels Domain-Driven Design identifiziert werden und Verkehr Schritt für Schritt zu neuen Services geleitet wird, während der Monolith weiterhin alles handhabt, was noch nicht extrahiert wurde, statt eine vollständige Neufassung zu versuchen. Legacy-Monolithen erreichen tendenziell einen Punkt, an dem jede Änderung, wie klein auch immer, das erneute Deployment und Testen der gesamten Anwendung erfordert, weil alle ihre Subsysteme zusammen kompiliert und ausgeliefert werden, was die Lieferung verlangsamt, Risiko in jedem Release konzentriert und es unmöglich macht, dass ein Team seinen Teil des Systems unabhängig vom Zeitplan jedes anderen Teams skaliert oder weiterentwickelt. Microservices adressieren dies direkt, indem sie den einzelnen Service statt der gesamten Anwendung zur Einheit von Deployment, Skalierung und Technologiewahl machen, was auch eine natürliche Abbildung zwischen Service-Eigentümerschaft und Teamgrenzen schafft und erlaubt, Legacy-Technologie schrittweise, Service für Service, statt auf einmal zu ersetzen. Der Zielkonflikt, den Legacy-Teams dafür eingehen, ist eine erhebliche Zunahme der Komplexität verteilter Systeme — Netzwerklatenz, Teilausfälle und dienstübergreifende Datenkonsistenz —, mit denen sich ein Monolith nie befassen musste, und vorzeitige oder übermäßig feingranulare Zerlegung eines schlecht verstandenen Legacy-Systems produziert häufig einen „verteilten Monolithen", der schwerer zu betreiben ist als das System, das er ersetzte.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie Bounded Contexts im Legacy-Monolithen mittels Domain-Driven-Design-Techniken
- Beginnen Sie mit der Extraktion des am wenigsten gekoppelten, am unabhängigsten deploybaren Moduls als ersten Microservice
- Nutzen Sie das Strangler-Fig-Muster, um Verkehr schrittweise vom Monolithen zu neuen Services zu leiten
- Definieren Sie klare API-Verträge zwischen Services mittels REST, gRPC oder Messaging, bevor Codebasen aufgeteilt werden
- Führen Sie ein API-Gateway oder Service Mesh ein, um Routing, Authentifizierung und Observability zu verwalten
- Richten Sie unabhängige CI/CD-Pipelines für jeden Service ein, um autonome Team-Deployments zu ermöglichen
- Implementieren Sie verteiltes Tracing und zentralisiertes Logging von Anfang an, um betriebliche Sichtbarkeit zu bewahren
- Planen Sie eine Datenzerlegungsstrategie, damit jeder Service seine Daten besitzt, statt eine einzige Datenbank zu teilen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Ermöglicht unabhängiges Deployment und Skalierung einzelner Services
- Erlaubt verschiedenen Services, unterschiedliche, für ihre Problemdomäne passende Technologie-Stacks zu nutzen
- Verringert den Explosionsradius von Ausfällen und Änderungen auf eine einzige Servicegrenze
- Erleichtert Teamautonomie, indem Service-Eigentümerschaft mit Teamgrenzen ausgerichtet wird
- Durchbricht Anbieter-Lock-in, indem schrittweise Plattformmigration Service für Service ermöglicht wird

**Kosten und Risiken:**
- Führt Komplexität verteilter Systeme ein, einschließlich Netzwerklatenz, Teilausfälle und Herausforderungen der Datenkonsistenz
- Erfordert erhebliche Investition in Infrastruktur, Monitoring und betriebliches Tooling
- Vorzeitige Zerlegung kann einen verteilten Monolithen erzeugen, der schwerer zu verwalten ist als das Original
- Dienstübergreifendes Refactoring und Schemaänderungen werden schwerer zu koordinieren
- Teams brauchen starke DevOps-Fähigkeiten, um unabhängige Service-Lebenszyklen zu verwalten

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein E-Commerce-Unternehmen hatte eine monolithische Anwendung, in der das Deployment einer Änderung an der Empfehlungs-Engine das erneute Deployment des gesamten Systems erforderte, einschließlich Checkout und Bestandsverwaltung. Das Team nutzte Domain-Driven-Design-Workshops, um fünf Bounded Contexts zu identifizieren, und begann, zuerst den Empfehlungsservice zu extrahieren, da er die wenigsten Datenbankabhängigkeiten hatte. Über achtzehn Monate extrahierten sie vier Services mittels des Strangler-Fig-Musters und leiteten Verkehr schrittweise vom Monolithen um. Jeder Service erhielt seine eigene Deployment-Pipeline und konnte unabhängig skaliert werden. Der Empfehlungsservice wurde wegen seiner ML-Bibliotheken zu Python migriert, während der Checkout-Service auf Java blieb, was die von Microservices gebotene Technologieflexibilität demonstrierte.
