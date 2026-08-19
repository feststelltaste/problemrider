---
title: Microservices
description: Ermöglichung schnellen Produktexperimentierens durch unabhängige,
  geschäftsausgerichtete Services.
category:
- Architecture
problems:
- monolithic-architecture-constraints
- deployment-coupling
- tight-coupling-issues
- slow-feature-development
- scaling-inefficiencies
- increased-time-to-market
- large-risky-releases
- stagnant-architecture
- team-silos
layout: solution
lang: de
en_slug: microservices
related_solutions:
- slug: microservices-architecture
  similarity: 0.9
- slug: strangler-fig-pattern
  similarity: 0.75
- slug: modularization-and-bounded-contexts
  similarity: 0.75
- slug: event-driven-architecture
  similarity: 0.75
- slug: service-mesh
  similarity: 0.7
- slug: business-event-processing
  similarity: 0.7
---

## Description

Microservices sind unabhängig deploybare, an Geschäftsfähigkeiten ausgerichtete Services, die es getrennten Teams erlauben, ihren Teil eines Systems nach eigenem Zeitplan zu entwerfen, zu bauen, auszuliefern und zu skalieren, koordiniert über explizite API-Verträge statt über die gemeinsame Codebasis und den gemeinsamen Release-Kalender. Diesen Ansatz in einem Legacy-Kontext zu übernehmen geschieht durch inkrementelle Extraktion statt Neufassung: Grenzen werden dort identifiziert, wo Datenaustausch und Geschäftslebenszyklus von Natur aus minimal sind, Funktionalität wird Schritt für Schritt mittels des Strangler-Fig-Musters in einen neuen Service abgeschält, und Observability wird eingerichtet, bevor die Extraktion beginnt, weil ein verteiltes System ohne Tracing und zentralisierte Logs weit schwerer zu debuggen ist als der Monolith, den es ersetzt. Legacy-Monolithen neigen dazu, jedes Team in denselben Deployment-Takt und denselben Technologie-Stack zu zwingen, unabhängig davon, ob das zum Problem passt, das jedes Team tatsächlich löst, was die Time-to-Market verlangsamt und verhindert, dass irgendein Teil des Systems unabhängig vom Rest skaliert, ausgeliefert oder modernisiert wird. Die Umstrukturierung um Microservices herum beseitigt diesen gemeinsamen Engpass und erlaubt einem Team, seinen eigenen Service neu zu schreiben oder zu skalieren, ohne sich mit jedem anderen Team abzustimmen, aber der Grad der Zerlegung zählt in der Praxis enorm: Zu viele, zu feingranulare Services aus einem System zu extrahieren, dessen Datenkopplung nie gut verstanden wurde, neigt dazu, einen langsamen, aber verständlichen Monolithen durch ein schnell scheiterndes Netz synchroner Serviceaufrufe zu ersetzen, das erheblich schwerer zu durchdenken und zu betreiben ist. Der realistische Legacy-Modernisierungspfad bevorzugt daher gröbere, geschäftsausgerichtete Services, die schrittweise extrahiert und einzeln validiert werden, statt einer Big-Bang-Zerlegung in so viele Services, wie das Domänenmodell theoretisch unterstützen könnte.

## How to Apply ◆

> Die Zerlegung eines Legacy-Monolithen in Microservices ist eine der häufigsten — und am häufigsten verpfuschten — Modernisierungsstrategien. Erfolg erfordert sorgfältige Grenzidentifikation und inkrementelle Extraktion.

- Identifizieren Sie natürliche Servicegrenzen, indem Sie das Domänenmodell des Legacy-Systems analysieren und nach Bereichen mit minimalem Datenaustausch und unabhängigen Geschäftslebenszyklen suchen.
- Nutzen Sie das Strangler-Fig-Muster, um Services inkrementell zu extrahieren, statt den Monolithen von Grund auf neu zu schreiben — leiten Sie spezifische Funktionalität zu neuen Services, während der Monolith weiterhin alles andere handhabt.
- Beginnen Sie mit dem am wenigsten gekoppelten, am besten verstandenen Teil des Systems, um Teamerfahrung aufzubauen, bevor Sie sich an Kerngeschäftslogik wagen.
- Etablieren Sie von Anfang an klare API-Verträge zwischen Services, mit Contract Testing, um Integrationsausfälle zu verhindern, während die Zahl der Services wächst.
- Implementieren Sie Service-Level-Observability (verteiltes Tracing, zentralisiertes Logging, Health Checks), bevor Sie den ersten Service extrahieren, weil das Debugging verteilter Systeme ohne Observability erheblich schwerer ist als das Debugging eines Monolithen.
- Widerstehen Sie dem Drang, feingranulare Services zu erstellen — in Legacy-Kontexten sind größere, an Geschäftsfähigkeiten ausgerichtete Services meist handhabbarer als Dutzende winziger Services.
- Planen Sie Dateneigentümerschaft sorgfältig: Jeder Service sollte seinen eigenen Datenspeicher besitzen, und gemeinsam genutzte Datenbanken müssen durch explizite Datensynchronisation oder event-getriebene Ansätze beseitigt werden.

## Tradeoffs ⇄

> Microservices tauschen Monolith-Komplexität gegen Komplexität verteilter Systeme — der Nettonutzen hängt davon ab, ob das Team die Infrastruktur und Fähigkeiten hat, letztere zu verwalten.

**Vorteile:**

- Ermöglicht unabhängiges Deployment von Services, was Teams erlaubt, Änderungen an einem Teil des Systems auszuliefern, ohne sich mit jedem anderen Team abzustimmen.
- Erlaubt verschiedenen Teilen des Systems, unabhängig basierend auf tatsächlicher Nachfrage zu skalieren, statt den gesamten Monolithen zu skalieren.
- Bietet natürliche Teamgrenzen, ausgerichtet an Geschäftsfähigkeiten, was Koordinationsaufwand verringert.
- Ermöglicht inkrementelle Technologiemodernisierung — einzelne Services können neu geschrieben oder aktualisiert werden, ohne den Rest des Systems zu beeinflussen.

**Kosten und Risiken:**

- Führt Komplexität verteilter Systeme ein, einschließlich Netzwerkausfälle, eventuelle Konsistenz und Debugging-Herausforderungen, die Monolithen nicht haben.
- Erfordert erhebliche Infrastrukturinvestition in Service Discovery, API-Gateways, Container-Orchestrierung und Monitoring.
- Vorzeitige Zerlegung eines schlecht verstandenen Legacy-Systems erzeugt oft verteilte Monolithen, die schwerer zu pflegen sind als das Original.
- Datenkonsistenz über Servicegrenzen hinweg erfordert sorgfältiges Design und führt oft eventuelle-Konsistenz-Muster ein, mit denen das Team möglicherweise nicht erfahren ist.
- Der betriebliche Overhead steigt erheblich — jeder Service braucht seine eigene Deployment-Pipeline, Monitoring und Bereitschaftsrotation.

## How It Could Be

> Die folgenden Szenarien veranschaulichen sowohl erfolgreiche als auch warnende Microservices-Einführung in Legacy-Kontexten.

Ein Logistikunternehmen mit einer 12 Jahre alten monolithischen Sendungsverfolgungsanwendung begann seine Zerlegung, indem es das Benachrichtigungssubsystem in einen eigenständigen Service extrahierte. Dies war ein natürlicher erster Kandidat, weil Benachrichtigungen eine klare Schnittstelle hatten (Sendungsereignisse rein, Nachrichten raus) und minimalen gemeinsamen Zustand mit dem Rest des Systems. Die Extraktion dauerte sechs Wochen und gab dem Team Erfahrung mit Service-Deployment, Interservice-Kommunikation und verteiltem Tracing. Über die folgenden 18 Monate extrahierte das Team vier weitere Services, wobei jedes Mal Lektionen aus der vorherigen Extraktion angewendet wurden.

Ein Einzelhandelsunternehmen versuchte, seinen Auftragsverwaltungsmonolithen in einem einzigen sechsmonatigen Projekt in 30 Microservices zu zerlegen. Das Team unterschätzte die Datenkopplung zwischen Komponenten und endete mit Services, die synchrone Aufrufe in langen Ketten aneinander machten, was kaskadierende Ausfallszenarien erzeugte, die weit schlimmer waren als alles, was der Monolith je erlebt hatte. Nach einer Reihe von Produktionsausfällen konsolidierte das Team zurück zu acht gröber granularen, an Geschäftsdomänen ausgerichteten Services, was sich als weit handhabbarer erwies.
