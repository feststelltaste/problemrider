---
title: Event-Driven Architecture
description: Entkopplung von Komponenten durch asynchrone Events.
category:
- Architecture
quality_tactics_url: https://qualitytactics.de/en/maintainability/event-driven-architecture/
problems:
- tight-coupling-issues
- deployment-coupling
- high-coupling-low-cohesion
- cascade-failures
- monolithic-architecture-constraints
- single-points-of-failure
- circular-dependency-problems
- bottleneck-formation
- load-balancing-problems
- service-timeouts
- upstream-timeouts
- cache-invalidation-problems
- synchronization-problems
layout: solution
lang: de
en_slug: event-driven-architecture
related_solutions:
- slug: event-driven-integration
  similarity: 0.8
- slug: strangler-fig-pattern
  similarity: 0.8
- slug: business-event-processing
  similarity: 0.75
- slug: architecture-decision-records
  similarity: 0.75
- slug: modularization-and-bounded-contexts
  similarity: 0.75
- slug: microservices
  similarity: 0.75
---

## Description

Event-Driven Architecture entkoppelt Komponenten, indem sie asynchrone Events — „Bestellung aufgegeben", „Zahlung erhalten" — über einen Broker veröffentlichen und darauf reagieren, statt sich gegenseitig direkt und synchron entlang einer eng verwobenen Kette aufzurufen. In einem Legacy-Kontext ist dies am wertvollsten als Nahtstelle: Es erlaubt neuen Komponenten, Events zu abonnieren, die ein Legacy-System bereits erzeugt (oder mittels einer Change-Data-Capture-Brücke erzeugen kann), ohne gleichzeitige, koordinierte Änderungen über die Aufrufketten des alten Systems zu erfordern, und es verhindert, dass eine langsame oder nicht verfügbare Legacy-Komponente den gesamten Anfragepfad für alles Nachgelagerte blockiert. Der Tradeoff ist, dass das Debuggen eines Prozesses, der einen Legacy-Publisher, einen Broker und mehrere Konsumenten umspannt, keinen einzelnen Call Stack zum Inspizieren hat, und Legacy-Systeme selten nativ Events veröffentlichen, sodass das Nachrüsten dieses Musters meist bedeutet, Outbox-Tabellen oder Polling-Brücken hinzuzufügen, die ihre eigene Fragilität tragen.

## How to Apply ◆

> In einem Legacy-Kontext ist Event-Driven Architecture am wertvollsten als Entkopplungsmechanismus, der neuen und alten Komponenten erlaubt, sich unabhängig weiterzuentwickeln, ohne gleichzeitige Änderungen über eng verwobene Aufrufketten zu erfordern.

- Bilden Sie die synchronen Aufrufketten im bestehenden Legacy-System ab und identifizieren Sie die Verbindungen, an denen eine Komponente eine andere unnötig blockiert — dies sind die ersten Kandidaten für den Ersatz durch asynchrone Events.
- Führen Sie einen Event-Broker (wie Kafka oder RabbitMQ) als Nahtstelle zwischen dem Legacy-System und neuen Komponenten ein, sodass Legacy-Komponenten Events veröffentlichen können, ohne zu wissen, wer sie konsumiert.
- Definieren Sie Domain Events, die bedeutungsvolle Geschäftstatsachen erfassen — „Bestellung aufgegeben", „Zahlung erhalten", „Bestand reserviert" —, mittels Sprache aus der Geschäftsdomäne statt der internen Terminologie des Legacy-Systems, und behandeln Sie diese Events als den stabilen öffentlichen Vertrag zwischen alten und neuen Teilen.
- Gestalten Sie alle neuen Event-Konsumenten von Anfang an idempotent; Legacy-Systeme haben oft Retry-Logik und Batch-Replays, die dasselbe Event mehr als einmal liefern werden, und nicht-idempotente Konsumenten werden Daten still korrumpieren.
- Nutzen Sie Dead Letter Queues für jeden Konsumenten und überwachen Sie deren Tiefe; von Legacy erzeugte Events enthalten häufig unerwartete Formate oder fehlende Felder, die Konsumenten brechen, und stilles Warteschlangen-Blockieren ist ein häufiger Fehlermodus während der Modernisierung.
- Wenden Sie Saga-Muster an, um verteilte Transaktionen zu ersetzen, die Legacy- und moderne Komponenten überspannen; modellieren Sie statt eines Two-Phase-Commits über ein altes RDBMS und einen neuen Service den Workflow als eine Sequenz kompensierender Events.
- Vermeiden Sie ereignisgetriebene Kommunikation für Interaktionen, die echt sofortige Antworten vom Legacy-Backend erfordern — Autorisierungsprüfungen und Echtzeit-Bestandsabfragen gehören auf synchrone Aufrufe; Hintergrundverarbeitung und Benachrichtigungen nicht.
- Versionieren Sie Event-Schemata von Anfang an explizit; sobald andere Teams oder neue Services einen Event-Stream abonnieren, ist das Brechen seines Formats ebenso störend wie das Brechen einer öffentlichen REST-API.

## Tradeoffs ⇄

> Event-Driven Architecture ist ein mächtiges Werkzeug, um den Griff zu lockern, den ein monolithisches Legacy-System auf die es umgebenden Komponenten hat, aber es verlagert Komplexität von Aufrufketten-Kopplung zu Event-Fluss-Verwaltung.

**Vorteile:**

- Bricht synchrone Aufrufketten, die kaskadierende Fehler im Legacy-System verursachen, sodass eine langsame oder nicht verfügbare Legacy-Komponente den gesamten nutzerseitigen Anfragepfad nicht mehr blockiert.
- Erlaubt neuen Services, dem Ökosystem hinzugefügt zu werden, ohne das Legacy-System zu modifizieren — sie abonnieren einfach bestehende Event-Topics.
- Absorbiert Lastdiskrepanzen zwischen einem schnellen Legacy-Producer und einem langsameren modernen Konsumenten durch den Puffer des Brokers, was den Bedarf reduziert, das Legacy-System zu drosseln.
- Schafft ein dauerhaftes Event-Log, das als Grundlage für Audit-Trails und Datenmigration dienen kann, beide häufig während der Legacy-Modernisierung benötigt.
- Ermöglicht unabhängiges Deployment neuer Konsumenten, ohne Releases mit dem Release-Zeitplan des Legacy-Systems koordinieren zu müssen.

**Kosten und Risiken:**

- Debugging ist erheblich schwerer, wenn ein Geschäftsprozess einen Legacy-Publisher, einen Broker und mehrere moderne Konsumenten überspannt, weil es keinen einzelnen Call Stack zum Inspizieren gibt.
- Eventual Consistency führt ein Fenster ein, während dessen das Legacy-System und neue Services unterschiedliche Sichten der Realität haben, was bewusste Handhabung erfordert und Teams, die an synchrone Legacy-Muster gewöhnt sind, oft unvertraut ist.
- Der Event-Broker selbst wird zu einer neuen operativen Abhängigkeit, die dimensioniert, überwacht und verfügbar gehalten werden muss — was Infrastrukturlast zu einer Zeit hinzufügt, in der Teams bereits Legacy-Infrastruktur verwalten.
- Legacy-Systeme wurden selten entworfen, um Events zu veröffentlichen; das Nachrüsten von Event-Veröffentlichung bedeutet oft, Outbox-Tabellen, logbasierte Change Data Capture oder Polling-Brücken hinzuzufügen, von denen jede ihre eigene Fragilität trägt.
- Event-Schemaänderungen in einem von Legacy erzeugten Feed sind schwer zu koordinieren, weil das Team, das das Legacy-System besitzt, möglicherweise nicht alle nachgelagerten Konsumenten kennt.

## How It Could Be

> Die folgenden Szenarien zeigen, wie Event-Driven Architecture genutzt wurde, um den Kopplungsdruck von Legacy-Systemen in echten Modernisierungsprogrammen zu lindern.

Ein nationaler Versicherungsanbieter betrieb die gesamte Schadensaufnahme, Zeichnung und Zahlung über eine einzige monolithische J2EE-Anwendung, deployt als eine WAR-Datei. Das Hinzufügen eines neuen Benachrichtigungskanals oder eines Betrugserkennungsschritts erforderte Änderungen an der Kernorchestrierungsschicht des Monolithen, was einen vollständigen Regressionszyklus und ein vierteljährliches Release auslöste. Das Team führte einen Kafka-Broker neben dem Monolithen ein und instrumentierte die Service-Schicht des Monolithen, um nach jeder erfolgreichen Aufnahme ein `ClaimSubmitted`-Event zu veröffentlichen. Neue Microservices für Betrugsprüfung und Kundenbenachrichtigungen abonnierten unabhängig, jeder nach eigenem Zeitplan deploybar. Der Release-Zyklus des Monolithen änderte sich nicht, aber neue Fähigkeiten konnten jetzt wöchentlich ausgeliefert werden.

Ein Fertigungsunternehmen integrierte ein dreißig Jahre altes ERP-System mit einem neuen Lagerverwaltungssystem. Das ERP exportierte alle vier Stunden Bestandsanpassungsdateien im Batch auf ein gemeinsames Netzlaufwerk; das Lagersystem pollte das Laufwerk und importierte die Dateien. Als der Dateiimport fehlschlug, hatte das Lagerteam keine Sichtbarkeit, bis Bestandszahlen weit genug divergierten, um Kommissionierfehler zu verursachen. Das Team ersetzte den Datei-Poll-Mechanismus durch eine Change-Data-Capture-Brücke, die das Datenbank-Transaktionsprotokoll des ERP las und Bestands-Events an RabbitMQ veröffentlichte. Das Lagersystem konsumierte die Warteschlange und verarbeitete Anpassungen nahezu in Echtzeit. Dead-Letter-Queue-Überwachung gab dem Betriebsteam sofortige Sichtbarkeit auf Importfehler, was den vorherigen vierstündigen blinden Fleck durch einen minutengenauen Alarm ersetzte.

Eine Regionalbank betrieb eine in den frühen 2000ern gebaute Kreditverwaltungsplattform, die Rückzahlungsverarbeitung synchron handhabte: Eingehende Rückzahlung löste Zahlungsverbuchung, Kontoauszugserstellung, Tilgungsplanneuberechnung und regulatorische Berichterstattung alle in einer einzigen Datenbanktransaktion aus. Während Transaktionsvolumina wuchsen, überschritt diese Kette routinemäßig Zeitlimits, was nutzerseitige Fehler verursachte und manuellen Abgleich erforderte. Das Modernisierungsteam zerlegte die Kette in Events: Das Legacy-System verbuchte die Zahlung und veröffentlichte ein `PaymentPosted`-Event. Kontoauszugserstellung, Planneuberechnung und regulatorische Berichterstattung wurden unabhängige Konsumenten. Die synchrone Transaktion schrumpfte auf den alleinigen Zahlungsverbuchungsschritt, was die Timeout-Häufigkeit dramatisch reduzierte, und die nachgelagerten Prozesse konnten unabhängig skaliert werden, um Spitzenvolumen am Quartalsende zu handhaben.
