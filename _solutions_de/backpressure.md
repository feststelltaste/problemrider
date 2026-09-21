---
title: Backpressure
description: Signalisierung an Produzenten, langsamer zu werden, wenn Konsumenten
  überlastet werden.
category:
- Performance
- Architecture
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/backpressure/
problems:
- growing-task-queues
- task-queues-backing-up
- work-queue-buildup
- insufficient-worker-capacity
- thread-pool-exhaustion
- resource-contention
- virtual-memory-thrashing
- memory-swapping
- high-connection-count
- cascade-failures
layout: solution
lang: de
en_slug: backpressure
related_solutions:
- slug: rate-limiting
  similarity: 0.75
- slug: timeout-management
  similarity: 0.7
- slug: capacity-planning
  similarity: 0.7
- slug: elastic-scaling
  similarity: 0.7
- slug: resource-usage-optimization
  similarity: 0.7
- slug: batch-processing
  similarity: 0.7
---

## Description

Backpressure gibt einem Konsumenten eine explizite Möglichkeit, einem überlasteten Produzenten zu signalisieren, langsamer zu werden, statt den Produzenten weiter Arbeit mit voller Geschwindigkeit erzeugen zu lassen, während eine Warteschlange unbegrenzt wächst. Legacy-Systemen fehlt diese Rückkopplungsschleife sehr häufig vollständig — Warteschlangen haben keine Maximalgröße, Produzenten haben keine Möglichkeit zu erfahren, dass ein nachgelagerter Konsument zurückgefallen ist —, sodass eine Lastspitze in unbegrenztes Speicherwachstum und schließlich einen kaskadierenden Ausfall übergeht. Die Einführung begrenzter Warteschlangen, Rate Limiting und Flusssteuerungsprotokolle an jeder Produzenten-Konsumenten-Grenze verwandelt diesen Fehlermodus in vorhersagbare, bewusste Degradation: überschüssige Arbeit wird gedrosselt, verzögert oder abgelehnt, statt akzeptiert zu werden und sich still anzuhäufen.

## How to Apply ◆

> Legacy-Systemen fehlt häufig jeglicher Mechanismus, mit dem nachgelagerte Komponenten vorgelagerten Produzenten signalisieren können, dass sie überlastet sind. Ohne Backpressure erzeugen Produzenten weiterhin Arbeit mit voller Geschwindigkeit, während Konsumenten weiter zurückfallen, was zu Warteschlangenaufbau, Speichererschöpfung und kaskadierenden Ausfällen führt. Die Einführung von Backpressure schafft eine Rückkopplungsschleife, die das System in einem nachhaltigen Betriebsbereich hält.

- Identifizieren Sie alle Produzenten-Konsumenten-Grenzen im System: Nachrichtenwarteschlangen, API-Endpunkte, die Hintergrundprozessoren speisen, Batch-Job-Pipelines und jeden Punkt, an dem Arbeit schneller erzeugt wird, als sie konsumiert werden kann. Diese Grenzen sind, wo Backpressure-Mechanismen angewendet werden müssen.
- Führen Sie begrenzte Warteschlangen an jeder Produzenten-Konsumenten-Grenze ein. Ersetzen Sie unbegrenzte Warteschlangen (die wachsen, bis der Speicher erschöpft ist) durch Warteschlangen mit expliziten Maximalgrößen. Wenn eine Warteschlange die Kapazität erreicht, muss der Produzent entweder blockieren, die Nachricht verwerfen oder einen Fehler erhalten — alles ist gegenüber stillem, unbegrenztem Wachstum vorzuziehen.
- Implementieren Sie Rate Limiting an API-Endpunkten und Eingangspunkten, um die Rate zu begrenzen, mit der neue Arbeit in das System eintritt. Nutzen Sie Token-Bucket- oder Sliding-Window-Algorithmen, um nachhaltige Durchsatzgrenzen basierend auf gemessener Konsumentenkapazität statt willkürlicher Schwellen durchzusetzen.
- Fügen Sie Warteschlangentiefen-Monitoring mit produzentenseitiger Drosselung hinzu: Wenn die Warteschlangentiefe eine konfigurierbare Schwelle überschreitet (typischerweise 70-80 % der Kapazität), verringern Sie die Ausgaberate des Produzenten. Dies kann als einfache Rückkopplungsschleife implementiert werden, bei der der Produzent die Warteschlangentiefe abfragt oder Warteschlangenmetriken abonniert und seine Senderate entsprechend anpasst.
- Nutzen Sie Reactive Streams oder Flusssteuerungsprotokolle (wie TCP-Flusssteuerung, gRPC-Flusssteuerung, Reactive Streams oder Kafka-Consumer-Group-Lag-basierte Drosselung), die eingebaute Backpressure-Semantik haben, statt benutzerdefinierte Lösungen zu implementieren. Diese Protokolle handhaben die komplexe Koordination zwischen Produzenten und Konsumenten automatisch.
- Implementieren Sie Circuit Breaker an Servicegrenzen, sodass, wenn ein nachgelagerter Service überlastet wird, vorgelagerte Aufrufer schnelle Fehlerantworten erhalten, statt blockierte Threads anzuhäufen, die auf den überlasteten Service warten. Der Circuit Breaker löst aus, wenn Fehler- oder Latenzschwellen überschritten werden, und setzt sich nach einer Abkühlphase zurück.
- Designen Sie Batch-Verarbeitungssysteme so, dass sie Arbeit abrufen, statt dass ihnen Arbeit zugeschoben wird. Pull-basierte Architekturen implementieren Backpressure natürlich, weil Worker nur dann neue Aufgaben anfragen, wenn sie Kapazität haben, was verhindert, dass sich Arbeit schneller anhäuft, als sie verarbeitet werden kann.
- Fügen Sie Load Shedding als letzten Backpressure-Mechanismus hinzu: Wenn das System kritisch überlastet ist, lehnen Sie selektiv niedrigpriore Anfragen ab oder deprioritisieren Sie sie, um Kapazität für kritische Operationen zu erhalten. Dokumentieren Sie, welche Operationen für Shedding infrage kommen, und kommunizieren Sie Ablehnung klar an Aufrufer, sodass sie später erneut versuchen können.
- Testen Sie Backpressure-Mechanismen unter realistischen Überlastbedingungen. Simulieren Sie Szenarien, in denen Produzenten das 2-5-fache des nachhaltigen Durchsatzes erzeugen, und verifizieren Sie, dass sich das System elegant degradiert — überschüssige Arbeit verlangsamend oder ablehnend —, statt allen verfügbaren Speicher zu verbrauchen, abzustürzen oder in einen Thrashing-Zustand zu geraten.

## Tradeoffs ⇄

> Backpressure verhindert katastrophale Ausfälle unter Überlastung, indem das System innerhalb seines nachhaltigen Betriebsbereichs gehalten wird, bedeutet aber, dass überschüssige Arbeit explizit abgelehnt, verzögert oder gedrosselt wird, statt still akzeptiert zu werden.

**Vorteile:**

- Verhindert Warteschlangenaufbau und eventuelle Speichererschöpfung, indem unbegrenzte Arbeitsanhäufung gestoppt wird, bevor sie Systemressourcen überlastet.
- Verwandelt unvorhersehbare Systemausfälle unter Überlastung in vorhersagbare, handhabbare Degradation, bei der überschüssige Anfragen klare Ablehnungssignale erhalten.
- Schützt nachgelagerte Services davor, von vorgelagerten Traffic-Spitzen überwältigt zu werden, und verhindert kaskadierende Ausfälle über verteilte Systeme hinweg.
- Ermöglicht dem System, konsistente Antwortzeiten für akzeptierte Arbeit selbst unter hoher Last aufrechtzuerhalten, statt die Performance für alle Anfragen gleichermaßen zu degradieren.
- Bietet klare operative Signale über Systemkapazitätsgrenzen, was es einfacher macht zu bestimmen, wann und um wie viel Skalierung nötig ist.

**Kosten und Risiken:**

- Aufrufer müssen so designt sein, dass sie Ablehnungs- oder Drosselungssignale handhaben, was Änderungen an vorgelagerten Systemen erfordert, die möglicherweise keinen Widerstand von nachgelagerten Services erwarten oder handhaben.
- Falsch konfigurierte Backpressure-Schwellen könnten Arbeit vorzeitig ablehnen, wenn noch Kapazität verfügbar ist, was den Systemdurchsatz während normalen Betriebs unnötig verringert.
- Begrenzte Warteschlangen bedeuten, dass während legitimer Traffic-Spitzen manche Arbeit abgelehnt oder verzögert werden könnte, obwohl sie schließlich verarbeitet würde — dies ist ein bewusster Tradeoff von Latenz und Verfügbarkeit gegen Durchsatz.
- Die Implementierung von Backpressure über Servicegrenzen in einem Legacy-System mit vielen Integrationspunkten erfordert koordinierte Änderungen über mehrere Komponenten hinweg, was schwierig ist, wenn verschiedene Teams verschiedene Services besitzen.
- Load-Shedding-Entscheidungen erfordern klare Geschäftsregeln darüber, welche Operationen entbehrlich und welche kritisch sind, und diese Prioritäten falsch zu setzen kann mehr Schaden anrichten als die Überlastung selbst.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie Backpressure-Mechanismen Systemausfälle in Legacy-Systemen unter Überlastung verhindern.

Ein Auftragsverarbeitungssystem nutzt eine RabbitMQ-Warteschlange, um eingehende Bestellungen von einem E-Commerce-Frontend zu handhaben. Während Flash-Sales springt das Bestellvolumen auf das 10-fache des normalen Niveaus, und die Warteschlange wächst auf Millionen von Nachrichten, was den gesamten verfügbaren Speicher auf dem Nachrichtenbroker verbraucht und ihn zum Absturz bringt. Das Team konfiguriert die Warteschlange mit einer Maximallänge von 50.000 Nachrichten und einem Dead-Letter-Exchange für Overflow. Wenn die Warteschlange die Kapazität erreicht, werden neue Nachrichten an den Dead-Letter-Exchange weitergeleitet, wo sie auf die Festplatte persistiert und in verkehrsarmen Zeiten erneut versucht werden. Zusätzlich implementiert das Frontend clientseitiges Rate Limiting, das ein „hohe Nachfrage"-Wartezimmer anzeigt, wenn die API 429-Antworten (Too Many Requests) zurückgibt. Beim nächsten Flash-Sale bleibt die Warteschlange innerhalb der Grenzen, der Broker bleibt stabil, und alle Bestellungen werden schließlich verarbeitet — Kunden mit hoher Priorität werden sofort verarbeitet, während Overflow-Bestellungen innerhalb von 2 Stunden abgeschlossen werden.

Eine Datenaufnahme-Pipeline empfängt Sensordaten von 10.000 IoT-Geräten und verarbeitet sie durch eine Reihe von Transformationsstufen. Während die Geräteflotte wuchs, wurde die zweite Stufe (Datenanreicherung) zu einem Engpass, was dazu führte, dass die erste Stufe zunehmend große Mengen roher Daten im Speicher puffern musste. Schließlich erschöpfte die Pufferung den verfügbaren RAM und löste virtuelles Speicher-Thrashing aus, das die gesamte Pipeline unbrauchbar machte. Das Team redesignte die Pipeline mit einer Pull-basierten Architektur, bei der jede Stufe Arbeit von der vorherigen Stufe nur anfragt, wenn sie Verarbeitungskapazität hat. Wenn die Anreicherungsstufe zurückfällt, verlangsamt die Aufnahmestufe automatisch ihre Annahme neuer Sensordaten und sendet Backpressure-Signale an das IoT-Gateway. Das Gateway reagiert, indem es sein Batching-Intervall erhöht, was die Nachrichtenrate pro Sekunde verringert, während sichergestellt wird, dass keine Daten verloren gehen. Der Speicherverbrauch stabilisierte sich bei 2 GB statt unbegrenzt zu wachsen, und die Pipeline handhabt anhaltende Überlastung ohne Degradation, indem sie die Aufnahmerate sanft drosselt.

Eine Legacy-Banking-Anwendung verarbeitet Überweisungen durch einen warteschlangenbasierten Workflow. Am Monatsende reichen Firmenkunden Tausende von Massentransferdateien gleichzeitig ein, was die Transferverarbeitungs-Worker überwältigt. Ohne Backpressure wächst die Warteschlange auf 500.000 ausstehende Transfers, und Worker beginnen zu thrashen, während sie um Datenbankverbindungen und externe Validierungsservice-Kapazität konkurrieren. Das Team implementiert ein gestuftes Backpressure-System: Das API-Gateway erzwingt eine Einreichungsrate von 100 Transfers pro Minute pro Kunde, die Warteschlange ist auf 10.000 Einträge begrenzt, mit Overflow, der zu einem sekundären persistenten Speicher weitergeleitet wird, und Worker implementieren Circuit Breaker beim externen Validierungsservice mit einem 5-Sekunden-Timeout und 30-Sekunden-Abkühlphase. Bei der nächsten Monatsend-Spitze verarbeitet das System Transfers mit einer stetigen Rate von 2.000 pro Minute, Kunden erhalten klares Feedback über Einreichungsratenlimits, und alle Transfers werden innerhalb desselben Geschäftstags abgeschlossen — verglichen mit dem vorherigen Monat, in dem das System abstürzte und 3 Tage manueller Wiederherstellung erforderte.
