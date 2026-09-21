---
title: Elastische Skalierung
description: Dynamische Anpassung der Ressourcenzuweisung an die aktuelle Last.
category:
- Performance
- Operations
quality_tactics_url: https://qualitytactics.de/en/performance-efficiency/elastic-scaling/
problems:
- scaling-inefficiencies
- capacity-mismatch
- insufficient-worker-capacity
- growing-task-queues
- task-queues-backing-up
- work-queue-buildup
- resource-contention
- service-discovery-failures
- high-connection-count
- thread-pool-exhaustion
- virtual-memory-thrashing
- load-balancing-problems
- memory-swapping
layout: solution
lang: de
en_slug: elastic-scaling
related_solutions:
- slug: elastic-resource-utilization
  similarity: 0.8
- slug: horizontal-scaling
  similarity: 0.75
- slug: capacity-planning
  similarity: 0.75
- slug: backpressure
  similarity: 0.7
- slug: resource-pooling
  similarity: 0.7
- slug: resource-usage-optimization
  similarity: 0.7
---

## Description

Elastische Skalierung passt die Anzahl laufender Instanzen oder Worker dynamisch basierend auf tatsächlich gemessener Nachfrage an und ersetzt die feste Ressourcenzuweisung, mit der die meisten Legacy-Systeme einmal, zum Installationszeitpunkt, ausgestattet und seither nie überarbeitet wurden. Diese statische Bereitstellung ist das, was über die Lebensdauer eines Systems beide Versagensmodi gleichzeitig verursacht: verschwendete Kapazität während ruhiger Perioden und regelrechte Sättigung während Lastspitzen, die die ursprüngliche Dimensionierung nie antizipierte. Echte Nachfragesignale zu instrumentieren — Warteschlangentiefe, Verbindungsanzahl, Auslastung — und zustandslose Komponenten zuerst gegen diese Signale zu skalieren lässt Infrastruktur tatsächliche Last verfolgen, statt eine jahrealte Vermutung, obgleich echt legacy Komponenten mit fest codierter Konfiguration oder In-Memory-Zustand meist erst skalierungsbereit gemacht werden müssen, bevor elastische Skalierung ihnen überhaupt helfen kann.

## How to Apply ◆

> Legacy-Systeme werden typischerweise mit festen, zum Installationszeitpunkt bestimmten Ressourcenzuweisungen deployt und selten überarbeitet. Elastische Skalierung ersetzt statische Bereitstellung durch dynamische Ressourcenanpassung, die Infrastrukturkapazität an tatsächliche Nachfrage anpasst und sowohl Überprovisionierungsverschwendung als auch Unterprovisionierungsausfälle verhindert.

- Instrumentieren Sie die Anwendung mit Kennzahlen, die tatsächliche Nachfrage widerspiegeln: Anfragerate, Warteschlangentiefe, aktive Verbindungsanzahl, CPU-Auslastung, Speichernutzung und Worker-Thread-Auslastung. Diese Kennzahlen bilden die Eingangssignale für Skalierungsentscheidungen und müssen mit ausreichender Granularität (typischerweise 1-Minuten-Intervalle) erfasst werden, um Nachfrageänderungen schnell zu erkennen.
- Definieren Sie Skalierungsauslöser basierend auf Auslastungsschwellenwerten und Warteschlangenwachstumsraten. Skalieren Sie für Worker-Pools hoch, wenn die durchschnittliche Warteschlangentiefe einen anhaltenden Schwellenwert überschreitet (z. B. 5 aufeinanderfolgende Minuten wächst), und skalieren Sie herunter, wenn Worker über einen längeren Zeitraum untätig sind. Vermeiden Sie Momentaufnahme-Messungen, die Oszillation zwischen Hoch- und Runterskalierung verursachen.
- Implementieren Sie horizontale Skalierung zuerst für zustandslose Komponenten — Webserver, API-Gateways und Hintergrund-Worker —, weil diese ohne Koordination hinzugefügt und entfernt werden können. Legacy-Systeme mit zustandsbehafteten Komponenten erfordern zusätzliche Muster (Session Affinity, verteilte Caches, Shared-Nothing-Architekturen), bevor horizontale Skalierung machbar wird.
- Nutzen Sie Container-Orchestrierungsplattformen (Kubernetes, ECS, Docker Swarm), um Instanzskalierung basierend auf Kennzahlenschwellenwerten zu automatisieren. Für noch nicht containerisierte Legacy-Anwendungen können Cloud-Anbieter-Auto-Scaling-Gruppen (AWS Auto Scaling, Azure VM Scale Sets) VM-Instanzen basierend auf benutzerdefinierten CloudWatch- oder Azure-Monitor-Kennzahlen skalieren.
- Implementieren Sie Service Discovery, um sicherzustellen, dass skalierte Instanzen automatisch registriert und von Load Balancern und abhängigen Services auffindbar sind. Nutzen Sie DNS-basierte Discovery, Service-Registries (Consul, Eureka) oder plattformnative Discovery (Kubernetes Services), um fest codierte Service-Adressen zu vermeiden, die Skalierung verhindern.
- Skalieren Sie Datenbankverbindungen proportional zu Anwendungsinstanzen. Verifizieren Sie beim Hinzufügen von Anwendungsinstanzen, dass der gesamte Verbindungsbedarf über alle Instanzen hinweg das Verbindungslimit der Datenbank nicht überschreitet. Nutzen Sie Connection Pooler wie PgBouncer oder ProxySQL als Zwischenschicht, die viele Anwendungsverbindungen über weniger Datenbankverbindungen multiplext.
- Implementieren Sie Abklingzeiten nach Skalierungsereignissen, um Thrashing zu verhindern: Warten Sie nach dem Hochskalieren mindestens 3-5 Minuten, bevor Sie bewerten, ob heruntergeskaliert werden soll, damit die neu hinzugefügte Kapazität die Last absorbieren und sich Kennzahlen stabilisieren können.
- Gestalten Sie Worker-Skalierung warteschlangenbewusst: Worker sollten basierend auf Warteschlangentiefe und Verarbeitungslatenz statt allein auf CPU skalieren, weil I/O-gebundene Worker niedrige CPU-Auslastung haben könnten, während sie vollständig damit beschäftigt sind, auf externe Services zu warten.
- Testen Sie das Skalierungsverhalten unter realistischen Lastbedingungen, bevor Sie sich in Produktion darauf verlassen. Simulieren Sie Nachfragespitzen und verifizieren Sie, dass neue Instanzen starten, sich bei der Service Discovery registrieren und innerhalb des erforderlichen Zeitrahmens beginnen, Anfragen zu verarbeiten. Verifizieren Sie auch, dass Herunterskalieren keine laufenden Anfragen verwirft.

## Tradeoffs ⇄

> Elastische Skalierung verhindert die Verschwendung statischer Überprovisionierung und die Ausfälle der Unterprovisionierung, erfordert aber Investition in Automatisierung, Überwachung und Infrastruktur, die dynamische Ressourcenanpassung unterstützt.

**Vorteile:**

- Passt Infrastrukturkapazität an tatsächliche Nachfrage an, was sowohl die Kostenverschwendung ungenutzter Ressourcen während Zeiten geringen Traffics als auch die Performance-Ausfälle unzureichender Ressourcen während Spitzen eliminiert.
- Handhabt unvorhersehbare Lastspitzen automatisch, was den Bedarf an manuellem Eingriff reduziert und dem System erlaubt, Traffic-Anstiege von Marketingkampagnen, saisonalen Ereignissen oder viralem Wachstum zu absorbieren.
- Reduziert Warteschlangenaufbau und Verarbeitungsverzögerungen, indem Worker hinzugefügt werden, wenn Aufgabenvolumina steigen, was die Verarbeitungslatenz innerhalb akzeptabler Grenzen hält, ohne dauerhafte Überprovisionierung.
- Verbessert die Systemresilienz, indem ausgefallene Instanzen automatisch ersetzt werden, was die Auswirkung einzelner Instanzausfälle auf die Gesamtsystemverfügbarkeit reduziert.
- Bietet einen kosteneffektiven Weg, wachsende Workloads zu handhaben, ohne sich auf dauerhafte Infrastrukturinvestitionen basierend auf Spitzenbedarfsprognosen festzulegen, die sich möglicherweise nicht materialisieren.

**Kosten und Risiken:**

- Legacy-Anwendungen mit fest codierten Konfigurationen, lokalen Dateiabhängigkeiten oder zustandsbehafteten In-Memory-Daten können ohne Refactoring nicht horizontal skaliert werden — die Anwendung muss skalierungsbereit gemacht werden, bevor elastische Skalierung Wert liefert.
- Auto-Scaling basierend auf falschen Kennzahlen oder Schwellenwerten kann Skalierungsstürme (schnelle Oszillation zwischen Hoch- und Runterskalierung) verursachen, die das System destabilisieren und Kosten erhöhen.
- Jede neue Anwendungsinstanz erzeugt zusätzlichen Datenbankverbindungsbedarf; ohne Connection-Pooling-Zwischenschichten kann die Skalierung der Anwendungsebene die Datenbankebene überwältigen.
- Service-Discovery-Ausfälle während Skalierungsereignissen können dazu führen, dass Traffic an Instanzen geroutet wird, die noch nicht bereit sind oder bereits terminiert wurden, was vorübergehende Fehler erzeugt.
- Cold-Start-Latenz für neue Instanzen (JVM-Aufwärmen, Cache-Befüllung, Verbindungsaufbau) bedeutet, dass kürzlich skalierte Instanzen anfangs mit reduzierter Kapazität operieren, und Skalierungsentscheidungen müssen diese Aufwärmphase berücksichtigen.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie elastische Skalierung Kapazitäts- und Warteschlangenverwaltungsprobleme in Legacy-Systemen adressiert.

Ein Steuervorbereitungsservice erlebt extreme saisonale Nachfrage: Der Traffic steigt in den zwei Monaten vor der Steuererklärungsfrist um das 20-Fache und fällt für die restlichen zehn Monate auf eine Baseline. Das Legacy-System war für Spitzenkapazität bereitgestellt und kostete ganzjährig 45.000 Dollar pro Monat an Cloud-Infrastruktur. Das Team containerisierte die Anwendung, deployte sie auf Kubernetes mit einem Horizontal Pod Autoscaler, konfiguriert zur Skalierung basierend auf Anfragerate und CPU-Auslastung, und implementierte einen PgBouncer-Connection-Pooler zur Verwaltung von Datenbankverbindungen. Außerhalb der Saison läuft das System auf 3 Pods für 4.500 Dollar pro Monat. Während der Steuersaison skaliert es auf 40 Pods, um die Last zu handhaben, mit einem Spitzenwert von 30.000 Dollar pro Monat. Die jährlichen Infrastrukturkosten sanken von 540.000 auf 135.000 Dollar, während sich die Performance in der Spitzensaison tatsächlich verbesserte, weil der Autoscaler schneller auf Nachfrage reagierte als der vorherige manuelle Skalierungsprozess.

Das Sendungsverfolgungssystem eines Logistikunternehmens verarbeitet Events von Zustellfahrern über eine Nachrichtenwarteschlange mit einem festen Pool von 8 Workern. Während der Weihnachtsversandsaison verdreifacht sich das Event-Volumen, und die Warteschlange wächst auf 200.000 ausstehende Events, was Verfolgungsaktualisierungen um 6 Stunden verzögert. Das Team implementierte warteschlangentiefenbasiertes Auto-Scaling: Wenn die durchschnittliche Warteschlangentiefe 1.000 für 3 aufeinanderfolgende Minuten überschreitet, startet eine neue Worker-Instanz; wenn die Warteschlangentiefe für 10 Minuten unter 100 fällt, werden überschüssige Worker terminiert. Während der nächsten Weihnachtssaison skalierte der Worker-Pool über 30 Minuten von 8 auf 24 Instanzen, während das Event-Volumen anstieg, wodurch die Warteschlangentiefe unter 500 und die Verarbeitungslatenz unter 2 Minuten blieb. Nachdem die Spitze abgeklungen war, skalierten Worker innerhalb einer Stunde zurück, und das Team musste nicht mehr manuell zusätzliche Kapazität in Erwartung saisonaler Nachfrage bereitstellen.

Ein SaaS-Unternehmen entdeckte, dass ihr Service-Discovery-Mechanismus (Consul) während Skalierungsereignissen intermittierend versagte, weil neue Instanzen sich registrierten, bevor sie bereit waren, Traffic zu bedienen, und terminierte Instanzen nicht prompt deregistriert wurden. Dies führte dazu, dass Load Balancer Anfragen an ungesunde Endpunkte routeten, was nach jedem Skalierungsereignis eine Flut von 500-Fehlern erzeugte. Das Team implementierte Health-Check-Endpunkte, die erst als gesund meldeten, nachdem die Anwendung die Initialisierung abgeschlossen hatte (Datenbank-Connection-Pool aufgewärmt, Caches befüllt, Health Checks bestanden für 30 Sekunden). Sie konfigurierten auch graziöses Herunterfahren, um sich von Consul zu deregistrieren und laufende Anfragen abzuschließen, bevor terminiert wird. Nach diesen Änderungen wurden Skalierungsereignisse für Nutzer transparent — die Fehlerrate während der Skalierung sank von 2 Prozent auf null, und das Betriebsteam gewann Vertrauen, aggressivere Auto-Scaling-Richtlinien zuzulassen.
