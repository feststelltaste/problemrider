---
title: Cold-Start-Minderung
description: Proaktive Reduzierung der Initialisierungslatenz in Serverless-, Container-
  und JVM-Anwendungen.
category:
- Performance
- Operations
problems:
- slow-application-performance
- slow-response-times-for-lists
- external-service-delays
- gradual-performance-degradation
- service-timeouts
layout: solution
lang: de
en_slug: cold-start-mitigation
related_solutions:
- slug: lazy-loading
  similarity: 0.75
- slug: lazy-evaluation
  similarity: 0.75
- slug: connection-pooling
  similarity: 0.75
- slug: distributed-caching
  similarity: 0.7
- slug: serverless-computing
  similarity: 0.7
- slug: rate-limiting
  similarity: 0.7
---

## Description

Cold-Start-Minderung deckt eine Reihe von Techniken zur Verringerung der Latenz ab, die eine Anwendung in dem Moment erleidet, in dem sie aus einem untätigen oder neu bereitgestellten Zustand startet — die Verzögerung durch Klassenladen, Initialisierung des Dependency-Injection-Containers, JIT-Aufwärmung und eifrige Ressourceneinrichtung, die eine laufende, bereits aufgewärmte Instanz nicht zahlt. Es ist gleichermaßen in Serverless-Funktionen, Container-Plattformen und JVM-basierten Anwendungen wichtig, überall dort, wo neue Instanzen dynamisch als Reaktion auf Skalierungsereignisse oder nach Untätigkeitsperioden erstellt werden, da die ersten an eine frische Instanz gerouteten Anfragen Latenz weit über der Steady-State-Performance der Anwendung erleben. Dies ist ein bedeutsames Problem für Legacy-Anwendungen, besonders ältere JVM-basierte Systeme, die in containerisierte oder Auto-Scaling-Umgebungen verschoben wurden, für die sie nie designt waren: umfangreiches Classpath-Scanning, eifriges Bean-Laden und Schema-Validierung, die tolerierbar waren, als die Anwendung einmal startete und monatelang lief, werden zu einer wiederkehrenden Steuer jedes Mal, wenn eine neue Instanz hochfährt, und während Skalierungsereignissen können neue Instanzen Traffic empfangen, bevor die Initialisierung tatsächlich abgeschlossen ist, was kaskadierende Timeouts verursacht. Techniken wie Lazy Initialization nicht-kritischer Komponenten, bereitgestellte oder vorgewärmte Instanzen, kleinere Container-Images und Ahead-of-Time-Kompilierung greifen jeweils eine andere Quelle von Startlatenz an und werden üblicherweise kombiniert statt einzeln angewendet. Readiness Probes, die genuin auf vollständige Initialisierung warten, bevor sie Traffic akzeptieren, sind es, was den Skalierungsereignis-Fehlermodus spezifisch verhindert und die Lücke zwischen „Instanz existiert" und „Instanz ist tatsächlich bereit, Anfragen zu bedienen" schließt. Der Tradeoff ist, dass Vorwärmung und bereitgestellte Nebenläufigkeit echte Infrastrukturausgaben kosten, um Instanzen zu unterhalten, die sonst herunterskaliert wären, und Techniken wie Ahead-of-Time-Kompilierung unterstützen möglicherweise nicht jedes Laufzeitfeature — Reflection oder dynamische Proxys, zum Beispiel —, auf das sich älterer Legacy-Code verlässt.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Messen Sie Cold-Start-Zeiten, um Baselines zu etablieren und die größten Beitragenden zur Initialisierungslatenz zu identifizieren
- Verringern Sie die Startzeit des Dependency-Injection-Containers, indem Sie Classpath-Scanning begrenzen und explizite Konfiguration nutzen
- Implementieren Sie Lazy Initialization für Komponenten, die während der ersten Anfrage nicht benötigt werden
- Nutzen Sie bereitgestellte Nebenläufigkeit oder vorgewärmte Instanzen für Serverless-Funktionen, die latenzsensitiven Traffic handhaben
- Optimieren Sie Container-Images durch Nutzung kleinerer Basis-Images und Multi-Stage-Builds, um Pull- und Startzeiten zu verringern
- Erwägen Sie Ahead-of-Time-Kompilierung (GraalVM Native Image, CDS-Archive) für JVM-basierte Legacy-Anwendungen
- Planen Sie periodische Aufwärmanfragen, um zu verhindern, dass Instanzen während Perioden geringen Traffics abkühlen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Eliminiert oder verringert den Latenzstrafe, die die erste Anfrage nach Untätigkeitsperioden erlebt
- Verbessert die Konsistenz der Nutzererfahrung, indem Antwortzeitvarianz verringert wird
- Ermöglicht verlässliche Nutzung von Auto-Scaling- und Serverless-Architekturen für Legacy-Arbeitslasten
- Verringert timeoutbezogene Fehler, die durch langsame Initialisierung verursacht werden

**Kosten und Risiken:**
- Bereitgestellte Nebenläufigkeit und Vorwärmung erhöhen Infrastrukturkosten
- Lazy Initialization könnte Latenz zu unerwarteten Punkten im Anfragelebenszyklus verschieben
- AOT-Kompilierung unterstützt möglicherweise nicht alle von Legacy-Anwendungen genutzten Laufzeitfeatures (Reflection, dynamische Proxys)
- Aufwärmanfragen fügen operative Komplexität hinzu und müssen im Monitoring von echtem Traffic unterschieden werden

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine zu Kubernetes migrierte Legacy-Spring-Boot-Anwendung erlebte Cold-Start-Zeiten von über 20 Sekunden aufgrund umfangreichen Classpath-Scannings, Hibernate-Schema-Validierung und eifrigen Ladens aller Bean-Definitionen. Während Auto-Scaling-Ereignissen empfingen neue Pods Traffic, bevor sie bereit waren, was kaskadierende Timeouts verursachte. Das Team adressierte dies durch den Wechsel zu expliziter Bean-Registrierung, Aktivierung von Hibernate-Lazy-Initialization und Implementierung von Readiness Probes, die auf vollständige Initialisierung warteten. Die Cold-Start-Zeit sank auf 6 Sekunden, und die Hinzufügung der CDS-Archiv-Unterstützung verringerte sie weiter auf 3 Sekunden, was Auto-Scaling während Traffic-Spitzen verlässlich machte.
