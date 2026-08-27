---
title: Service Level Indicators
description: Nachverfolgung zentraler Metriken der Softwarezuverlässigkeit und
  -performance.
category:
- Operations
- Management
problems:
- monitoring-gaps
- gradual-performance-degradation
- constant-firefighting
- slow-application-performance
- poor-operational-concept
- difficulty-quantifying-benefits
- unpredictable-system-behavior
layout: solution
lang: de
en_slug: service-level-indicators
related_solutions:
- slug: service-level-agreements
  similarity: 0.85
- slug: service-level-objectives
  similarity: 0.85
- slug: monitoring
  similarity: 0.8
- slug: transparent-performance-metrics
  similarity: 0.8
- slug: error-budgets
  similarity: 0.8
- slug: continuous-performance-monitoring
  similarity: 0.8
---

## Description

Ein Service Level Indicator ist ein direkt gemessenes, quantitatives Signal nutzerseitigen Verhaltens — Latenz, Fehlerrate, Durchsatz oder eine ähnliche Metrik —, kontinuierlich aus dem laufenden System erfasst, statt abgeleitet oder anekdotisch berichtet. SLIs sind die rohe Messschicht unter Service Level Objectives und Agreements: Ohne einen zuverlässigen SLI ist ein SLO-Ziel nicht durchsetzbar und eine SLA-Zusage nicht verifizierbar. In Legacy-Systemen ist diese Messschicht oft das fehlende Stück, weil Komponenten gebaut wurden, bevor Observability ein Designanliegen war, und keine natürlichen Anknüpfungspunkte zur Erfassung von Request-Timing, Erfolgsraten oder Warteschlangentiefe bieten. Die Definition von SLIs beginnt daher damit, zu entscheiden, wie "gut" aus der Perspektive des Nutzers aussieht, und dann das Legacy-System zu instrumentieren, nachzurüsten oder extern abzufragen, bis dieses Signal zuverlässig erfasst und einer spezifischen Grenze zugeordnet werden kann (wie etwa Latenz von Load-Balancer bis Antwort, ausschließlich Client-Netzwerkzeit). Die resultierenden Daten ersetzen Rätselraten und stilles Wissen über die Systemgesundheit durch einen kontinuierlichen, trendfähigen Datensatz, was es überhaupt erst möglich macht, langsame Degradation zu bemerken — einen schleichenden Anstieg der p99-Latenz, eine schrumpfende Fehlermarge — lange bevor sie sich als Ausfall manifestiert. Da SLIs die Grundlage für Fehlerbudgets und Burn-Rate-Alarmierung sind, ist die korrekte Definition des Indikators eine Voraussetzung für jede nachgelagerte Zuverlässigkeitspraxis, die darauf aufbaut.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Identifizieren Sie die Metriken, die die Nutzererfahrung für jeden Legacy-Systemdienst am besten repräsentieren (Latenz, Fehlerrate, Durchsatz)
- Instrumentieren Sie Legacy-Anwendungen, um SLI-Daten durch Metrikerfassung, Log-Aggregation oder synthetisches Monitoring zu emittieren
- Definieren Sie Messgrenzen klar (z. B. Latenz gemessen vom Load-Balancer-Empfang bis zur Antwort, ausschließlich Client-Netzwerkzeit)
- Etablieren Sie Baselines aus historischen Daten, bevor Sie Ziele setzen
- Erstellen Sie Dashboards, die SLI-Trends über die Zeit anzeigen und Abweichungen vom erwarteten Verhalten hervorheben
- Nutzen Sie SLIs, um Fehlerbudgets abzuleiten, die Zuverlässigkeitsinvestition mit Feature-Entwicklungsgeschwindigkeit ausbalancieren
- Überprüfen Sie SLIs in regelmäßigen Betriebsmeetings, um das Bewusstsein für Legacy-Systemgesundheitstrends aufrechtzuerhalten

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Bietet objektive, quantitative Sichtbarkeit in die Legacy-Systemzuverlässigkeit
- Ermöglicht trendbasierte Frühwarnung, bevor Nutzer Probleme erfahren
- Schafft eine gemeinsame Sprache für die Diskussion der Systemgesundheit über technische und geschäftliche Teams hinweg
- Unterstützt datengetriebene Entscheidungen darüber, wann Legacy-Systeme Investition brauchen versus wann sie stabil genug sind

**Kosten und Risiken:**
- Die Wahl der falschen SLIs kann ein irreführendes Bild der Systemgesundheit liefern
- Legacy-Systeme so zu instrumentieren, dass sie zuverlässige Metriken emittieren, kann erheblichen Aufwand erfordern
- Ausschließlicher Fokus auf messbare Indikatoren kann wichtige qualitative Aspekte vernachlässigen
- SLI-Daten ohne Kontext können zu fehlgeleiteten Optimierungsbemühungen führen

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Das Betriebsteam einer Legacy-E-Commerce-Plattform verließ sich auf anekdotische Berichte, um die Systemgesundheit zu bewerten. Durch die Implementierung von SLIs, die p50- und p99-Anfrage-Latenz, Fehlerraten pro Endpunkt und Checkout-Abschlussraten verfolgten, entdeckte das Team, dass die durchschnittliche Performance zwar akzeptabel war, die p99-Latenz aber seit sechs Monaten aufgrund wachsender Datenbanktabellengrößen stetig gestiegen war. Diese datengetriebene Erkenntnis führte zu einer gezielten Datenbankoptimierungsanstrengung, die die p99-Latenz um 70 % reduzierte und die Checkout-Abschlussraten um 8 % verbesserte, was den Geschäftswert der Zuverlässigkeitsinvestition im Legacy-System direkt demonstrierte.
