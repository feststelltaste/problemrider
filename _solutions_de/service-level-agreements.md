---
title: Service Level Agreements
description: Definition von Erwartungen an Softwareverfügbarkeit und -performance.
category:
- Management
- Operations
problems:
- poor-operational-concept
- stakeholder-frustration
- unclear-goals-and-priorities
- constant-firefighting
- customer-dissatisfaction
- system-outages
- modernization-roi-justification-failure
- communication-risk-outside-project
- legal-disputes
- upstream-timeouts
- vendor-relationship-strain
- poor-contract-design
- stakeholder-confidence-loss
- voided-vendor-support
layout: solution
lang: de
en_slug: service-level-agreements
related_solutions:
- slug: service-level-objectives
  similarity: 0.9
- slug: service-level-indicators
  similarity: 0.85
- slug: performance-budgets
  similarity: 0.8
- slug: incident-management
  similarity: 0.75
- slug: secure-software
  similarity: 0.75
- slug: transparent-performance-metrics
  similarity: 0.75
---

## Description

Ein Service Level Agreement ist eine formale, ausgehandelte Zusage zwischen einem Dienstanbieter und seinen Konsumenten, die messbare Ziele für Verfügbarkeit, Performance und Support-Reaktionsfähigkeit definiert, zusammen mit den Konsequenzen, wenn diese Ziele verfehlt werden. Anders als eine interne Erwartung ist ein SLA ein vertragswertiges Artefakt: Es spezifiziert genau, wie jede Metrik gemessen wird, über welchen Zeitraum, und welche Abhilfemaßnahme — Gutschriften, Eskalation, Vertragsstrafenklauseln — auf eine Verletzung folgt. Legacy-Systeme laufen häufig jahrelang ohne eine solche Vereinbarung, wodurch "akzeptable" Performance zu einer Frage unausgesprochener Annahme wird, die sich je nachdem verschiebt, wer gefragt wird und wie kürzlich etwas kaputtgegangen ist. Diese Abwesenheit wird zu einer ernsthaften Haftungsgefahr, sobald ein Legacy-System kundenseitige Verpflichtungen, Lieferantenverträge oder regulatorische Auflagen untermauert, weil es keine objektive Grundlage gibt, um zu entscheiden, ob das System angemessen läuft, oder um knappe Entwicklungszeit auf Zuverlässigkeitsarbeit zu priorisieren. Die Einführung von SLAs erzwingt ein explizites Gespräch darüber, was das Legacy-System heute realistisch liefern kann versus was Stakeholder annehmen, dass es liefert, was oft der erste Moment ist, in dem eine chronische Performance- oder Verfügbarkeitslücke sichtbar und umsetzbar wird. Einmal etabliert, wird das SLA auch zu einem Hebel für die Rechtfertigung von Modernisierungsinvestitionen, da ein quantifiziertes Defizit weit leichter zu finanzieren ist als eine vage Beschwerde über Instabilität.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Definieren Sie messbare Verfügbarkeits- und Performance-Ziele für Legacy-Systemdienste basierend auf Geschäftsbedürfnissen
- Spezifizieren Sie SLA-Metriken klar: Uptime-Prozentsatz, Antwortzeit-Perzentile, Fehlerratenschwellen
- Etablieren Sie Messmethodik und Berichtshäufigkeit für jede SLA-Metrik
- Definieren Sie Konsequenzen und Abhilfeprozesse, wenn SLAs nicht erfüllt werden
- Richten Sie SLA-Ziele an den realistischen Fähigkeiten des Legacy-Systems und der geplanten Verbesserungstrajektorie aus
- Nutzen Sie SLA-Daten, um Investitionen in Legacy-Systemmodernisierung und Zuverlässigkeitsverbesserungen zu rechtfertigen
- Überprüfen und passen Sie SLAs periodisch an, während sich Systemfähigkeiten und Geschäftsanforderungen weiterentwickeln

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Schafft gemeinsame Erwartungen zwischen Geschäfts-Stakeholdern und Entwicklungsteams
- Bietet objektive Kriterien zur Bewertung der Legacy-Systemperformance
- Rechtfertigt Investitionen in Zuverlässigkeitsverbesserungen mit messbaren Zielen
- Ermöglicht datengetriebene Priorisierung von Wartung versus Feature-Arbeit

**Kosten und Risiken:**
- Zu aggressiv gesetzte SLAs für Legacy-Systeme erzeugen dauerhaften Fehlschlag und Team-Demoralisierung
- Die Messung von SLAs erfordert Monitoring-Infrastruktur, die Legacy-Systemen fehlen könnte
- SLAs können politisiert und als Schuldzuweisungsinstrumente statt Verbesserungswerkzeuge genutzt werden
- Enger SLA-Fokus könnte wichtige Qualitätsdimensionen vernachlässigen, die nicht von der Vereinbarung abgedeckt werden

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein B2B-SaaS-Unternehmen betrieb eine Legacy-Plattform ohne definierte Verfügbarkeitszusagen. Kunden beschwerten sich über häufige Ausfälle, aber ohne SLAs hatte das Entwicklungsteam keinen Rahmen, um Zuverlässigkeitsarbeit gegen Feature-Anfragen zu priorisieren. Nach der Etablierung eines 99,5 %-Monats-Verfügbarkeits-SLA mit definierter Messmethodik entdeckte das Team, dass es aktuell nur 98,2 % erreichte. Diese Daten rechtfertigten eine dedizierte Zuverlässigkeitsverbesserungsinitiative, die die Verfügbarkeit innerhalb von drei Monaten über das SLA-Ziel brachte. Das explizite Ziel half dem Team auch, sich gegen Feature-Anfragen zu wehren, die die Stabilität kompromittiert hätten.
