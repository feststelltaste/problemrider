---
title: Service Level Objectives
description: Definition messbarer Ziele für Systemzuverlässigkeit und
  -performance.
category:
- Operations
- Management
problems:
- monitoring-gaps
- slow-incident-resolution
- stakeholder-dissatisfaction
- unclear-goals-and-priorities
- system-outages
- gradual-performance-degradation
- quality-blind-spots
- modernization-roi-justification-failure
- vendor-relationship-strain
layout: solution
lang: de
en_slug: service-level-objectives
related_solutions:
- slug: service-level-agreements
  similarity: 0.9
- slug: service-level-indicators
  similarity: 0.85
- slug: site-reliability-engineering-sre
  similarity: 0.8
- slug: error-budgets
  similarity: 0.75
- slug: performance-budgets
  similarity: 0.75
- slug: capacity-planning
  similarity: 0.75
---

## Description

Ein Service Level Objective ist ein intern vereinbarter Zielwert für einen Service Level Indicator über ein definiertes Zeitfenster — zum Beispiel, dass 99,9 % der Anfragen innerhalb von zwei Sekunden abgeschlossen werden, gemessen über einen rollierenden 28-Tage-Zeitraum. Während ein SLI einfach das ist, was gemessen wird, und ein SLA das ist, was einer externen Partei vertraglich zugesagt wird, ist das SLO die Betriebsschwelle, an die sich Entwicklungsteams selbst halten, und es ist das, was eine rohe Metrik in eine umsetzbare Entscheidungsregel verwandelt. Die Lücke, die es in Legacy-Systemen schließt, ist die Abwesenheit einer gemeinsamen Definition von "zuverlässig genug": Ohne sie wird jede Verlangsamung oder Störung als gleich dringend behandelt, Eskalationen werden von demjenigen getrieben, der am lautesten beschwert, und Ingenieure haben keine prinzipielle Grundlage, um zu entscheiden, ob ein bestimmter Fehlschlag eine All-Hands-Reaktion rechtfertigt oder toleriert werden kann. Durch die Ableitung eines Fehlerbudgets aus dem SLO — die Menge an zulässigem Fehlschlag, bevor das Ziel verletzt wird — schafft die Praxis einen expliziten, quantifizierten Auslöser dafür, wann Zuverlässigkeitsarbeit Priorität vor Feature-Entwicklung haben sollte, was politische Verhandlung durch eine gemeinsame, datengestützte Regel ersetzt. Dies ist besonders wertvoll in Legacy-Modernisierung, wo das SLO auch als Ziel für das Ersatzsystem und als Baseline fungiert, gegen die die Lücke zwischen aktueller und gewünschter Zuverlässigkeit gemessen und an finanzierende Stakeholder kommuniziert werden kann.

## How to Apply ◆

> Legacy-Systeme operieren oft ohne klar definierte Zuverlässigkeitsziele, was es unmöglich macht, akzeptable Degradation von echten Vorfällen zu unterscheiden. Service Level Objectives bieten messbare Schwellen, die Entwicklungsprioritäten mit Geschäftserwartungen abgleichen.

- Inventarisieren Sie alle nutzerseitigen und internen Dienste im Legacy-System und identifizieren Sie die kritischsten Nutzerreisen. Bestimmen Sie für jede Reise, welche Metriken (Verfügbarkeit, Latenz, Fehlerrate, Durchsatz) die Nutzerzufriedenheit am direktesten beeinflussen.
- Arbeiten Sie mit Geschäfts-Stakeholdern zusammen, um SLO-Ziele zu definieren, die tatsächliche Nutzererwartungen widerspiegeln statt aspirative Perfektion. Ein 99,9 %-Verfügbarkeitsziel bedeutet ein sehr anderes Fehlerbudget als 99,99 %, und Legacy-Systeme brauchen selten Letzteres — oder können es erreichen.
- Instrumentieren Sie das Legacy-System, um SLI-Metriken (Service Level Indicator) zuverlässig zu messen. Dies erfordert oft das Hinzufügen von Monitoring zu Komponenten, die nie für Observability konzipiert wurden, wie Batch-Prozesse, Message-Queues oder Mainframe-Transaktionen.
- Etablieren Sie Fehlerbudgets, die aus jedem SLO abgeleitet werden. Wenn das Fehlerbudget aufgebraucht ist, priorisieren Sie Zuverlässigkeitsarbeit vor Feature-Entwicklung. Dies schafft einen datengetriebenen Mechanismus zur Rechtfertigung von technischer-Schulden-Reduktion gegenüber dem Management.
- Richten Sie automatisierte Alarmierung basierend auf Burn-Rate-Analyse statt statischer Schwellen ein. Alarmieren Sie, wenn das Fehlerbudget schneller als erwartet verbraucht wird, nicht wenn eine einzelne Metrik eine Linie überschreitet. Dies reduziert Alarmmüdigkeit in Systemen, die bereits übermäßigen Lärm produzieren.
- Überprüfen Sie SLOs vierteljährlich und passen Sie Ziele basierend auf gemessener Performance und sich ändernden Geschäftsbedürfnissen an. Übermäßig ambitionierte SLOs für Legacy-Systeme erzeugen konstantes Feuerwehrlöschen; übermäßig nachsichtige verschleiern echte Probleme.
- Dokumentieren Sie SLOs und ihre Begründung an einem zentralen, zugänglichen Ort, sodass alle Teams verstehen, für welche Zuverlässigkeitsziele sie verantwortlich sind und warum diese Ziele gewählt wurden.

## Tradeoffs ⇄

> SLOs bieten eine gemeinsame Sprache für Zuverlässigkeitsgespräche und einen Rahmen zur Priorisierung von Arbeit, erfordern aber Investition in Messinfrastruktur und organisatorische Ausrichtung.

**Vorteile:**

- Bieten objektive Kriterien für die Entscheidung, wann in Zuverlässigkeit versus Features investiert werden soll, und ersetzen subjektive Argumente durch datengetriebene Entscheidungen.
- Bringen Entwicklungsteams und Geschäfts-Stakeholder in Einklang darüber, was "zuverlässig genug" bedeutet, was Konflikte über die Priorisierung technischer-Schulden-Arbeit reduziert.
- Ermöglichen proaktive Identifikation von Zuverlässigkeitstrends, bevor sie zu Ausfällen werden, durch Fehlerbudget-Tracking und Burn-Rate-Monitoring.
- Schaffen eine verteidigbare Rechtfertigung für Modernisierungsinvestitionen, indem sie die Lücke zwischen aktueller Zuverlässigkeit und Geschäftsanforderungen quantifizieren.

**Kosten und Risiken:**

- Erfordern erhebliche Vorabinvestition in Monitoring und Instrumentierung, was besonders herausfordernd für Legacy-Systeme mit begrenzter Observability sein kann.
- Schlecht gewählte SLO-Ziele können entweder falsche Dringlichkeit erzeugen (zu aggressiv) oder echte Probleme verschleiern (zu nachsichtig), und das richtige Niveau zu finden erfordert Iteration.
- Fehlerbudget-Richtlinien können Spannungen zwischen Teams erzeugen, wenn Zuverlässigkeitsarbeit mit Feature-Lieferung konkurriert, was starkes organisatorisches Engagement erfordert.
- Die genaue Messung von SLIs in Legacy-Systemen mit komplexen, undurchsichtigen Architekturen ist nicht trivial und kann irreführende Metriken produzieren, wenn die Instrumentierung unvollständig ist.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie SLOs Legacy-Systemen helfen, Zuverlässigkeitserwartungen zu verwalten und Verbesserungsarbeit zu priorisieren.

Ein Finanzdienstleistungsunternehmen betreibt ein Legacy-Transaktionsverarbeitungssystem, das während Spitzenhandelszeiten periodische Verlangsamungen erlebt. Ohne SLOs löst jede Verlangsamung eine Eskalation an das obere Management aus, und Entwicklungsteams verbringen die meiste Zeit im reaktiven Feuerwehrlöschmodus. Das Team definiert ein SLO von 99,5 % der Transaktionen, die innerhalb von 2 Sekunden abgeschlossen werden, gemessen über ein rollierendes 28-Tage-Fenster. Dies gibt ihnen ein konkretes Fehlerbudget von etwa 3,6 Stunden degradierter Performance pro Monat. Als Monitoring zeigt, dass das Fehlerbudget in der ersten Woche des Monats mit doppelter erwarteter Rate verbraucht wird, untersucht das Team proaktiv und entdeckt eine langsame Datenbankabfrage, die durch eine kürzliche Änderung eingeführt wurde. Sie beheben sie, bevor ein für Kunden sichtbarer Ausfall auftritt. Über sechs Monate sinkt die Anzahl der Eskalationen um 70 %, weil Stakeholder verstehen, dass gelegentliche Latenzspitzen innerhalb des Fehlerbudgets erwartet und akzeptabel sind.

Ein Einzelhandelsunternehmen betreibt eine Legacy-E-Commerce-Plattform, bei der der Checkout-Flow von fünf verschiedenen Backend-Diensten abhängt, von denen mehrere jahrzehntealte Systeme ohne Monitoring sind. Das Team instrumentiert jeden Dienst und definiert Pro-Dienst-SLOs. Sie entdecken, dass die Zahlungs-Gateway-Integration eine Erfolgsrate von 97 % hat — weit unter dem 99,5 %-Ziel, das für eine reibungslose Checkout-Erfahrung nötig ist. Diese Daten liefern die Rechtfertigung, die nötig ist, um die Ersetzung der brüchigen Zahlungsintegration zu priorisieren, ein Projekt, das wiederholt zugunsten neuer Features verschoben worden war. Nach der Ersetzung verbessern sich die Checkout-Erfolgsraten auf 99,7 %, was sich direkt auf den Umsatz auswirkt.
