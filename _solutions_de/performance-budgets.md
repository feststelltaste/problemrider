---
title: Performance-Budgets
description: Definition von Performance-Kennzahlen als Teil der Anforderungen.
category:
- Performance
- Requirements
problems:
- gradual-performance-degradation
- slow-application-performance
- quality-blind-spots
- inadequate-requirements-gathering
- feature-creep-without-refactoring
- high-client-side-resource-consumption
- graphql-complexity-issues
- high-resource-utilization-on-client
- inefficient-code
- inefficient-frontend-code
layout: solution
lang: de
en_slug: performance-budgets
related_solutions:
- slug: transparent-performance-metrics
  similarity: 0.8
- slug: performance-measurements
  similarity: 0.8
- slug: continuous-performance-monitoring
  similarity: 0.8
- slug: service-level-agreements
  similarity: 0.8
- slug: service-level-objectives
  similarity: 0.75
- slug: performance-optimization
  similarity: 0.75
---

## Description

Ein Performance-Budget ist ein messbares Ziel — Seitenladezeit unter zwei Sekunden, API-Antwort unter 500 Millisekunden, eine maximale Bundle-Größe —, das als durchzusetzende Anforderung behandelt wird, statt als erhoffte Ambition. Legacy-Systeme scheitern selten katastrophal an Performance; sie verschlechtern sich schrittweise, während Feature um Feature hinzugefügt wird, ohne dass jemand die kumulativen Kosten verfolgt, bis das System, das sich einst schnell anfühlte, still trödelig genug geworden ist, dass sich niemand mehr genau erinnert, wann sich das geändert hat. Das Verdrahten von Budgetprüfungen in die CI/CD-Pipeline verwandelt diese unsichtbare Drift in einen sofortigen, sichtbaren Build-Fehler in dem Moment, in dem eine Änderung das Limit überschreiten würde, und gibt dem Team objektive Kriterien — statt eines subjektiven Streits — zur Bewertung, ob die Performance-Kosten eines neuen Features akzeptabel sind.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Etablieren Sie messbare Performance-Ziele für zentrale nutzerseitige Operationen (z. B. Seitenladezeit unter 2 Sekunden, API-Antwort unter 500 ms)
- Erfassen Sie aktuelle Performance-Kennzahlen als Basislinie, um die Lücke zwischen aktuellem Zustand und gewünschten Zielen zu verstehen
- Integrieren Sie Performance-Budget-Prüfungen in die CI/CD-Pipeline, sodass Regressionen vor der Bereitstellung erkannt werden
- Definieren Sie Budgets für Bundle-Größe, Time to Interactive, Speicherverbrauch und API-Antwortzeiten
- Weisen Sie Teams oder Komponenten Performance-Budgets zu, sodass Verantwortung für Performance verteilt wird
- Überprüfen und passen Sie Budgets vierteljährlich an, während sich das System weiterentwickelt und sich Nutzererwartungen ändern
- Machen Sie Performance-Budget-Verletzungen so sichtbar wie fehlschlagende Tests, um schrittweise Verschlechterung zu verhindern

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Verhindert den schrittweisen Performance-Verfall, der langlebige Legacy-Systeme plagt
- Schafft gemeinsame Verantwortlichkeit für Performance im gesamten Entwicklungsteam
- Liefert objektive Kriterien zur Bewertung der Performance-Auswirkung neuer Features
- Macht Performance zu einer erstklassigen Anforderung statt einem nachträglichen Gedanken

**Kosten und Risiken:**
- Zu aggressiv gesetzte Budgets können Feature-Entwicklung verlangsamen und Teams frustrieren
- Budgets erfordern laufende Kalibrierung, während sich System und Nutzungsmuster weiterentwickeln
- Legacy-Systeme können weit von jedem vernünftigen Budget entfernt sein, was die anfängliche Lücke demoralisierend macht
- Genaue Performance-Messung erfordert Investition in Monitoring-Infrastruktur
- Teams optimieren möglicherweise auf die Kennzahl statt auf die tatsächliche Nutzererfahrung

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine SaaS-Plattform hatte erlebt, dass die Ladezeit ihres Haupt-Dashboards über drei Jahre von 1,5 Sekunden auf 8 Sekunden anstieg, während Features ohne Performance-Aufsicht hinzugefügt wurden. Das Team etablierte ein Performance-Budget von 3 Sekunden für das initiale Dashboard-Laden und 200 ms für nachfolgende Interaktionen. Sie fügten Lighthouse-CI-Prüfungen zu ihrer Build-Pipeline hinzu, die den Build fehlschlagen ließen, wenn Budgets überschritten wurden. Innerhalb von sechs Monaten hatte das Team die Ladezeit durch inkrementelle Optimierungen auf 2,8 Sekunden reduziert, und die Budgetprüfungen verhinderten mehrere Regressionen, bevor sie Produktion erreichten. Die Budgets gaben dem Team außerdem einen klaren, nicht-konfrontativen Weg, sich gegen Feature-Anfragen zu wehren, die das Budget gesprengt hätten.
