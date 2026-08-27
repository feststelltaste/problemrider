---
title: Technischer Spike
description: Validierung, dass eine Architektur unter erwartetem Wachstum
  wartbar bleibt.
category:
- Architecture
- Process
problems:
- analysis-paralysis
- implementation-starts-without-design
- modernization-strategy-paralysis
- fear-of-change
- assumption-based-development
- premature-technology-introduction
- decision-avoidance
- cv-driven-development
- decision-paralysis
- delayed-decision-making
- extended-research-time
- inability-to-innovate
- procrastination-on-complex-tasks
- reduced-innovation
- complex-implementation-paths
layout: solution
lang: de
en_slug: technical-spike
related_solutions:
- slug: functional-spike
  similarity: 0.8
- slug: walking-skeleton
  similarity: 0.7
- slug: chaos-engineering
  similarity: 0.65
- slug: living-documentation
  similarity: 0.65
- slug: pattern-language
  similarity: 0.65
- slug: risk-analysis
  similarity: 0.65
---

## Description

Ein technischer Spike ist eine strikt zeitlich begrenzte Untersuchung — typischerweise ein bis drei Tage —, gebaut, um eine einzelne, spezifische architektonische Frage durch den einfachstmöglichen Prototyp zu beantworten, wobei der Code selbst verworfen wird, sobald die Antwort erfasst ist. Legacy-Modernisierungsentscheidungen neigen besonders dazu, in ungelöster Debatte stecken zu bleiben, genau weil sie oft an Unbekannten hängen, die keine Menge an Diskussion klären kann — ob ein Migrationsansatz tatsächlich unter Last performen wird, ob ein neues Framework sauber mit einer Legacy-API integriert —, und ein Spike ersetzt diese Debatte durch empirische Evidenz, die direkt gegen das echte System gesammelt wird. Den Prototyp-Code anschließend zu verwerfen, statt ihn in Richtung Produktion abgleiten zu lassen, hält die Übung ehrlich: Der Wert ist die Antwort auf die Frage, nicht ein Vorsprung bei der Implementierung, und ein Spike, dessen Umfang nicht straff gehalten wird, kann sich still in ein offenes Nebenprojekt verwandeln statt in einen schnellen, entscheidenden Input für die tatsächliche Entscheidung.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Definieren Sie eine klare Frage oder Hypothese, die der Spike beantworten soll, bevor Sie beginnen
- Setzen Sie eine strikte Zeitbox für den Spike (typischerweise ein bis drei Tage), um zu verhindern, dass er zu einem offenen Projekt wird
- Bauen Sie den einfachstmöglichen Prototyp, der die Hypothese validiert oder widerlegt
- Fokussieren Sie sich auf die riskantesten Unbekannten: Integration mit Legacy-APIs, Performance unter Last oder Migrationsmachbarkeit
- Dokumentieren Sie Befunde und Entscheidungen, unabhängig davon, ob der Spike erfolgreich ist oder scheitert
- Verwerfen Sie den Spike-Code, nachdem die Erkenntnisse erfasst wurden; lassen Sie Prototyp-Code nicht in Produktion abgleiten
- Präsentieren Sie Spike-Ergebnisse dem Team, um kollektive Entscheidungsfindung zu informieren

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Reduziert Risiko durch Validierung von Annahmen, bevor eine teure Implementierung verpflichtet wird
- Liefert konkrete Evidenz zur Unterstützung oder Infragestellung architektonischer Entscheidungen
- Durchbricht Analyseparalyse, indem theoretische Debatten in empirische Untersuchungen verwandelt werden
- Baut Teamvertrauen in den gewählten Ansatz auf

**Kosten und Risiken:**
- Für Spikes aufgewendete Zeit produziert nicht direkt produktionsreifen Code
- Schlecht abgegrenzte Spikes können sich hinziehen und zu Mini-Projekten werden
- Spike-Ergebnisse könnten falsch interpretiert werden, wenn die Prototyp-Bedingungen nicht der Produktionsrealität entsprechen
- Teams könnten von Spikes abhängig werden und sich weigern, sich ohne einen zu verpflichten

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Team debattierte, ob die Datenzugriffsschicht eines Legacy-Monolithen von rohem JDBC zu einem ORM-Framework migriert werden sollte. Die Meinungen waren geteilt, und die Diskussion war seit Wochen ins Stocken geraten. Der Architekt schlug einen zweitägigen Spike vor, bei dem ein Entwickler ein einzelnes, repräsentatives Modul zum ORM migrierte und die Auswirkung auf Performance, Codekomplexität und Testschreibbarkeit maß. Der Spike offenbarte, dass das ORM 90 % der Abfragen gut handhabte, aber mit den komplexen Berichtsabfragen des Systems kämpfte. Diese Evidenz führte das Team dazu, das ORM für Standard-CRUD-Operationen zu übernehmen, während optimiertes SQL für Berichte beibehalten wurde, was die Debatte mit einer pragmatischen, evidenzbasierten Entscheidung beendete.
