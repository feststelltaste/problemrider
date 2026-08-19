---
title: Chaos Engineering
description: Gezielte Einführung von Störungen, um die Widerstandsfähigkeit des Systems
  zu testen.
category:
- Operations
- Testing
problems:
- cascade-failures
- single-points-of-failure
- system-outages
- unpredictable-system-behavior
- slow-incident-resolution
- monitoring-gaps
- fear-of-change
layout: solution
lang: de
en_slug: chaos-engineering
related_solutions:
- slug: resilience
  similarity: 0.85
- slug: stress-testing
  similarity: 0.85
- slug: incident-management
  similarity: 0.8
- slug: secure-software
  similarity: 0.8
- slug: site-reliability-engineering-sre
  similarity: 0.8
- slug: error-budgets
  similarity: 0.8
---

## Description

Chaos Engineering ist die Disziplin, bewusst kontrollierte Fehler in ein System zu injizieren — Prozesse abzutöten, Netzwerkverbindungen zu degradieren, Ressourcen zu erschöpfen, Abhängigkeiten zu deaktivieren —, um empirisch zu validieren, ob die Resilienz-Annahmen des Systems tatsächlich zutreffen, statt zu vertrauen, dass sie es tun, weil die Architektur so designt wurde. Jedes Experiment beginnt mit einer expliziten Hypothese über erwartetes Verhalten unter einem spezifischen Fehler, und das Experiment bestätigt entweder diese Hypothese oder deckt eine Lücke zwischen angenommenem und tatsächlichem Fehlerverhalten auf. Dies ist besonders wichtig in Legacy-Systemen, wo sich Failover-Logik, Wiederholungsverhalten und Single Points of Failure oft über viele Jahre undokumentiert und ungetestet angehäuft haben, sodass niemand im aktuellen Team mit Sicherheit sagen kann, was tatsächlich passiert, wenn eine gegebene Abhängigkeit ausfällt. Statt auf einen echten Produktionsvorfall zu warten, der diese Lücken zum schlechtestmöglichen Zeitpunkt offenbart, bringt Chaos Engineering sie unter kontrollierten Bedingungen ans Licht, mit anwesendem Team, aktivem Monitoring und einem bereiten Abbruchmechanismus, um das Experiment zu stoppen, falls der Explosionsradius zu groß wird. Die Praxis hängt davon ab, dass das System bereits einigermaßen ausgereifte Observability hat, da ohne sie die Auswirkung eines injizierten Fehlers nicht verlässlich gemessen oder eingedämmt werden kann, was in Legacy-Umgebungen, die vor ordentlichem Monitoring entstanden, oft die schwierigere Voraussetzung ist. Über die Zeit verschiebt das systematische Durchführen dieser Experimente die Beziehung eines Teams zu Fehlern von Angst und Vermeidung hin zu evidenzbasiertem Vertrauen, was oft ist, was tatsächlich schnellere und häufigere Legacy-Systemänderungen ermöglicht.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Beginnen Sie mit einer Hypothese darüber, was geschehen sollte, wenn ein spezifischer Fehler auftritt (z. B. „das System sollte innerhalb von 30 Sekunden auf die Backup-Datenbank umschalten")
- Beginnen Sie Chaos-Experimente in Nicht-Produktionsumgebungen, um Vertrauen aufzubauen und offensichtliche Lücken zu identifizieren
- Führen Sie kontrollierte Fehler ein wie das Abtöten von Prozessen, das Injizieren von Netzwerklatenz, das Füllen von Festplatten oder das Deaktivieren von Abhängigkeiten
- Nutzen Sie etablierte Werkzeuge wie Chaos Monkey, Gremlin oder Litmus, um Experimente sicher zu verwalten
- Implementieren Sie einen Abbruchmechanismus, der das Experiment sofort stoppen kann, wenn die Auswirkung akzeptable Schwellen überschreitet
- Führen Sie Experimente während der Geschäftszeiten mit anwesendem Team durch, sodass Probleme in Echtzeit beobachtet und angegangen werden können
- Dokumentieren Sie Befunde aus jedem Experiment und verfolgen Sie die Behebung entdeckter Schwächen

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Offenbart versteckte Fehlermodi und Single Points of Failure, bevor sie Produktionsvorfälle verursachen
- Baut Teamvertrauen in die Resilienz des Systems durch empirische Validierung auf
- Verbessert Vorfallreaktionsfähigkeiten, indem Teams kontrollierten Fehlerszenarien ausgesetzt werden
- Identifiziert Monitoring- und Alarmierungslücken, die sonst unbemerkt blieben
- Treibt architektonische Verbesserungen basierend auf beobachteten Schwächen an

**Kosten und Risiken:**
- Schlecht kontrollierte Experimente können echte Produktionsausfälle verursachen
- Erfordert ausgereiftes Monitoring und Observability, um die Auswirkung injizierter Fehler zu erkennen
- Teams könnten sich gegen die Praxis wehren aus Angst, Vorfälle zu verursachen
- Legacy-Systeme ohne ordentliche Failover-Mechanismen könnten während Experimenten katastrophal ausfallen
- Erfordert organisatorische Zustimmung, da Experimente das Systemverhalten vorübergehend degradieren

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Eine Legacy-E-Commerce-Plattform erlebte unerklärliche Ausfälle während Traffic-Spitzenereignissen. Das Team vermutete verschiedene Single Points of Failure, hatte aber keine Möglichkeit, ihre Resilienz-Annahmen zu validieren. Sie begannen Chaos Engineering in ihrer Staging-Umgebung, indem sie systematisch einzelne Services abtöteten und Systemverhalten beobachteten. Das erste Experiment offenbarte, dass der Session-Management-Service kein Failover hatte, was vollständiges Checkout-Versagen verursachte, wenn er ausfiel. Nach der Behebung dieses Problems gingen sie zu Netzwerkpartitionsexperimenten über, die einen drei Jahre schlummernden Datenbankverbindungs-Wiederholungsbug aufdeckten. Über sechs Monate löste das Team 14 kritische Resilienzprobleme und verringerte ungeplante Ausfallzeit um 60 %.
