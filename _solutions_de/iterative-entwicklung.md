---
title: Iterative Entwicklung
description: Inkrementelle Entwicklung und Auslieferung von Software in kurzen
  Zyklen.
category:
- Process
- Management
quality_tactics_url: https://qualitytactics.de/en/functional-suitability/iterative-development/
problems:
- deadline-pressure
- time-pressure
- unrealistic-deadlines
- unrealistic-schedule
- constantly-shifting-deadlines
- cascade-delays
- budget-overruns
- delayed-project-timelines
- poor-planning
- poor-project-control
- approval-dependencies
- large-feature-scope
- planning-dysfunction
- planning-credibility-issues
- project-resource-constraints
- missed-deadlines
- reduced-predictability
- procrastination-on-complex-tasks
- process-design-flaws
- market-pressure
- perfectionist-culture
layout: solution
lang: de
en_slug: iterative-development
related_solutions:
- slug: short-iteration-cycles
  similarity: 0.95
- slug: feature-driven-development
  similarity: 0.75
- slug: evolutionary-requirements-development
  similarity: 0.75
- slug: stakeholder-feedback-loops
  similarity: 0.75
- slug: continuous-feedback
  similarity: 0.75
- slug: ci-cd-pipeline
  similarity: 0.75
---

## Description

Iterative Entwicklung liefert funktionierende Software in kurzen Zyklen fester Länge — ein vorführbares Inkrement alle ein bis vier Wochen — statt sich auf ein einziges langes Release festzulegen, gebaut gegen vorab getroffene Annahmen, die in der Legacy-Modernisierung besonders wahrscheinlich falsch sind. Die Illusion langfristiger Gewissheit gegen kurzfristige Vorhersagbarkeit einzutauschen, gestützt durch tatsächlich gelieferte Evidenz, zählt genau dort am meisten, wo Vertrauen bereits durch eine Geschichte verpasster Termine und abgebrochener mehrjähriger Neufassungsversuche beschädigt wurde, da eine Erfolgsbilanz kleiner, eingehaltener Zusagen ist, was Stakeholder-Vertrauen wieder aufbaut, das ein großer Vorabplan nie konnte. Die echte Schwierigkeit in Legacy-Kontexten ist, dass frühe Iterationen ihre Kapazität oft auf grundlegende Infrastruktur verwenden müssen — Testautomatisierung, Deployment-Pipelines — bevor irgendein geschäftlich sichtbares Feature erscheint, was für Stakeholder, die ab der ersten Iteration Features erwarten, wie langsamer Fortschritt aussehen kann.

## How to Apply ◆

> In Legacy-Systemen, in denen lange Release-Zyklen, Big-Bang-Deployments und wasserfallartige Planung zur Norm geworden sind, führt iterative Entwicklung kurze, vorhersagbare Lieferzyklen ein, die Risiko verringern und Stakeholder-Vertrauen durch demonstrierten statt versprochenen Fortschritt wiederherstellen.

- Definieren Sie Iterationslängen von ein bis vier Wochen und behandeln Sie sie als feste Zeitgrenzen. Am Ende jeder Iteration sollte das Team ein funktionierendes Inkrement der Software haben, das Stakeholdern vorgeführt werden kann. In Legacy-Kontexten könnten sich frühe Iterationen auf den Aufbau der Lieferpipeline selbst fokussieren — automatisierte Builds, Testinfrastruktur und Deployment-Mechanismen —, bevor geschäftliche Features geliefert werden.
- Zerlegen Sie große Features in dünne vertikale Scheiben, die End-to-End-Funktionalität liefern, statt horizontale Schichten. Statt zuerst die Datenbankschicht, dann die Servicecicht, dann die UI zu bauen, liefern Sie eine minimale, aber vollständige Feature-Scheibe, die alle Schichten berührt. Dies ist besonders wichtig in der Legacy-Modernisierung, wo großumfängliche Ersetzungen unverhältnismäßiges Risiko tragen.
- Führen Sie Iterationsplanungssitzungen durch, in denen das Team eine kleine, erreichbare Menge an Arbeitsposten basierend auf gemessener Velocity statt optimistischer Schätzungen auswählt. Nutzen Sie den tatsächlichen Durchsatz des Teams aus vorherigen Iterationen als primären Input für die Planung, nicht Management-Ziele oder externe Fristen.
- Halten Sie am Ende jeder Iteration ein Review oder eine Vorführung ab, bei dem Stakeholder funktionierende Software sehen und Feedback geben können. In Legacy-Umgebungen dienen diese Reviews als Beleg dafür, dass die Modernisierung voranschreitet, und helfen, Planungsglaubwürdigkeit wiederherzustellen, die durch vergangene Projektüberschreitungen beschädigt sein könnte.
- Führen Sie Retrospektiven an Iterationsgrenzen durch, um Prozessverbesserungen zu identifizieren. Kurze Zyklen machen die Feedback-Schleife eng genug, dass Probleme erkannt und angegangen werden, bevor sie sich aufsummieren. Teams, die an Legacy-Systemen arbeiten, entdecken oft systemische Probleme — wie Genehmigungsengpässe oder fehlende Testabdeckung —, die inkrementell angegangen werden können, statt eine separate Verbesserungsinitiative zu erfordern.
- Entkoppeln Sie Iterationszusagen von externen Fristen. Das Team verpflichtet sich zu dem, was es in der nächsten Iteration basierend auf Kapazität liefern kann, und der Product Owner passt Umfang und Priorität entsprechend an. Dies verhindert das Muster unrealistischer Fristen, die Teams auferlegt werden, und schafft stattdessen einen verlässlichen Liefertakt.
- Etablieren Sie eine „Definition of Done", die Testen, Dokumentation und Deployment-Bereitschaft einschließt, damit jede Iteration echt auslieferbare Arbeit produziert. In Legacy-Systemen könnte das Erreichen dieses Standards anfängliche Investition in Testautomatisierung und kontinuierliche Integration erfordern, was als explizite Iterationsziele geplant werden sollte.
- Nutzen Sie Iterationsmetriken — Velocity, Zykluszeit, Defektrate —, um objektive Daten für Planungsgespräche bereitzustellen. Wenn Stakeholder konsistente Lieferdaten über mehrere Iterationen sehen können, verschieben sich Planungsdiskussionen von gegnerischen Verhandlungen zu kollaborativer Priorisierung.

## Tradeoffs ⇄

> Iterative Entwicklung tauscht die Illusion langfristiger Gewissheit gegen kurzfristige Vorhersagbarkeit, gestützt durch empirische Evidenz, was besonders effektiv in Legacy-Umgebungen ist, wo Unsicherheit hoch und Vertrauen möglicherweise niedrig ist.

**Vorteile:**

- Verringert das Risiko großmaßstäblichen Projektversagens, indem Wert in kleinen Inkrementen geliefert wird, was erlaubt, die Richtung zu ändern, bevor erhebliche Investition verschwendet wird — ein kritischer Vorteil in der Legacy-Modernisierung, wo anfängliche Annahmen häufig falsch sind.
- Stellt Planungsglaubwürdigkeit wieder her, indem eine Erfolgsbilanz der Erfüllung kurzfristiger Zusagen etabliert wird, was schrittweise Stakeholder-Vertrauen wiederaufbaut, das durch Jahre verpasster Fristen und Budgetüberschreitungen beschädigt sein könnte.
- Macht Termindruck handhabbar, indem der Umfang jeder Zusage auf das begrenzt wird, was das Team nachweislich liefern kann, statt Teams zu verlangen, sich zu Monaten Arbeit mit unsicheren Anforderungen zu verpflichten.
- Bietet Frühwarnung vor kaskadierenden Verzögerungen und Abhängigkeitsproblemen, weil Integration jede Iteration statt am Ende eines Projekts geschieht, was blockierende Probleme Wochen statt Monate vor dem geplanten Liefertermin sichtbar macht.
- Ermöglicht sinnvolle Projektkontrolle durch Fortschrittsverfolgung auf Iterationsebene, was Projektmanagern objektive Daten über tatsächlichen Fortschritt statt subjektiver Statusberichte gibt.
- Verringert die Auswirkung von Genehmigungsabhängigkeiten, indem Arbeit so strukturiert wird, dass Genehmigungen für kleine, gut definierte Inkremente statt große, mehrdeutige Arbeitspakete eingeholt werden können.

**Kosten und Risiken:**

- Erfordert organisatorische Disziplin, Iterationsgrenzen zu respektieren und der Versuchung zu widerstehen, mitten in der Iteration Arbeit hinzuzufügen, was in Umgebungen, die an Ad-hoc-Prioritätsänderungen gewöhnt sind, schwierig sein kann.
- Anfängliche Iterationen in Legacy-Umgebungen liefern möglicherweise wenig sichtbaren Geschäftswert, weil das Team in grundlegende Fähigkeiten wie Testautomatisierung, kontinuierliche Integration und Deployment-Infrastruktur investieren muss.
- Stakeholder, die an detaillierte Langzeitpläne gewöhnt sind, könnten iterative Entwicklung als Mangel an Planung statt als bewusste Strategie zum Umgang mit Unsicherheit wahrnehmen, was Aufklärung und Geduld erfordert.
- Legacy-Features in dünne Scheiben zu zerlegen kann echt schwierig sein, wenn die bestehende Architektur monolithisch und eng gekoppelt ist, was manchmal architektonische Investition erfordert, bevor iterative Lieferung praktikabel wird.
- Der Overhead von Iterationszeremonien — Planung, Review, Retrospektive — kann sich für kleine Teams belastend anfühlen und muss proportional zur Iterationslänge gehalten werden, um nicht zu viel Entwicklungszeit zu verbrauchen.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie iterative Entwicklung Zeitplan-, Planungs- und Umfangsherausforderungen in Legacy-System-Kontexten adressiert.

Ein Finanzdienstleistungsunternehmen hatte drei Jahre lang mit einem Wasserfallansatz versucht, sein zwanzig Jahre altes Kreditvergabesystem zu ersetzen, wobei jeder Versuch nach zwölf bis achtzehn Monaten mit Abbruch endete, als das Projekt hoffnungslos hinter Zeitplan und Budget zurückfiel. Das Ersatzteam wechselte zu zweiwöchigen Iterationen, begann mit dem einfachsten Kredittyp und lieferte innerhalb des ersten Monats ein funktionierendes System, das grundlegende Anträge End-to-End verarbeiten konnte. Jede nachfolgende Iteration fügte Unterstützung für komplexere Kredittypen, zusätzliche Validierungsregeln oder regulatorische Anforderungen hinzu. Nach sechs Monaten stetigen, sichtbaren Fortschritts — mit Stakeholdern, die an zweiwöchentlichen Vorführungen teilnahmen — hatte das Projekt mehr funktionierende Funktionalität geliefert als der vorherige dreijährige Versuch, und die Planungsglaubwürdigkeit des Entwicklungsteams war so weit wiederhergestellt, dass die Geschäftsführung begann, aktiv an der Priorisierung mitzuwirken, statt Fristen aufzuerlegen.

Das ERP-System eines Fertigungsunternehmens benötigte ein größeres Upgrade, das anfänglich als neunmonatiges Projekt mit einem festen Liefertermin, gekoppelt an eine regulatorische Compliance-Frist, umrissen wurde. Das Team zerlegte die Arbeit in dreiwöchige Iterationen und priorisierte zuerst die compliance-kritischen Features. Bis zur vierten Iteration waren die Compliance-Features vollständig und deployt, was den Termindruck von der verbleibenden Arbeit nahm. Die nicht kritischen Features wurden dann über nachfolgende Iterationen basierend auf geschäftlicher Priorität geliefert. Dieser Ansatz beseitigte die kaskadierenden Verzögerungen, die aufgetreten wären, hätte man die Compliance-Frist verpasst, und die Gesamtkosten waren niedriger als die ursprüngliche Schätzung, weil das Team die Überstunden und Nacharbeit vermied, die typischerweise fristgetriebene Projekte in dieser Organisation begleiteten.
