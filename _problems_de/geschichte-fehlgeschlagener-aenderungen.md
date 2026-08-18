---
title: Geschichte fehlgeschlagener Änderungen
description: Eine vergangene Vorgeschichte fehlgeschlagener Deployments oder Änderungen
  schafft eine Kultur der Angst und des Widerstands gegen künftige Modifikationen.
category:
- Culture
- Process
related_problems:
- slug: resistance-to-change
  similarity: 0.75
- slug: fear-of-change
  similarity: 0.75
- slug: fear-of-failure
  similarity: 0.7
- slug: fear-of-breaking-changes
  similarity: 0.65
- slug: maintenance-paralysis
  similarity: 0.65
- slug: past-negative-experiences
  similarity: 0.65
solutions:
- architecture-decision-records
- blameless-postmortems
- functional-spike
- mikado-method
- parallel-run
- team-retrospectives
- pilot-projects
- small-change-batches
- characterization-tests
- delivery-performance-metrics
- executive-sponsorship
- staged-investment-with-decision-gates
layout: problem
lang: de
en_slug: history-of-failed-changes
---

## Description
Eine Geschichte fehlgeschlagener Änderungen kann eine anhaltende negative Auswirkung auf die Kultur und Entwicklungsgeschwindigkeit eines Teams haben. Wenn vergangene Deployments zu erheblichen Ausfällen oder Rollbacks geführt haben, werden Entwickler zurückhaltend, weitere Änderungen vorzunehmen, was zu einer Kultur der Angst und Risikoscheu führt. Dies kann Innovation ersticken und es schwierig machen, technische Schulden anzugehen oder neue Features einzuführen.

## Indicators ⟡
- Entwickler zögern, Aufgaben zu übernehmen, die die Modifikation kritischer Systemteile beinhalten.
- Das Team hat einen sehr langsamen und umständlichen Änderungsgenehmigungsprozess.
- Es gibt ein allgemeines Gefühl von "Was nicht kaputt ist, sollte man nicht reparieren."
- Das Team hat eine Geschichte langer und stressiger Release-Zyklen.

## Symptoms ▲

- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Vergangene Deployment-Fehlschläge schaffen einen anhaltenden emotionalen Widerstand gegen Modifikationen, selbst wenn Änderungen notwendig sind.
- [Widerstand gegen Veränderung](widerstand-gegen-veraenderung.md)
<br/>  Teams, die fehlgeschlagene Änderungen erlebt haben, entwickeln organisatorischen Widerstand gegen künftige Modifikationen und Modernisierungsbemühungen.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Übermäßige Vorsicht und bürokratische Genehmigungsprozesse, geboren aus vergangenen Fehlschlägen, verlangsamen das Entwicklungstempo.
- [Unfähigkeit zu innovieren](unfaehigkeit-zu-innovieren.md)
<br/>  Angst, die aus vergangenen Fehlschlägen entspringt, hindert Teams daran, neue Ansätze oder Technologien auszuprobieren.
- [Stagnierende Architektur](stagnierende-architektur.md)
<br/>  Zurückhaltung gegenüber Veränderung führt zu einer Architektur, die eingefroren bleibt und sich nicht mit sich ändernden Anforderungen weiterentwickeln kann.

## Causes ▼

- [Unzureichendes Testen](unzureichendes-testen.md)
<br/>  Unzureichendes Testen erlaubte es Defekten, in vergangenen Deployments die Produktion zu erreichen, was die Fehlschläge verursachte, die diese Angst schufen.
- [Große, riskante Releases](grosse-riskante-releases.md)
<br/>  Große, seltene Releases tragen ein höheres Fehlschlagsrisiko, und wenn sie fehlschlagen, ist die Auswirkung schwerwiegend genug, um anhaltende Angst zu schaffen.
- [Hohe Fehlerrate in Produktion](hohe-fehlerrate-in-produktion.md)
<br/>  Wiederholte Produktionsfehler aus Releases bauen eine Vorgeschichte fehlgeschlagener Änderungen auf, die eine risikoscheue Kultur verstärkt.
- [Fehlende Rollback-Strategie](fehlende-rollback-strategie.md)
<br/>  Ohne Rollback-Fähigkeit verursachen fehlgeschlagene Deployments verlängerte Ausfälle, die die negative Auswirkung und Angst verstärken.

## Detection Methods ○
- **Deployment-Häufigkeit:** Nachverfolgung, wie oft das Team Änderungen in die Produktion deployt. Eine niedrige Deployment-Häufigkeit kann ein Zeichen für Angst sein.
- **Durchlaufzeit für Änderungen:** Messung der Zeit vom Code-Commit bis zum Produktions-Deployment.
- **Änderungsfehlerrate:** Nachverfolgung des Prozentsatzes der Deployments, die zu einem Fehlschlag führen.
- **Entwicklerbefragungen:** Befragung von Entwicklern zu ihrem Vertrauen in den Deployment-Prozess und ihrer Bereitschaft, Änderungen vorzunehmen.

## Examples
Ein Team bei einem Finanzdienstleistungsunternehmen erlebte einen größeren Ausfall nach einem kürzlichen Deployment. Der Vorfall verursachte erhebliche finanzielle Verluste und Reputationsschäden. Infolgedessen implementierte das Unternehmen einen langwierigen und bürokratischen Änderungsgenehmigungsprozess. Jetzt erfordert selbst die kleinste Änderung mehrere Genehmigungsebenen und kann Wochen dauern, bis sie deployt wird. Die Entwickler haben so viel Angst, einen weiteren Ausfall zu verursachen, dass sie es vermeiden, überhaupt Änderungen vorzunehmen, es sei denn, sie sind absolut notwendig. Dies hat zu einem stagnierenden Produkt und einem frustrierten Entwicklungsteam geführt.
