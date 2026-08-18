---
title: Große, riskante Releases
description: Seltene Releases führen zu großen, komplexen Deployments, die schwer
  zu testen, fehleranfällig sind und erhebliche Auswirkungen auf Nutzer haben, wenn
  sie schiefgehen.
category:
- Code
- Operations
- Process
related_problems:
- slug: long-release-cycles
  similarity: 0.75
- slug: deployment-risk
  similarity: 0.75
- slug: complex-deployment-process
  similarity: 0.7
- slug: release-instability
  similarity: 0.7
- slug: large-pull-requests
  similarity: 0.65
- slug: manual-deployment-processes
  similarity: 0.65
solutions:
- blue-green-canary-deployments
- ci-cd-pipeline
- feature-flags
- canary-releases
- continuous-delivery
- continuous-integration-and-delivery
- dark-launches
- feature-toggles
- microservices
- microservices-architecture
- rollback-mechanisms
- rolling-updates
- trunk-based-development
- continuous-deployment
layout: problem
lang: de
en_slug: large-risky-releases
---

## Description
Große, riskante Releases sind ein verbreitetes Problem in Organisationen mit langen Release-Zyklen. Wenn Releases selten sind, tendieren sie dazu, groß und komplex zu sein. Dies liegt daran, dass sie eine große Anzahl von Änderungen enthalten, die auf unerwartete Weise interagieren können. Große Releases sind schwer zu testen und schlagen mit höherer Wahrscheinlichkeit fehl als kleine Releases. Wenn ein großes Release fehlschlägt, kann dies erhebliche Auswirkungen auf Nutzer und das Geschäft haben. Es kann auch schwierig und zeitaufwendig sein, ein großes Release zurückzurollen, was den Ausfall verlängern kann.

## Indicators ⟡
- Releases sind ein größeres Ereignis, das viel Planung und Koordination erfordert.
- Das Team ist ängstlich und gestresst bezüglich Deployments.
- Es gibt eine hohe Rate an Fehlern und anderen Problemen nach dem Deployment.
- Rollbacks sind ein häufiges Vorkommnis.

## Symptoms ▲

- [Release-Angst](release-angst.md)
<br/>  Der hohe Einsatz großer, seltener Releases erzeugt Stress und Angst beim Entwicklungsteam rund um Deployment-Ereignisse.
- [Release-Instabilität](release-instabilitaet.md)
<br/>  Große Releases mit vielen Änderungen sind inhärent weniger stabil, wobei mehr unerwartete Interaktionen Produktionsprobleme verursachen.
- [Häufige Hotfixes und Rollbacks](haeufige-hotfixes-und-rollbacks.md)
<br/>  Komplexe Releases mit vielen gebündelten Änderungen schlagen mit höherer Wahrscheinlichkeit fehl und erfordern Notfall-Hotfixes oder vollständige Rollbacks.
- [Systemausfälle](systemausfaelle.md)
<br/>  Fehlgeschlagene große Releases können erhebliche Serviceunterbrechungen aufgrund der Komplexität der Änderungen und der Schwierigkeit des Rollbacks verursachen.
- [Kundenunzufriedenheit](kundenunzufriedenheit.md)
<br/>  Große, riskante Releases, die fehlschlagen oder Fehler einführen, betreffen Nutzer direkt und verursachen Frustration und Unzufriedenheit.
- [Erhöhte Fehleranzahl](erhoehte-fehleranzahl.md)
<br/>  Große Releases mit vielen gebündelten Änderungen führen mit höherer Wahrscheinlichkeit Fehler ein, aufgrund komplexer Interaktionen zwischen Änderungen.

## Causes ▼

- [Lange Release-Zyklen](lange-release-zyklen.md)
<br/>  Seltene Releases bedeuten, dass sich zwischen Deployments mehr Änderungen anhäufen, was jedes Release größer und riskanter macht.
- [Komplexer Deployment-Prozess](komplexer-deployment-prozess.md)
<br/>  Manuelle, fehleranfällige Deployment-Prozesse entmutigen häufige Releases, was zur Bündelung von Änderungen in größere Releases führt.
- [Großer Feature-Umfang](grosser-feature-umfang.md)
<br/>  Features, die nicht in inkrementelle Liefergegenstände aufgeteilt werden können, zwingen dazu, mehrere große Änderungen zusammen zu releasen.
- [Manuelle Deployment-Prozesse](manuelle-deployment-prozesse.md)
<br/>  Wenn Deployments manuelles Eingreifen erfordern, vermeiden Teams häufiges Releasen, was dazu führt, dass sich Änderungen zu riskanten Batches anhäufen.

## Detection Methods ○
- **Release-Größe:** Nachverfolgung der Anzahl an Änderungen in jedem Release.
- **Release-Fehlerrate:** Nachverfolgung des Prozentsatzes der Releases, die zu einem kritischen Fehlschlag führen.
- **Mean Time to Recovery (MTTR):** Messung der durchschnittlichen Zeit, die zur Erholung von einem fehlgeschlagenen Release benötigt wird.
- **Anzahl der Fehler nach Release:** Zählung der Fehler, die in den Tagen und Wochen nach einem Release gemeldet werden.

## Examples
Ein Unternehmen veröffentlicht einmal jährlich eine neue Version seiner Software. Das jährliche Release ist ein größeres Ereignis, das Monate an Planung und Koordination erfordert. Das Release enthält eine große Anzahl neuer Features und Fehlerbehebungen. Der Testprozess ist lang und mühsam, aber es ist unmöglich, jede mögliche Kombination von Änderungen zu testen. Infolgedessen ist das Release immer riskant und schlägt oft fehl. Wenn das Release fehlschlägt, kann es Tage dauern, es zurückzurollen, was erhebliche Auswirkungen auf die Kunden des Unternehmens hat.
