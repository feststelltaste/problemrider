---
title: Release-Angst
description: Das Entwicklungsteam ist wegen Deployments ängstlich und gestresst,
  aufgrund des hohen Fehlerrisikos und des Drucks, es richtig zu machen.
category:
- Code
- Operations
- Process
related_problems:
- slug: release-instability
  similarity: 0.7
- slug: reviewer-anxiety
  similarity: 0.7
- slug: large-risky-releases
  similarity: 0.65
- slug: deployment-risk
  similarity: 0.6
- slug: fear-of-failure
  similarity: 0.6
- slug: history-of-failed-changes
  similarity: 0.6
solutions:
- blue-green-canary-deployments
- ci-cd-pipeline
- feature-flags
- canary-releases
- continuous-delivery
- dark-launches
- continuous-deployment
- small-change-batches
- parallel-run
- delivery-performance-metrics
- production-readiness-criteria
layout: problem
lang: de
en_slug: release-anxiety
---

## Description
Release-Angst ist das Gefühl von Stress und Furcht, das Entwickler erleben, wenn sie kurz davor stehen, eine neue Version ihrer Software zu deployen. Dies ist ein häufiges Problem in Organisationen mit schlechtem Release-Prozess und einer Schuldzuweisungskultur. Wenn Releases riskant sind und Fehlschläge häufig, ist es natürlich, dass Entwickler deswegen ängstlich sind. Diese Angst kann einen negativen Einfluss auf die Team-Moral und -Produktivität haben. Sie kann auch zu Zurückhaltung führen, neue Features zu veröffentlichen, was einen negativen Einfluss auf das Geschäft haben kann.

## Indicators ⟡
- Das Team ist am Release-Tag sichtbar gestresst und ängstlich.
- Es gibt viel Fingerzeigen und Schuldzuweisung, wenn etwas schiefgeht.
- Das Team zögert, riskante Aufgaben zu übernehmen.
- Es gibt einen allgemeinen Mangel an Vertrauen in den Release-Prozess.

## Symptoms ▲

- [Widerstand gegen Veränderung](widerstand-gegen-veraenderung.md)
<br/>  Angst vor Releases macht Teams zurückhaltend, Änderungen oder Verbesserungen vorzunehmen, und bevorzugt die Sicherheit des Status quo über das Risiko eines schlechten Deployments.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Deployment-Angst verursacht, dass Entwickler übermäßig testen, überpräparieren und zögern, was das gesamte Tempo der Feature-Lieferung verlangsamt.

## Causes ▼

- [Release-Instabilität](release-instabilitaet.md)
<br/>  Eine Historie instabiler Releases schafft berechtigte Angst und Furcht vor zukünftigen Deployments, während Teams erwarten, dass etwas schiefgeht.
- [Schuldzuweisungskultur](schuldzuweisungskultur.md)
<br/>  Wenn Fehlschläge mit Schuldzuweisung statt konstruktiver Analyse begegnet wird, werden Entwickler persönlich ängstlich, für Deployment-Probleme verantwortlich gemacht zu werden.
- [Manuelle Deployment-Prozesse](manuelle-deployment-prozesse.md)
<br/>  Manuelle, fehleranfällige Deployment-Prozesse erhöhen die Wahrscheinlichkeit menschlicher Fehler, was jedes Release zu einem Hochrisiko-Ereignis macht, das Angst erzeugt.
- [Schlechte Testabdeckung](schlechte-testabdeckung.md)
<br/>  Ohne angemessene Testabdeckung fehlt Teams das Vertrauen, dass ihre Änderungen korrekt funktionieren, was Angst darüber schürt, was in Produktion brechen könnte.
- [Große, riskante Releases](grosse-riskante-releases.md)
<br/>  Große, riskante Releases verursachen direkt Angst, weil größere Releases mehr potenzielle Fehlerpunkte haben.

## Detection Methods ○
- **Entwicklerbefragungen:** Befragung von Entwicklern zu ihren Gefühlen bezüglich des Release-Prozesses.
- **Team-Retrospektiven:** Diskussion der Gefühle des Teams zu Releases in Ihren Retrospektiven.
- **Verhalten am Release-Tag:** Beobachtung des Verhaltens des Teams am Release-Tag. Sind sie gestresst und ängstlich?
- **Bereitschaft zu releasen:** Ist das Team eifrig, neue Features zu veröffentlichen, oder zögert es?

## Examples
Ein Unternehmen hat eine Schuldzuweisungskultur. Wenn ein Release fehlschlägt, ist die erste Frage, die gestellt wird: „Wessen Schuld ist es?" Infolgedessen haben Entwickler Angst, Risiken einzugehen, und sind sehr ängstlich bezüglich Releases. Der Release-Prozess des Unternehmens ist auch sehr manuell und fehleranfällig, was die Angst nur verstärkt. Das Team hat eine lange Liste von Features, die es gerne veröffentlichen würde, hat aber Angst davor, aus Furcht vor Fehlschlag.
