---
title: Hohe Fehlerrate in Produktion
description: Eine große Anzahl von Fehlern wird nach einem Release in der Live-Umgebung
  entdeckt, was auf zugrunde liegende Probleme im Entwicklungs- und Qualitätssicherungsprozess
  hindeutet.
category:
- Business
- Code
related_problems:
- slug: insufficient-testing
  similarity: 0.8
- slug: high-bug-introduction-rate
  similarity: 0.7
- slug: inadequate-test-data-management
  similarity: 0.65
- slug: increased-bug-count
  similarity: 0.65
- slug: release-instability
  similarity: 0.65
- slug: high-technical-debt
  similarity: 0.6
solutions:
- definition-of-done
- test-coverage-strategy
- canary-releases
- dark-launches
- functional-tests
- incident-management
- monitoring
- root-cause-analysis
- security-tests
- smoke-testing
- error-budgets
- error-reporting-and-analysis
- vulnerability-scans
- change-impact-analysis
- parallel-run
- production-like-test-data
- production-readiness-criteria
- defect-triage-process
- delivery-performance-metrics
- exploratory-testing
- risk-quantification
layout: problem
lang: de
en_slug: high-defect-rate-in-production
---

## Description
Eine hohe Fehlerrate in Produktion ist ein klares Zeichen für ernsthafte Probleme mit der Qualität eines Produkts. Dies kann durch verschiedene Faktoren verursacht werden, von unzureichendem Testen und mangelhaften Code-Reviews bis zu einem fehlenden ordentlichen Release-Prozess. Wenn ein Produkt nicht gründlich getestet wird, hat es wahrscheinlich eine hohe Anzahl an Fehlern, was zu einem schlechten Nutzererlebnis, einem Vertrauensverlust und einer erheblichen Menge an Nacharbeit führen kann. Eine umfassende Teststrategie sollte eine Mischung aus automatisiertem und manuellem Testen umfassen und von Anfang an in den Entwicklungsprozess integriert sein. Investitionen in Tests sind Investitionen in die Qualität und Stabilität des Produkts.

## Indicators ⟡
- Die Anzahl der Fehlerberichte von Nutzern steigt.
- Das Team verbringt mehr Zeit mit dem Beheben von Fehlern als mit dem Bauen neuer Features.
- Das Team hat Angst, Änderungen an der Codebasis vorzunehmen, aus Furcht, etwas zu brechen.
- Das Team befindet sich ständig im Feuerlösch-Modus.
- Das Team hat eine niedrige Testabdeckung.

## Symptoms ▲

- [Häufige Hotfixes und Rollbacks](haeufige-hotfixes-und-rollbacks.md)
<br/>  Eine hohe Anzahl an Produktionsdefekten erfordert Notfall-Patches und Rollbacks, um die Servicestabilität wiederherzustellen.
- [Hohe Wartungskosten](hohe-wartungskosten.md)
<br/>  Ständige Fehlerbehebung in Produktion lenkt Entwicklungsressourcen von neuen Features ab und erhöht die gesamte Wartungslast.
- [Kundenunzufriedenheit](kundenunzufriedenheit.md)
<br/>  Nutzer erleben Fehler in der Live-Umgebung, was zu Frustration und Vertrauensverlust in das Produkt führt.
- [Geschichte fehlgeschlagener Änderungen](geschichte-fehlgeschlagener-aenderungen.md)
<br/>  Wiederholte Produktionsdefekte bauen eine Vorgeschichte problematischer Releases auf, die Angst vor künftigen Änderungen schafft.
- [Angst vor Veränderung](angst-vor-veraenderung.md)
<br/>  Wenn Releases häufig Fehler einführen, werden Entwickler zurückhaltend, Änderungen vorzunehmen, was die Entwicklungsgeschwindigkeit verlangsamt.

## Causes ▼

- [Unzureichendes Testen](unzureichendes-testen.md)
<br/>  Ohne ausreichende Testabdeckung gelangen Fehler, die vor dem Release hätten erfasst werden können, in die Produktion.
- [Große, riskante Releases](grosse-riskante-releases.md)
<br/>  Seltene, große Releases bündeln viele Änderungen zusammen, was die Erkennung von Defekten erschwert und das Risiko von Produktionsproblemen erhöht.
- [Unzureichendes Code-Review](unzureichendes-code-review.md)
<br/>  Ohne Peer-Review von Codeänderungen bleiben logische Fehler und Qualitätsprobleme unentdeckt, bevor sie die Produktion erreichen.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Angehäufte Abkürzungen und Komplexität machen die Codebasis brüchig und anfällig für unbeabsichtigte Nebeneffekte bei Änderungen.
- [Unzureichende Integrationstests](unzureichende-integrationstests.md)
<br/>  Fehlende Integrationstests erlauben es Fehlern auf Integrationsebene, die Produktion zu erreichen, was direkt zu einer hohen Fehlerrate beiträgt.

## Detection Methods ○

- **Fehlerverfolgungsmetriken:** Beobachtung von Metriken wie der Anzahl neuer Fehler pro Release, der Zeit bis zu ihrer Behebung und der Anzahl kritischer Fehler.
- **Retrospektiven:** Regelmäßige Team-Retrospektiven zur Diskussion jüngster Fehlschläge und Identifikation der Grundursachen.
- **Code-Abdeckungs-Analyse:** Nutzung von Werkzeugen zur Messung der Code-Abdeckung und Identifikation von Bereichen der Codebasis, die nicht gut getestet sind.
- **Nutzerfeedback-Analyse:** Systematische Sammlung und Analyse von Nutzerfeedback zur Identifikation häufiger Schmerzpunkte und wiederkehrender Probleme.
- **Testautomatisierungsberichte:** Analyse von Berichten aus automatisierten Testläufen zur Identifikation von Lücken oder Fehlschlägen.
- **Manuelle Testfall-Überprüfung:** Überprüfung manueller Testfälle zur Identifikation von Bereichen, in denen Automatisierung eingeführt oder die Abdeckung verbessert werden könnte.

## Examples
Ein Softwareunternehmen veröffentlicht eine neue Version seines Flaggschiffprodukts. Innerhalb von Stunden wird der Support-Desk mit Anrufen von Nutzern überflutet, die Abstürze und Datenverlust erleben. Das Entwicklungsteam ist gezwungen, rund um die Uhr zu arbeiten, um einen Patch zu veröffentlichen, und der Ruf des Unternehmens wird geschädigt. In einem anderen Fall verlässt sich ein Team stark auf manuelles Testen. Ein wichtiger Tester ist während eines Release-Zyklus im Urlaub, und ein kritischer Fehler in einem neuen Feature wird übersehen. Der Fehler gelangt in die Produktion und verursacht einen größeren Ausfall. Dieses Problem ist oft ein Zeichen dafür, dass ein Entwicklungsteam erhebliche "technische Schulden" angehäuft hat. Das Team ist so auf kurzfristige Termine fokussiert, dass es nicht in die langfristige Gesundheit seiner Codebasis und Entwicklungsprozesse investiert.

Ein neues Feature wird veröffentlicht, und sofort melden Nutzer, dass ein kritischer Workflow defekt ist. Die Untersuchung zeigt, dass zwar einzelne Komponenten getestet wurden, der End-to-End-Fluss mit mehreren Diensten aber nie in einer integrierten Umgebung getestet wurde. In einem anderen Fall nimmt ein Entwickler eine kleine Änderung an einer Utility-Funktion vor. Ohne Unit-Tests für diese Funktion merkt er nicht, dass sie einen Nebeneffekt hat, der einen anderen, scheinbar unzusammenhängenden Teil der Anwendung bricht, was zu einem Regressionsfehler in Produktion führt. Dieses Problem entspringt oft einer Kultur, die Geschwindigkeit über Qualität priorisiert, oder mangelndem Verständnis der langfristigen Vorteile einer robusten Teststrategie. Es kann zu erheblichen technischen Schulden und einem ständigen Zustand des Feuerlöschens führen.
