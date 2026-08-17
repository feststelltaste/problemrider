---
title: Ständiges Feuerlöschen
description: Das Entwicklungsteam ist dauerhaft damit beschäftigt, Fehler zu beheben
  und dringende Probleme zu lösen, was wenig bis keine Zeit für neue Feature-Entwicklung
  lässt.
category:
- Code
- Process
related_problems:
- slug: development-disruption
  similarity: 0.7
- slug: constantly-shifting-deadlines
  similarity: 0.65
- slug: slow-feature-development
  similarity: 0.65
- slug: developer-frustration-and-burnout
  similarity: 0.65
- slug: time-pressure
  similarity: 0.65
- slug: frequent-changes-to-requirements
  similarity: 0.65
solutions:
- blameless-postmortems
- observability-and-monitoring
- technical-debt-backlog
- failover-mechanisms
- graceful-degradation
- heartbeat
- incident-management
- monitoring
- on-call-duty
- resilience
- root-cause-analysis
- security-incident-handling
- site-reliability-engineering-sre
- status-monitoring
- watchdog
- emergency-drills
- endpoint-detection-and-response
- error-budgets
- error-logs
- error-reporting-and-analysis
- incident-response-measures
- runbooks
- self-monitoring-and-diagnosis
- service-level-agreements
- service-level-indicators
- work-in-progress-limits
- production-readiness-criteria
- workaround-registry
- defect-triage-process
layout: problem
lang: de
en_slug: constant-firefighting
---

## Description
Ständiges Feuerlöschen, auch bekannt als "reaktive Entwicklung", ist ein Zustand, in dem ein Entwicklungsteam so sehr von dringender, ungeplanter Arbeit vereinnahmt ist, dass es wenig oder keine Zeit für geplante, proaktive Arbeit hat. Das Team befindet sich ständig im Krisenmodus und taumelt von einem Notfall zum nächsten. Dies ist eine höchst ineffiziente und stressige Arbeitsweise und ein klares Zeichen dafür, dass das System instabil und der Entwicklungsprozess kaputt ist.

## Indicators ⟡
- Der Großteil der Zeit des Teams wird für ungeplante Arbeit aufgewendet.
- Das Team wechselt häufig den Kontext zwischen unterschiedlichen dringenden Aufgaben.
- Es herrscht ein Gefühl von Chaos und Dringlichkeit in der täglichen Arbeit des Teams.
- Das Team verpasst durchgängig seine Termine für geplante Arbeit.

## Symptoms ▲

- [Entwicklerfrustration und Burnout](entwicklerfrustration-und-burnout.md)
<br/>  Dauerhafter Krisenmodus erschöpft Entwickler emotional und körperlich, was direkt zu Burnout führt.
- [Unfähigkeit zu innovieren](unfaehigkeit-zu-innovieren.md)
<br/>  Wenn die gesamte Zeit von dringenden Fixes verbraucht wird, bleibt keine Kapazität mehr, um Verbesserungen oder neue Ansätze zu erkunden.
- [Zunehmende technische Abkürzungen](zunehmende-technische-abkuerzungen.md)
<br/>  Unter ständiger Dringlichkeit nehmen Entwickler Abkürzungen, um Probleme schnell zu lösen, was mehr technische Schulden und künftige Brände erzeugt.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Geplante Feature-Arbeit wird durchgängig zugunsten dringender Fehlerbehebungen zurückgestellt, was die Lieferung neuer Funktionalität verlangsamt.
- [Qualitätsverschlechterung](qualitaetsverschlechterung.md)
<br/>  Überstürzte Fixes unter Krisenbedingungen führen oft zu neuen Problemen, was die Gesamtqualität des Systems im Laufe der Zeit verschlechtert.

## Causes ▼

- [Hohe Fehlerrate in Produktion](hohe-fehlerrate-in-produktion.md)
<br/>  Eine hohe Rate an Produktionsfehlern erzeugt den Strom dringender Probleme, der das Team im ständigen Feuerlösch-Modus hält.
- [Schlechte Testabdeckung](schlechte-testabdeckung.md)
<br/>  Ohne ausreichende Testabdeckung gelangen Fehler häufig in die Produktion, was die anhaltenden Notfälle erzeugt, die das Feuerlöschen antreiben.
- [Brüchige Codebasis](bruechige-codebasis.md)
<br/>  Eine brüchige Codebasis, bei der kleine Änderungen bestehende Funktionalität brechen, erzeugt einen ständigen Strom von Produktionsproblemen, die dringende Aufmerksamkeit erfordern.
- [Monitoring-Lücken](monitoring-luecken.md)
<br/>  Unzureichendes Monitoring bedeutet, dass Probleme nicht früh erkannt werden und zu Notfällen eskalieren, die sofortige Feuerlösch-Reaktionen erfordern.
- [Hohe technische Schulden](hohe-technische-schulden.md)
<br/>  Hohe technische Schulden führen dazu, dass Systeme häufiger ausfallen, was den Strom dringender Probleme erzeugt, der Teams im Feuerlösch-Modus hält.

## Detection Methods ○
- **Ungeplante Arbeit nachverfolgen:** Messung des Prozentsatzes der Teamzeit, der für ungeplante Arbeit aufgewendet wird. Wenn diese Zahl durchgängig hoch ist, ist das ein klares Zeichen für ein Problem.
- **Fehlerberichte analysieren:** Suche nach Mustern in Fehlerberichten. Treten dieselben Probleme immer wieder auf? Das ist ein Zeichen, dass das Team die Grundursachen der Probleme nicht angeht.
- **Team-Retrospektiven:** Befragung des Teams zu seiner Erfahrung mit Feuerlöschen. Fühlen sie sich überwältigt? Haben sie das Gefühl, Fortschritte zu machen?
- **Wichtige Metriken überwachen:** Nachverfolgung von Metriken wie Mean Time to Recovery (MTTR) und Change Failure Rate. Eine hohe MTTR und eine hohe Change Failure Rate sind beides Indikatoren für ein Team, das mit Feuerlöschen kämpft.

## Examples
Ein Team ist für die Wartung einer geschäftskritischen Anwendung verantwortlich. Die Anwendung ist alt und hat viele technische Schulden. Das Team verbringt den Großteil seiner Zeit damit, Produktionsprobleme zu beheben. Es wird ständig von geplanter Arbeit abgezogen, um sich mit Notfällen zu befassen. Infolgedessen kann es nie Fortschritte bei den langfristigen Verbesserungen machen, die die Anwendung stabiler machen würden. Das Team steckt in einem Teufelskreis aus Feuerlöschen fest und brennt zunehmend aus.
